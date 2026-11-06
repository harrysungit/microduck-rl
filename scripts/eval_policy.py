"""Evaluate trained checkpoints deterministically on a task's training environment.

Runs each checkpoint's actor mean (no exploration noise) on the same worlds
(same seed: same startup randomization, same command and push sequence) for
one episode length, with the task's curricula at the same stage for every
checkpoint: --curriculum-step sets the env step counter they key on (default:
the end of a 10,000-iteration run, every stage reached). Loading a .pt would
otherwise restore that checkpoint's own counter, and an ONNX would start at 0. A checkpoint is an rsl_rl ``model_<N>.pt`` or an exported
``.onnx`` policy with the normalization in the graph (run on the CPU with
onnxruntime, as on the robot). Reports per checkpoint:
- falls: terminations other than the time-out, per world and episode-second;
- fell_share: the share of worlds that fell at least once;
- velocity tracking: mean |commanded - actual| planar speed (m/s) and yaw rate
  (rad/s) in the body frame, over worlds that are up (tasks with a twist command);
- posture, for tasks that do not end an episode on a fall (VelStand, StandUp):
  the share of world-steps upright (trunk tilt below 30 degrees), tip-overs
  (upright to above 60 degrees) and recoveries (back below 30) per world-minute;
- every reward term's mean per step, weighted and raw, and every termination
  term's rate per world-minute.

    python scripts/eval_policy.py --task Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910 \
        --checkpoints <run>/model_1000.pt <run>/model_2000.pt --njmax 128 --out <file.json>
"""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

UP_DEG, DOWN_DEG = 30.0, 60.0


def onnx_policy(path: Path, device: str):
    """An exported policy as a batched callable; the export fixes the batch at 1."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path))
    for value in (*model.graph.input, *model.graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def policy(obs):
        x = obs["actor"].detach().cpu().numpy().astype(np.float32)
        return torch.from_numpy(session.run(None, {name: x})[0]).to(device)

    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", help="one per checkpoint; default: the iteration")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--curriculum-step", type=int, default=10_000 * 24,
                    help="env step counter the curricula see (iterations x steps per env)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    configure_torch_backends()
    device = "cuda:0"
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.njmax:
        env_cfg.sim.njmax = args.njmax
    agent_cfg = load_rl_cfg(args.task)
    try:
        base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    except ValueError as e:
        # mujoco_warp sizes the budget from the initial MjData too; on rough terrain its rows depend on the
        # seed (552 with seed 0 on VelStand-Rough-Backlash, where the robot's default pose meets the terrain).
        m = re.search(r"njmax must be >= (\d+)", str(e))
        if not m:
            raise
        env_cfg.sim.njmax = int(m.group(1)) + 64
        print(f"initial state needs {m.group(1)} constraint rows; njmax {env_cfg.sim.njmax}", flush=True)
        base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    robot = base.scene["robot"]
    steps = int(args.seconds / base.step_dt)
    has_twist = "twist" in base.command_manager.active_terms
    reward_names = list(base.reward_manager.active_terms)
    term_names = list(base.termination_manager.active_terms)

    rows = []
    labels = args.labels or [None] * len(args.checkpoints)
    for ckpt, label in zip(args.checkpoints, labels):
        if ckpt.suffix == ".onnx":
            policy = onnx_policy(ckpt, device)
        else:
            runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
            policy = runner.get_inference_policy(device=device)
        base.common_step_counter = args.curriculum_step  # after any .pt load, which restores its own
        torch.manual_seed(args.seed)
        env.reset()
        falls = torch.zeros(args.num_envs, device=device)
        err_xy = err_wz = 0.0
        counted = 0
        reward_sum = torch.zeros(len(reward_names), device=device)
        term_count = torch.zeros(len(term_names), device=device)
        posture = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)  # 1 up, 0 down, -1 between
        tip_overs = torch.zeros(args.num_envs, device=device)
        recoveries = torch.zeros(args.num_envs, device=device)
        upright = torch.zeros(args.num_envs, device=device)
        with torch.no_grad():
            for _ in range(steps):
                cmd = base.command_manager.get_command("twist").clone() if has_twist else None
                _, _, dones, extras = env.step(policy(env.get_observations()))
                up = ~dones.bool()
                fallen = dones.bool() & ~extras["time_outs"].bool()
                falls += fallen.float()
                reward_sum += base.reward_manager._step_reward.mean(dim=0)  # weighted, not scaled by dt
                term_count += torch.stack([base.termination_manager.get_term(n).float().sum() for n in term_names])
                tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
                is_up, is_down = tilt < UP_DEG, tilt > DOWN_DEG
                fresh = dones.bool()  # reset this step: the pose belongs to the next episode
                tip_overs += ((posture == 1) & is_down & ~fresh).float()
                recoveries += ((posture == 0) & is_up & ~fresh).float()
                posture = torch.where(is_up, 1, torch.where(is_down, 0, torch.where(fresh, -1, posture)))
                upright += is_up.float()
                if not has_twist:
                    continue
                v = robot.data.root_link_lin_vel_b[:, :2]
                w = robot.data.root_link_ang_vel_b[:, 2]
                if up.any():  # at the time-out every world resets on the same step
                    err_xy += (cmd[:, :2] - v).norm(dim=-1)[up].mean().item()
                    err_wz += (cmd[:, 2] - w).abs()[up].mean().item()
                    counted += 1
        err_xy, err_wz = err_xy / max(counted, 1), err_wz / max(counted, 1)
        per_min = 60.0 / args.seconds
        weights = [base.reward_manager.get_term_cfg(n).weight for n in reward_names]
        reward_mean = (reward_sum / steps).tolist()
        m = re.search(r"model_(\d+)\.pt$", ckpt.name)
        row = {"checkpoint": str(ckpt), "label": label or (m.group(1) if m else ckpt.stem),
               "iteration": int(m.group(1)) if m else None,
               "falls_per_world_minute": falls.mean().item() * per_min,
               "fell_share": (falls > 0).float().mean().item(),
               "err_vxy_mps": err_xy if has_twist else None, "err_wz_radps": err_wz if has_twist else None,
               "upright_share": upright.mean().item() / steps,
               "tip_overs_per_world_minute": tip_overs.mean().item() * per_min,
               "recoveries_per_world_minute": recoveries.mean().item() * per_min,
               "reward_per_step": dict(zip(reward_names, reward_mean)),
               "reward_raw_per_step": {n: v / w for n, v, w in zip(reward_names, reward_mean, weights) if w},
               "terminations_per_world_minute": {n: c * per_min / args.num_envs
                                                 for n, c in zip(term_names, term_count.tolist())}}
        rows.append(row)
        tracking = f", |v_xy err| {err_xy:.3f} m/s, |w_z err| {err_wz:.3f} rad/s" if has_twist else ""
        print(f"{row['label']:>16}: falls/world/min {row['falls_per_world_minute']:.3f}, fell {row['fell_share']:.1%}, "
              f"upright {row['upright_share']:.1%}, tip-overs/world/min {row['tip_overs_per_world_minute']:.3f}, "
              f"recoveries/world/min {row['recoveries_per_world_minute']:.3f}, reward/step {sum(reward_mean):.3f}"
              f"{tracking}", flush=True)
    env.close()
    if args.out:
        args.out.write_text(json.dumps({"task": args.task, "num_envs": args.num_envs, "seconds": args.seconds,
                                        "seed": args.seed, "curriculum_step": args.curriculum_step,
                                        "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
