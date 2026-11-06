"""Drive the training world (MJWarp) through the sim2sim command schedule: the reference for level L0.

The task the policy was trained on, one world, with what sim2sim's nominal
levels leave out switched off: pushes and every randomization event,
observation noise, IMU misalignment and IMU delays (joint speeds stay one
control step late, as in training). The actuator latency is fixed at 4
physics steps (20 ms) and the supply at 7.7 V with sag gain 0.1, as in L0.
The schedule's command goes into the twist command every step; the head and
body commands are zero. Writes reference_traces.csv in sim2sim.py's format.

    python scripts/sim2sim_reference.py --onnx <policy.onnx> --out <dir>/reference_traces.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import onnx_policy  # noqa: E402
from sim2sim import CONTROL_DT, DURATION, FELL_DEG, command_at  # noqa: E402

KEEP_EVENTS = {"reset_base", "reset_robot_joints", "reset_action_history", "expand_bam_friction_fields"}


def reference_cfg(task: str, njmax: int):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.sim.njmax = njmax
    cfg.episode_length_s = DURATION + 5.0
    for name in [n for n in cfg.events if n not in KEEP_EVENTS]:
        del cfg.events[name]
    cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.125, 0.125),
                                                      "yaw": (0.0, 0.0)}
    actor = cfg.observations["actor"]
    actor.enable_corruption = False
    for name in ("base_ang_vel", "projected_gravity"):
        actor.terms[name].params["max_angle_deg"] = 0.0
        actor.terms[name].delay_min_lag = actor.terms[name].delay_max_lag = 0
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = twist.rel_forward_envs = twist.rel_turn_in_place_envs = 0.0
    for name in ("head_pose", "body_pose"):
        cfg.commands[name].resampling_time_range = (1e9, 1e9)
        cfg.commands[name].ranges = tuple((0.0, 0.0) for _ in cfg.commands[name].ranges)
    cfg.curriculum.clear()
    (actuator,) = cfg.scene.entities["robot"].articulation.actuators
    actuator.delay_min_lag = actuator.delay_max_lag = 4
    actuator.vin_range, actuator.vin_drop_gain_range = (7.7, 7.7), (0.1, 0.1)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck-HD1910")
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--njmax", type=int, default=160)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda:0"
    base = ManagerBasedRlEnv(cfg=reference_cfg(args.task, args.njmax), device=device)
    env = RslRlVecEnvWrapper(base, clip_actions=load_rl_cfg(args.task).clip_actions)
    policy = onnx_policy(args.onnx, device)
    robot = base.scene["robot"]
    twist = base.command_manager.get_term("twist")
    obs, _ = env.reset()
    n = base.action_manager.action.shape[1]
    keys = ["t", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "z", "tilt"] + [f"a{i}" for i in range(n)]
    rows = []
    with torch.no_grad():
        for k in range(round(DURATION / CONTROL_DT)):
            command = torch.tensor(command_at(k * CONTROL_DT), dtype=torch.float32, device=device)
            twist.vel_command_b[:] = command
            twist.is_standing_env[:] = False
            obs["actor"][:, 48:51] = command  # the observation was taken before this tick's command was set
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
            g = robot.data.projected_gravity_b[0]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2].item()))))
            v, w = robot.data.root_link_lin_vel_b[0], robot.data.root_link_ang_vel_b[0]
            rows.append([(k + 1) * CONTROL_DT, *command.tolist(), v[0].item(), v[1].item(), w[2].item(),
                         robot.data.root_link_pos_w[0, 2].item(), tilt, *action[0].tolist()])
            if dones[0] or tilt > FELL_DEG:
                print(f"fell or reset at {(k + 1) * CONTROL_DT:.2f} s")
                break
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows([f"{v:.6g}" for v in row] for row in rows)
    print(f"Written {args.out}: {len(rows)} control steps")


if __name__ == "__main__":
    main()
