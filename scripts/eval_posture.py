"""Does a sitstand policy lower its trunk when told to sit, in the training world?

Runs policies in the task's own mjlab environment (same worlds, same randomization and curricula as the training) with
the posture command scripted instead of drawn: the same stand / sit / stand / sit / stand schedule as the MuJoCo C check
(sim2sim.py --task sitstand), from a standing or a seated start, with the same pass criterion (sim2sim.posture_segments:
trunk at most 7.5 cm high for a sit, at least 10 cm for a stand, within 30 degrees of upright, held to the end of the
segment). eval_policy.py cannot answer this: its commands are random, and no metric there says whether the trunk lowers.

Variants of the training world, from the training's own to the MuJoCo C check's clean levels:
- train: as trained: pushes, observation noise and delay, head commands, randomization, seated-only kicks;
- no-disturb: no pushes, no observation noise or delay, head command zero, no seated-only kicks; randomization stays;
- nominal: no-disturb without IMU misalignment and encoder bias.

    python scripts/eval_posture.py --policy ours=<policy.onnx> pt=<model.pt> --out-dir <dir>
"""

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from eval_policy import onnx_policy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import sim2sim  # noqa: E402  # the schedules, thresholds and pass criterion, shared with the MuJoCo C check

TASK = "Mjlab-SitStand-Flat-MicroDuck-HD1910"
VARIANTS = ("train", "no-disturb", "nominal")
SEATED_KICKS = ("seated_head_jerk", "seated_tip")
PUSH_CURRICULA = ("push_magnitude", "seated_head_jerk_prob", "seated_tip_range", "head_pose_range")


def make_cfg(variant: str, start: str, args) -> tuple:
    schedule = sim2sim.SIT_FROM_SEAT if start == "seated" else sim2sim.SIT_FROM_STAND
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.njmax = args.njmax
    cfg.episode_length_s = sum(d for d, _, _ in schedule) + 5.0  # the training's 12 s would end the schedule
    cfg.commands["twist"].resampling_time_range = (1e9, 1e9)  # the posture flag is written by this script
    spawn = cfg.events["set_ground_state"].params
    spawn.update(sitting_prob=float(start == "seated"), standing_prob=float(start != "seated"))
    if variant in ("no-disturb", "nominal"):
        cfg.observations["actor"].enable_corruption = False
        for name in ("push_robot", *SEATED_KICKS):
            cfg.events.pop(name)
        for name in PUSH_CURRICULA:  # they drive the events and the head command range removed here
            cfg.curriculum.pop(name)
        cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
    if variant == "nominal":
        for term in ("base_ang_vel", "projected_gravity"):
            terms = cfg.observations["actor"].terms
            key = term if term in terms else next(k for k in terms if "gravity" in k)
            terms[key].params = {**(terms[key].params or {}), "max_angle_deg": 0.0}
        cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    return cfg, schedule


def build_env(cfg, args, device):
    try:
        base = ManagerBasedRlEnv(cfg=cfg, device=device)
    except ValueError as e:
        m = re.search(r"njmax must be >= (\d+)", str(e))
        if not m:
            raise
        cfg.sim.njmax = int(m.group(1)) + 64
        print(f"initial state needs {m.group(1)} constraint rows; njmax {cfg.sim.njmax}", flush=True)
        base = ManagerBasedRlEnv(cfg=cfg, device=device)
    return base


def rollout(base, env, policy, schedule, args, device) -> dict:
    """Trunk height and tilt of every world at every control step, under the scripted posture command."""
    term = base.command_manager.get_term("twist")
    robot = base.scene["robot"]
    dt = base.step_dt
    steps = round(sum(d for d, _, _ in schedule) / dt)
    z = torch.zeros(steps, args.num_envs, device=device)
    tilt = torch.zeros_like(z)
    reset = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    base.common_step_counter = args.curriculum_step
    torch.manual_seed(args.seed)
    env.reset()
    with torch.no_grad():
        for k in range(steps):
            term.vel_command_b[:, 0] = float(sim2sim.command_at(k * dt, schedule)[0])
            _, _, dones, _ = env.step(policy(env.get_observations()))
            reset |= dones.bool()
            z[k] = robot.data.root_link_pos_w[:, 2] - base.scene.terrain.env_origins[:, 2]
            tilt[k] = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)))
    return {"z": z.cpu().numpy(), "tilt": tilt.cpu().numpy(), "reset": reset.cpu().numpy(), "dt": dt}


def summarize(run: dict, schedule: list) -> dict:
    z, tilt, dt = run["z"], run["tilt"], run["dt"]
    keep = np.flatnonzero(~run["reset"])  # a world that reset mid-schedule has no schedule to score
    t = (np.arange(len(z)) + 1) * dt
    zeros = np.zeros(len(t))
    per_env = [sim2sim.posture_segments({"t": t, "z": z[:, i], "tilt": tilt[:, i], "vx": zeros, "vy": zeros}, schedule)
               for i in keep]
    segs = [s for env_segs in per_env for s in env_segs]

    def block(kind):
        rows = [s for s in segs if s["transition"] == kind]
        settle = [s["settled_s"] for s in rows if s["ok"]]
        return {"reached": int(sum(s["ok"] for s in rows)), "tries": len(rows),
                "settle_median_s": float(np.median(settle)) if settle else None,
                "settle_p90_s": float(np.percentile(settle, 90)) if settle else None}

    sit_z = [s["z_end"] for s in segs if s["command"][0] > 0.5 and "z_end" in s]
    stand_z = [s["z_end"] for s in segs if s["command"][0] < 0.5 and "z_end" in s]
    return {"worlds": int(len(keep)), "reset_midway": int(run["reset"].sum()),
            "sit_down": block("sit-down"), "stand_up": block("stand-up"), "hold": block("hold"),
            "sit_z_end_median_cm": 100 * float(np.median(sit_z)), "stand_z_end_median_cm": 100 * float(np.median(stand_z)),
            "tipped_over": int((tilt[:, keep] > sim2sim.FELL_DEG).any(axis=0).sum()),
            "curve": {"t": t[::10].round(2).tolist(),  # 5 Hz is enough for the plot
                      "median": np.median(z[:, keep], axis=1)[::10].round(5).tolist(),
                      "p10": np.percentile(z[:, keep], 10, axis=1)[::10].round(5).tolist(),
                      "p90": np.percentile(z[:, keep], 90, axis=1)[::10].round(5).tolist()}}


def pct(block: dict) -> str:
    return f"{block['reached']} of {block['tries']} ({block['reached'] / max(block['tries'], 1):.0%})"


def settle(block: dict) -> str:
    if block["settle_median_s"] is None:
        return "-"
    return f"{block['settle_median_s']:.1f} / {block['settle_p90_s']:.1f}"


def write_report(out: Path, rows: list[dict], args) -> None:
    lines = ["# SitStand in the training world: does the trunk lower when told to sit?", "",
             f"Task `{args.task}`, {args.num_envs} worlds per row, seed {args.seed}, curricula at the end stage "
             f"(step {args.curriculum_step}), actor mean. The posture command is scripted, not drawn: "
             "stand 3 s, sit 5 s, stand 5 s, sit 5 s, stand 5 s from a standing start; sit 3 s, stand 5 s, sit 5 s, "
             "stand 5 s from a seated start. Same schedule and same pass criterion as the MuJoCo C check "
             f"(`sim2sim.py --task sitstand`): trunk at most {sim2sim.SIT_Z_MAX * 100:g} cm high for a sit, at least "
             f"{sim2sim.STAND_Z_MIN * 100:g} cm for a stand, within {sim2sim.UPRIGHT_DEG:g} degrees of upright, held "
             "to the end of the segment.", "",
             "Variants: **train** is the training world as trained (pushes, observation noise and delay, head commands, "
             "randomization, seated-only kicks); **no-disturb** has no pushes, observation noise or delay, head "
             "command zero, no seated-only kicks, randomization kept; **nominal** also has no IMU misalignment and "
             "no encoder bias.", "",
             "| variant | start | policy | worlds | sit-down reached | settle median / p90 (s) | stand-up reached | "
             "trunk height at the end of a sit, median (cm) | tipped over |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["summary"]
        lines.append(f"| {r['variant']} | {r['start']} | {r['policy']} | {s['worlds']} | {pct(s['sit_down'])} | "
                     f"{settle(s['sit_down'])} | {pct(s['stand_up'])} | {s['sit_z_end_median_cm']:.1f} | "
                     f"{s['tipped_over']} |")
    lines += ["", "A sit-down is a sit segment after a stand, a stand-up a stand segment after a sit. Worlds that reset "
              "mid-schedule are left out of the counts.", "", "![height](posture_z_vs_time.png)", ""]
    (out / "README.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    standing = [r for r in rows if r["start"] == "standing"]
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in standing)]
    if not variants:
        return
    fig, axes = plt.subplots(len(variants), 1, figsize=(11, 3.2 * len(variants)), sharex=True, squeeze=False)
    for ax, variant in zip(axes[:, 0], variants):
        rr = [r for r in standing if r["variant"] == variant]
        t = rr[0]["summary"]["curve"]["t"]
        target = [sim2sim.SIT_Z if sim2sim.command_at(x - 1e-6, sim2sim.SIT_FROM_STAND)[0] > 0.5 else sim2sim.STAND_Z
                  for x in t]
        ax.plot(t, target, "k--", lw=1, label="training rest height for the command")
        for r in rr:
            c = r["summary"]["curve"]
            ax.plot(c["t"], c["median"], lw=1.3, label=f"{r['policy']} (median)")
            ax.fill_between(c["t"], c["p10"], c["p90"], alpha=0.15)
        ax.axhline(sim2sim.SIT_Z_MAX, color="0.6", lw=0.8, ls=":")
        ax.axhline(sim2sim.STAND_Z_MIN, color="0.6", lw=0.8, ls=":")
        ax.set_ylabel("trunk height (m)")
        ax.set_title(f"{variant}: median and 10-90% band over worlds", fontsize="small")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="center right", fontsize="small")
    axes[-1, 0].set_xlabel("time (s); dotted: the reach thresholds")
    fig.tight_layout()
    fig.savefig(out / "posture_z_vs_time.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--policy", nargs="+", required=True, metavar="LABEL=PATH", help="an rsl_rl .pt or an exported .onnx")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--seated-policies", nargs="*", default=[], metavar="LABEL",
                    help="policies that also run from the seated start, in the train variant")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--njmax", type=int, default=256)
    ap.add_argument("--curriculum-step", type=int, default=10_000 * 24)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = dict(p.split("=", 1) for p in args.policy)

    configure_torch_backends()
    device = "cuda:0"
    agent_cfg = load_rl_cfg(args.task)
    plan = [(v, "standing", list(policies)) for v in args.variants]
    if args.seated_policies:
        plan.append(("train", "seated", args.seated_policies))
    rows = []
    for variant, start, labels in plan:
        cfg, schedule = make_cfg(variant, start, args)
        base = build_env(cfg, args, device)
        env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
        for label in labels:
            path = Path(policies[label])
            if path.suffix == ".onnx":
                policy = onnx_policy(path, device)
            else:
                runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
                policy = runner.get_inference_policy(device=device)
            t0 = time.time()
            summary = summarize(rollout(base, env, policy, schedule, args, device), schedule)
            rows.append({"variant": variant, "start": start, "policy": label, "summary": summary})
            print(f"{variant:<11} {start:<8} {label:<14} sit-down {pct(summary['sit_down'])}, stand-up "
                  f"{pct(summary['stand_up'])}, sit height {summary['sit_z_end_median_cm']:.1f} cm, "
                  f"tipped {summary['tipped_over']}, reset midway {summary['reset_midway']} ({time.time() - t0:.0f} s)",
                  flush=True)
        env.close()
    (args.out_dir / "summary.json").write_text(json.dumps(
        {"task": args.task, "num_envs": args.num_envs, "seed": args.seed, "curriculum_step": args.curriculum_step,
         "policies": policies, "rows": rows}, indent=1))
    write_report(args.out_dir, rows, args)


if __name__ == "__main__":
    main()
