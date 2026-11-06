"""Hold the HOME pose with zero actions: can the servos carry the robot?

Builds each task's environment without pushes, commands zero actions (every
servo targets HOME) and steps for a few seconds. Reports the share of worlds
that fell (terminated before the time-out), the trunk height, and how far the
servo joints sag from HOME under gravity at the end; a soft firmware gain or
an underpowered actuator model shows up as sag and falls before any training.

    python scripts/hold_home.py --tasks Mjlab-Velocity-Flat-MicroDuck \
        Mjlab-Velocity-Flat-MicroDuck-HD1910 --njmax 128 --out <file.json>
"""

import argparse
import json

import torch

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def hold(task: str, num_envs: int, seconds: float, njmax: int | None) -> dict:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    if njmax:
        cfg.sim.njmax = njmax
    for name in [n for n in cfg.events if "push" in n]:
        del cfg.events[name]
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    servo = [i for i, n in enumerate(robot.joint_names) if not n.startswith("passive_")]
    actions = torch.zeros(env.action_space.shape, device="cuda:0")
    steps = int(seconds / env.step_dt)
    fell = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
    sag, height = [], []
    for k in range(steps):
        _, _, terminated, _, _ = env.step(actions)
        fell |= terminated
        if k >= steps - 50:
            err = (robot.data.joint_pos[:, servo] - robot.data.default_joint_pos[:, servo]).abs()
            sag.append(err[~fell].mean().item() if (~fell).any() else float("nan"))
            height.append(robot.data.root_link_pos_w[~fell, 2].mean().item() if (~fell).any() else float("nan"))
    env.close()
    return {"task": task, "num_envs": num_envs, "seconds": seconds, "fell_share": fell.float().mean().item(),
            "servo_sag_rad_mean": sum(sag) / len(sag), "trunk_height_m": sum(height) / len(height)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    for task in args.tasks:
        rows.append(hold(task, args.num_envs, args.seconds, args.njmax))
        r = rows[-1]
        print(f"{task:<50} fell {r['fell_share']:6.1%}  sag {r['servo_sag_rad_mean']:.4f} rad  "
              f"trunk {r['trunk_height_m']:.4f} m", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
