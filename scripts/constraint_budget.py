"""Measure how much of its contact and constraint budget a task actually uses.

Steps the task's environment with random actions, so falls and resets are
included, and records per step the largest number of constraint rows in any
world (mujoco_warp Data.nefc) and the number of contacts across all worlds
(Data.nacon). The task's budgets, njmax constraint rows per world and nconmax
contacts per world, size mujoco_warp's scratch buffers, so a budget far above
use costs memory traffic on every step.

    python scripts/constraint_budget.py --num-envs 4096 --steps 500 --out <file.json>
"""

import argparse
import json

import numpy as np
import torch
import warp as wp  # noqa: F401  # after torch, so the ROCm libraries bundled with torch load first

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--action-noise", type=float, default=0.5)
    ap.add_argument("--out")
    args = ap.parse_args()

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env.reset()
    data = env.sim._wp_data
    njmax, nconmax = data.njmax, env_cfg.sim.nconmax

    nefc_max, nacon = [], []
    for _ in range(args.steps):
        env.step(args.action_noise * torch.randn(env.action_space.shape, device="cuda:0"))
        nefc_max.append(int(data.nefc.numpy().max()))
        nacon.append(int(data.nacon.numpy()[0]))
    nefc_max, nacon = np.array(nefc_max), np.array(nacon)

    result = {
        "task": args.task, "num_envs": args.num_envs, "steps": args.steps, "action_noise": args.action_noise,
        "njmax": njmax, "nconmax_per_world": nconmax,
        "nefc_max_any_world": int(nefc_max.max()), "nefc_max_p50_over_steps": float(np.median(nefc_max)),
        "nacon_total_max": int(nacon.max()), "nacon_per_world_max_avg": float(nacon.max() / args.num_envs),
        "nacon_budget_total": data.naconmax,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
