"""Record a seeded rollout of a task, or compare recorded rollouts.

Steps the environment with a fixed seed and a seeded sequence of random actions
and saves the policy observations and rewards of every step, so two runs that
differ in one setting (for example the Warp allocator) can be compared. The GPU
does not reproduce a rollout bit for bit, so compare a change against two runs
of the baseline: the change is clean if it moves the rollout no more than the
baseline moves itself.

    MICRODUCK_HIP_CACHING_ALLOCATOR=0 python rollout_digest.py record a1.npz
    MICRODUCK_HIP_CACHING_ALLOCATOR=0 python rollout_digest.py record a2.npz
    python rollout_digest.py record b.npz
    python rollout_digest.py compare a1.npz a2.npz b.npz
"""

import argparse
import sys

import numpy as np


def record(args) -> None:
    import torch
    import warp as wp  # noqa: F401  # after torch, so the ROCm libraries bundled with torch load first

    import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    obs, _ = env.reset()
    gen = torch.Generator(device="cuda:0").manual_seed(args.seed)
    observations, rewards = [], []
    for _ in range(args.steps):
        action = args.action_noise * torch.randn(env.action_space.shape, device="cuda:0", generator=gen)
        obs, reward, *_ = env.step(action)
        observations.append(obs["actor"].cpu().numpy())
        rewards.append(reward.cpu().numpy())
    np.savez_compressed(args.file, obs=np.stack(observations), reward=np.stack(rewards))
    print(f"recorded {args.steps} steps x {args.num_envs} envs -> {args.file}")


def compare(args) -> None:
    runs = [np.load(f) for f in args.files]
    base = runs[0]
    for f, run in zip(args.files[1:], runs[1:]):
        d = np.abs(run["obs"] - base["obs"]).max(axis=(1, 2))
        r = np.abs(run["reward"] - base["reward"]).max(axis=1)
        finite = bool(np.isfinite(run["obs"]).all() and np.isfinite(run["reward"]).all())
        print(f"{f} vs {args.files[0]}: finite {finite}; max |obs diff| at steps 1/10/50/last "
              f"{d[0]:.1e} {d[min(9, len(d) - 1)]:.1e} {d[min(49, len(d) - 1)]:.1e} {d[-1]:.1e}; "
              f"max |reward diff| {r.max():.1e}; envs with obs diff > 1e-2 at the end: "
              f"{int((np.abs(run['obs'][-1] - base['obs'][-1]).max(axis=1) > 1e-2).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("file")
    rec.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    rec.add_argument("--num-envs", type=int, default=1024)
    rec.add_argument("--steps", type=int, default=100)
    rec.add_argument("--seed", type=int, default=0)
    rec.add_argument("--action-noise", type=float, default=0.5)
    cmp = sub.add_parser("compare")
    cmp.add_argument("files", nargs="+", help="baseline first")
    args = ap.parse_args()
    {"record": record, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
