"""Break mjlab's environment reset (``_reset_idx``) down by sub-call and event term.

Builds a task's environment with N worlds and resets a fixed fraction of them
repeatedly, timing with a device synchronisation on both sides each call that
``_reset_idx`` makes (curriculum, simulation reset, scene reset, reset events,
manager resets) and each reset event term on its own.

    python scripts/profile_reset.py --num-envs 4096 --fraction 0.05 --reps 10
"""

import argparse
import json
import statistics
import time
from collections import defaultdict

import torch  # before warp, so the ROCm libraries bundled with torch load first
import warp as wp

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

CALLS = [
    ("curriculum_manager", "compute"), ("sim", "reset"), ("scene", "reset"), ("event_manager", "apply"),
    ("observation_manager", "reset"), ("action_manager", "reset"), ("reward_manager", "reset"),
    ("metrics_manager", "reset"), ("curriculum_manager", "reset"), ("command_manager", "reset"),
    ("event_manager", "reset"), ("termination_manager", "reset"),
]


def sync() -> None:
    torch.cuda.synchronize()
    wp.synchronize_device("cuda:0")


def timed(fn, bucket, key):
    def wrapper(*a, **k):
        sync()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sync()
            bucket[key].append(time.perf_counter() - t)
    return wrapper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--njmax", type=int)
    ap.add_argument("--fraction", type=float, default=0.05, help="share of the worlds reset per call")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    if args.njmax:
        cfg.sim.njmax = args.njmax
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()

    calls: dict[str, list[float]] = defaultdict(list)
    terms: dict[str, list[float]] = defaultdict(list)
    for owner, name in CALLS:
        obj = getattr(env, owner)
        setattr(obj, name, timed(getattr(obj, name), calls, f"{owner}.{name}"))
    em = env.event_manager
    for tname, tcfg in zip(em._mode_term_names.get("reset", []), em._mode_term_cfgs.get("reset", [])):
        tcfg.func = timed(tcfg.func, terms, tname)

    n = max(1, int(args.fraction * args.num_envs))
    env_ids = torch.randperm(args.num_envs, device="cuda:0")[:n]
    env._reset_idx(env_ids)  # warm-up
    calls.clear()
    terms.clear()
    totals = []
    for _ in range(args.reps):
        sync()
        t = time.perf_counter()
        env._reset_idx(env_ids)
        sync()
        totals.append(time.perf_counter() - t)

    ms = lambda v: sum(v) / args.reps * 1e3  # noqa: E731
    result = {
        "task": args.task, "num_envs": args.num_envs, "reset_envs": n, "njmax": env.sim._wp_data.njmax,
        "reset_ms": statistics.mean(totals) * 1e3,
        "calls_ms": dict(sorted(((k, ms(v)) for k, v in calls.items()), key=lambda kv: -kv[1])),
        "event_terms_ms": dict(sorted(((k, ms(v)) for k, v in terms.items()), key=lambda kv: -kv[1])),
    }
    print(f"reset of {n} of {args.num_envs} worlds: {result['reset_ms']:.1f} ms")
    for k, v in result["calls_ms"].items():
        print(f"  {k:<32} {v:8.2f} ms")
    print("reset event terms:")
    for k, v in result["event_terms_ms"].items():
        print(f"  {k:<32} {v:8.2f} ms")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
