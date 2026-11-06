"""Compare training runs: mean reward against PPO iteration and against wall-clock time.

    python scripts/compare_runs.py --out-dir logs/results/<TS>_compare \
        "original=logs/rsl_rl/velocity/<run_a>" "all fixes=logs/rsl_rl/velocity/<run_b>"
"""

import argparse
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(run_dir: str) -> dict:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    reward = acc.Scalars("Train/mean_reward")
    fps = [e.value for e in acc.Scalars("Perf/total_fps")]
    t0 = reward[0].wall_time
    return {
        "iteration": [e.step for e in reward],
        "reward": [e.value for e in reward],
        "minutes": [(e.wall_time - t0) / 60 for e in reward],
        "median_steps_per_s": statistics.median(fps[2:]) if len(fps) > 2 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for spec in args.runs:
        label, run_dir = spec.split("=", 1)
        runs[label] = {"run_dir": run_dir, **load(run_dir)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for label, r in runs.items():
        ax_it.plot(r["iteration"], r["reward"], label=label)
        ax_t.plot(r["minutes"], r["reward"], label=label)
    ax_it.set_xlabel("PPO iteration")
    ax_t.set_xlabel("wall-clock time (min)")
    ax_it.set_ylabel("mean reward")
    ax_t.legend(loc="lower right", fontsize="small")  # empty there: the curves rise steeply on the left
    fig.tight_layout()
    fig.savefig(args.out_dir / "reward_curves.png", dpi=120)

    summary = {
        label: {"run_dir": r["run_dir"], "iterations": r["iteration"][-1] + 1,
                "minutes": round(r["minutes"][-1], 1), "median_steps_per_s": r["median_steps_per_s"],
                "final_mean_reward": r["reward"][-1]}
        for label, r in runs.items()
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Training runs compared", "",
             "| Run | Iterations | Wall time (min) | Median steps/s | Final mean reward | Run directory |",
             "|---|---|---|---|---|---|"]
    for label, s in summary.items():
        lines.append(f"| {label} | {s['iterations']} | {s['minutes']} | {s['median_steps_per_s']:.0f} | "
                     f"{s['final_mean_reward']:.1f} | `{s['run_dir']}` |")
    lines += ["", "![reward curves](reward_curves.png)"]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:-2]))


if __name__ == "__main__":
    main()
