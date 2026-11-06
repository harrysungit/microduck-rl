"""Tabulate eval_policy.py results over seeds: mean and range per task and policy.

    python scripts/eval_table.py <dir>/eval_*.json > table.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = [("falls_per_world_minute", "falls / world / min", ".3f"), ("fell_share", "worlds that fell", ".1%"),
           ("err_vxy_mps", "planar speed error (m/s)", ".3f"), ("err_wz_radps", "yaw rate error (rad/s)", ".3f")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    runs = defaultdict(lambda: defaultdict(list))  # task -> label -> rows
    seeds, envs, seconds = set(), set(), set()
    for f in args.files:
        ev = json.loads(f.read_text())
        seeds.add(ev["seed"])
        envs.add(ev["num_envs"])
        seconds.add(ev["seconds"])
        for row in ev["rows"]:
            runs[ev["task"]][row["label"]].append(row)
    print(f"{', '.join(map(str, sorted(envs)))} worlds, one {', '.join(f'{s:g}' for s in sorted(seconds))} s episode, "
          f"seeds {', '.join(map(str, sorted(seeds)))}; mean over seeds, range in brackets.\n")
    for task, labels in runs.items():
        print(f"**`{task}`**\n")
        print("| policy | " + " | ".join(name for _, name, _ in METRICS) + " |")
        print("|---|" + "---|" * len(METRICS))
        for label, rows in labels.items():
            cells = []
            for key, _, f in METRICS:
                v = [r[key] for r in rows]
                cells.append(f"{format(sum(v) / len(v), f)} [{format(min(v), f)}–{format(max(v), f)}]")
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
