"""Side-by-side table of sim2sim.py runs: falls, tracking and getting up per level, per policy.

    python scripts/sim2sim_table.py ours=<dir>/metrics.json "reference walk"=<dir>/metrics.json
"""

import argparse
import json
import statistics
from pathlib import Path


def cell(r: dict) -> tuple[str, str, str]:
    """(falls or got up, planar speed error, yaw rate error) for one level of one run."""
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    if "episode" in r:
        e = r["episode"]
        return ("no" if e["fell_at_s"] is None else f"at {e['fell_at_s']:.1f} s"), fmt(e["err_vxy_mean"]), fmt(e["err_wz_mean"])
    eps = r["episodes"]
    if r["level"].get("start", "standing") != "standing":
        ups = [e["up_at_s"] for e in eps if e["up_at_s"] is not None]
        got = f"up {len(ups)} of {len(eps)}" + (f", {statistics.median(ups):.1f} s" if ups else "")
        return got, "-", "-"
    up = [e for e in eps if e["fell_at_s"] is None]
    mean = lambda k: statistics.mean(e[k] for e in up) if up else None  # noqa: E731
    return f"{len(eps) - len(up)} of {len(eps)}", fmt(mean("err_vxy_mean")), fmt(mean("err_wz_mean"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=metrics.json")
    args = ap.parse_args()
    runs = {}
    for item in args.runs:
        label, path = item.split("=", 1)
        runs[label] = {k: v for k, v in json.loads(Path(path).read_text()).items() if k not in ("reference", "provenance")}
    keys = list(next(iter(runs.values())))
    head = "| level | change | " + " | ".join(f"{label}: falls | {label}: speed err | {label}: yaw err" for label in runs) + " |"
    print(head)
    print("|---|---|" + "---|---|---|" * len(runs))
    for key in keys:
        title = next(iter(runs.values()))[key]["level"]["title"]
        cells = []
        for r in runs.values():
            cells += list(cell(r[key])) if key in r else ["-", "-", "-"]
        print(f"| {key} | {title} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
