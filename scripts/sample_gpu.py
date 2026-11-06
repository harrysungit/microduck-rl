"""Sample APU power, GPU use and GPU memory every few seconds until a file appears.

    python scripts/sample_gpu.py --out gpu.csv --until logs/console/<TS>.stop
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_throughput import read_gpu, read_ppt_watts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--until", required=True, help="stop once this file exists")
    ap.add_argument("--interval", type=float, default=10.0, help="s")
    args = ap.parse_args()
    t0 = time.time()
    with open(args.out, "w") as f:
        f.write("elapsed_s,ppt_w,gpu_use_pct,gpu_mem_gib\n")
        while not os.path.exists(args.until):
            use, mem = read_gpu()
            ppt = read_ppt_watts()
            fmt = lambda v, p: "" if v is None else format(v, p)  # noqa: E731
            f.write(f"{time.time() - t0:.0f},{fmt(ppt, '.1f')},{fmt(use, '.0f')},{fmt(mem, '.3f')}\n")
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
