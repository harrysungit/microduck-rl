"""Training throughput sweep on the Strix Halo GPU.

Runs `train <task>` for a few PPO iterations at each num_envs, parses rsl_rl's
per-iteration "Steps per second / Collection time / Learning time" lines, and
samples APU package power (lm-sensors PPT), GPU use and GPU memory once a
second while each run is going. Writes sweep.csv, README.md and sweep.png.

Run from the repository root with env/activate.sh sourced:

    python scripts/bench_throughput.py --out-dir logs/results/<TS>_velocity_flat_sweep
"""

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import threading
import time
from pathlib import Path

PATTERNS = {
    "fps": re.compile(r"Steps per second:\s+(\d+)"),
    "collect_s": re.compile(r"Collection time:\s+([\d.]+)s"),
    "learn_s": re.compile(r"Learning time:\s+([\d.]+)s"),
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_ppt_watts() -> float | None:
    out = subprocess.run(["sensors", "-j"], capture_output=True, text=True).stdout
    for chip, values in json.loads(out or "{}").items():
        if chip.startswith("amdgpu") and "PPT" in values:
            return float(values["PPT"]["power1_average"])
    return None


def read_gpu() -> tuple[float | None, float | None]:
    """GPU use (%) and GPU memory in use (GiB, VRAM + GTT) from rocm-smi."""
    out = subprocess.run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "gtt", "--json"],
                         capture_output=True, text=True).stdout
    try:
        card = next(iter(json.loads(out).values()))
    except (json.JSONDecodeError, StopIteration):
        return None, None
    use = card.get("GPU use (%)")
    used = sum(int(v) for k, v in card.items() if k.endswith("Total Used Memory (B)"))
    return (float(use) if use is not None else None), used / 2**30


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[tuple[float, float | None, float | None, float | None]] = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            use, mem = read_gpu()
            self.samples.append((time.time(), read_ppt_watts(), use, mem))
            self.stop.wait(1.0)


def run_one(task: str, num_envs: int, iters: int, log_path: Path, train_args: str = ""):
    cmd = ["train", task, "--env.scene.num-envs", str(num_envs), "--agent.max-iterations", str(iters),
           "--agent.logger", "tensorboard", "--agent.experiment-name", "bench_throughput",
           "--agent.run-name", f"n{num_envs}"] + shlex.split(train_args)
    per_iter: dict[str, list[float]] = {k: [] for k in PATTERNS}
    first_iter_at = None
    sampler = Sampler()
    sampler.start()
    with open(log_path, "w") as log, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ) as proc:
        for line in proc.stdout:
            log.write(line)
            clean = ANSI.sub("", line)
            for key, pat in PATTERNS.items():
                m = pat.search(clean)
                if m:
                    per_iter[key].append(float(m.group(1)))
                    if key == "fps" and first_iter_at is None:
                        first_iter_at = time.time()
        code = proc.wait()
    sampler.stop.set()
    sampler.join()
    return code, per_iter, first_iter_at, sampler.samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    ap.add_argument("--iters", type=int, default=12, help="PPO iterations per point")
    ap.add_argument("--warmup", type=int, default=2, help="iterations left out of the medians")
    ap.add_argument("--train-args", default="", help="extra arguments for train, e.g. '--env.sim.njmax 128'")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ec = Path("/sys/class/ec_su_axb35/apu/power_mode")
    context = {
        "task": args.task,
        "train_args": args.train_args,
        "ec_power_mode": ec.read_text().strip() if ec.exists() else "unknown (driver not loaded)",
        "load_average_before": os.getloadavg(),
        "idle_ppt_w": read_ppt_watts(),
    }

    rows = []
    for n in args.num_envs:
        code, it, t0, samples = run_one(args.task, n, args.iters, args.out_dir / f"train_n{n}.log", args.train_args)
        steady = [s for s in samples if t0 is not None and s[0] >= t0]
        mean = lambda xs: statistics.mean(xs) if xs else None  # noqa: E731
        fps = it["fps"][args.warmup:]
        row = {
            "num_envs": n,
            "exit_code": code,
            "iterations_parsed": len(it["fps"]),
            "steps_per_s": statistics.median(fps) if fps else None,
            "collect_s": statistics.median(it["collect_s"][args.warmup:]) if fps else None,
            "learn_s": statistics.median(it["learn_s"][args.warmup:]) if fps else None,
            "ppt_w": mean([s[1] for s in steady if s[1] is not None]),
            "gpu_use_pct": mean([s[2] for s in steady if s[2] is not None]),
            "gpu_mem_gib_max": max((s[3] for s in steady if s[3] is not None), default=None),
        }
        if row["steps_per_s"] and row["ppt_w"]:
            row["steps_per_joule"] = row["steps_per_s"] / row["ppt_w"]
        rows.append(row)
        print(json.dumps(row))

    fields = ["num_envs", "exit_code", "iterations_parsed", "steps_per_s", "collect_s", "learn_s",
              "ppt_w", "gpu_use_pct", "gpu_mem_gib_max", "steps_per_joule"]
    with open(args.out_dir / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "context.json").write_text(json.dumps(context, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["steps_per_s"]]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([r["num_envs"] for r in ok], [r["steps_per_s"] for r in ok], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("num_envs")
    ax.set_ylabel("env steps / s (PPO loop)")
    ax.set_title(args.task)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sweep.png", dpi=120)

    fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    lines = [f"# Throughput sweep: {args.task}", "",
             f"Extra train arguments: `{args.train_args or '-'}`.", "",
             f"EC power mode: {context['ec_power_mode']}; idle PPT {fmt(context['idle_ppt_w'], '.1f')} W; "
             f"load average before: {', '.join(f'{x:.2f}' for x in context['load_average_before'])}.",
             f"Medians over iterations {args.warmup}..{args.iters - 1}; power, GPU use and memory sampled at 1 Hz "
             "after the first iteration.", "",
             "| num_envs | steps/s | collection (s) | learning (s) | PPT (W) | GPU use (%) | GPU mem max (GiB) | steps/J |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['num_envs']} | {fmt(r['steps_per_s'], '.0f')} | {fmt(r['collect_s'], '.3f')} | "
                     f"{fmt(r['learn_s'], '.3f')} | {fmt(r['ppt_w'], '.1f')} | {fmt(r['gpu_use_pct'], '.0f')} | "
                     f"{fmt(r['gpu_mem_gib_max'], '.1f')} | {fmt(r.get('steps_per_joule'), '.0f')} |")
    lines += ["", "![sweep](sweep.png)"]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")

    failed = [r["num_envs"] for r in rows if r["exit_code"] or r["iterations_parsed"] < args.iters]
    if failed:
        raise SystemExit(f"train failed or stopped early at num_envs {failed}; see train_n<N>.log")


if __name__ == "__main__":
    main()
