"""Summarise a training run into a results directory.

Reads the run's tensorboard scalars and plots mean reward and episode length,
exports a checkpoint (the last one by default) to ONNX through microduck_rl's
export path, checks the ONNX interface, and writes summary.json with the ONNX
hash, the stack versions and the upstream commits, so the model can be traced
back to the environment that trained it.

    python scripts/summarize_run.py Mjlab-Velocity-Flat-MicroDuck logs/rsl_rl/velocity/<run> \
        --out-dir logs/results/<TS>_velocity_flat_probe
"""

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_root(package: str, depth: int) -> Path:
    """The source tree an editable install of <package> points at, found without importing it."""
    return Path(importlib.util.find_spec(package).origin).resolve().parents[depth]


# Pollen's microduck_rl checkout (src/mjlab_microduck/__init__.py) and the Warp build (warp/__init__.py).
UPSTREAM = source_root("mjlab_microduck", 2)
WARP = source_root("warp", 1)
DISTS = ["torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "rsl-rl-lib", "better-actuator-models"]


def scalars(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [(e.step, e.value, e.wall_time) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def num_envs(run_dir: Path) -> int | None:
    env_yaml = run_dir / "params" / "env.yaml"
    if env_yaml.exists():
        m = re.search(r"^\s*num_envs:\s*(\d+)", env_yaml.read_text(), re.M)
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="default: the run's last model_*.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(args.run_dir.glob("model_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    ckpt = args.checkpoint or ckpts[-1]
    onnx_path = args.out_dir / "policy.onnx"
    subprocess.run([sys.executable, str(UPSTREAM / "scripts/export.py"), args.task, "--checkpoint-file", str(ckpt),
                    "--onnx-file", str(onnx_path)], check=True)

    import numpy as np
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    probe = sess.run(None, {inp.name: np.zeros((1, inp.shape[1]), np.float32)})[0]

    s = scalars(args.run_dir)
    reward = s.get("Train/mean_reward", [])
    length = s.get("Train/mean_episode_length", [])
    fps = [v for _, v, _ in s.get("Perf/total_fps", [])]
    ec = Path("/sys/class/ec_su_axb35/apu/power_mode")
    summary = {
        "task": args.task,
        "run_dir": str(args.run_dir),
        "checkpoint": ckpt.name,
        "iterations": (reward[-1][0] + 1) if reward else None,
        "num_envs": num_envs(args.run_dir),
        "wall_time_s": round(reward[-1][2] - reward[0][2], 1) if len(reward) > 1 else None,
        "median_steps_per_s": statistics.median(fps[2:]) if len(fps) > 2 else None,
        "final_mean_reward": reward[-1][1] if reward else None,
        "final_mean_episode_length": length[-1][1] if length else None,
        "onnx": {
            "file": onnx_path.name,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "input": [inp.name, inp.shape],
            "output": [out.name, out.shape],
            "zero_obs_output_finite": bool(np.isfinite(probe).all()),
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ec_power_mode": ec.read_text().strip() if ec.exists() else "unknown",
        "versions": {d: md.version(d) for d in DISTS},
        "commits": {"microduck-rl": git_head(ROOT), "microduck_rl": git_head(UPSTREAM), "warp": git_head(WARP)},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, series, label in [(axes[0], reward, "mean reward"), (axes[1], length, "mean episode length (steps)")]:
        ax.plot([p[0] for p in series], [p[1] for p in series])
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(label)
    fig.suptitle(f"{args.task} on {summary['device']}")
    fig.tight_layout()
    fig.savefig(args.out_dir / "curve.png", dpi=120)

    fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    lines = [f"# {args.task}: training probe", "",
             f"Run `{args.run_dir}`, checkpoint `{ckpt.name}`; {summary['num_envs']} envs, "
             f"{summary['iterations']} iterations in {fmt(summary['wall_time_s'], '.0f')} s "
             f"(median {fmt(summary['median_steps_per_s'], '.0f')} steps/s) on {summary['device']}, "
             f"EC mode {summary['ec_power_mode']}.", "",
             f"Final mean reward {fmt(summary['final_mean_reward'], '.2f')}, mean episode length "
             f"{fmt(summary['final_mean_episode_length'], '.0f')} steps.", "",
             f"ONNX `{inp.name}{inp.shape}` -> `{out.name}{out.shape}`, sha256 `{summary['onnx']['sha256'][:16]}…` "
             "(the .onnx file stays on the dev machine).", "",
             "![curve](curve.png)"]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
