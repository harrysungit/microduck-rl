"""Check the ROCm training environment.

Reports package versions and the torch and Warp devices, runs one Warp kernel
and one torch matmul on the GPU, imports the simulation stack, confirms the
ROCm patch is active and lists the registered MicroDuck tasks. Exits non-zero
if any check fails. With --out, also writes the report as Markdown.
"""

import argparse
import importlib.metadata as md
import platform
import sys
from datetime import datetime

import numpy as np
import torch  # before warp, so the ROCm libraries bundled with torch load first
import warp as wp

DISTS = [
    "torch", "warp-lang", "mujoco", "mujoco-warp", "mjlab", "rsl-rl-lib",
    "better-actuator-models", "mjlab-microduck", "microduck-rl", "numpy", "onnxruntime",
]


@wp.kernel
def _axpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


def _warp_device() -> str:
    d = wp.get_device("cuda:0")
    if not d.is_hip:
        raise RuntimeError(f"{d} is not a HIP device")
    return f"{d.name}, arch {d.arch_str}, is_hip={d.is_hip}, warp {wp.config.version}"


def _warp_kernel() -> str:
    n = 1 << 20
    x = wp.array(np.arange(n, dtype=np.float32), device="cuda:0")
    y = wp.ones(n, dtype=float, device="cuda:0")
    wp.launch(_axpy, dim=n, inputs=[2.0, x, y], device="cuda:0")
    out = y.numpy()
    if not np.allclose(out, 2.0 * np.arange(n) + 1.0):
        raise RuntimeError("axpy result mismatch")
    return f"axpy over {n} floats matches numpy"


def _torch_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no GPU")
    a = torch.randn(512, 512, device="cuda")
    err = (a @ a - (a.cpu() @ a.cpu()).cuda()).abs().max().item()
    if err > 1e-2:
        raise RuntimeError(f"matmul differs from CPU by {err}")
    return f"{torch.cuda.get_device_name(0)}, hip {torch.version.hip}, matmul ok"


def _stack() -> str:
    import mjlab  # loads the mjlab.tasks entry points, which applies the ROCm patch
    import mujoco
    import mujoco_warp  # noqa: F401
    from mjlab.sim import sim as mjsim

    if not getattr(mjsim.Simulation._should_use_cuda_graph, "_microduck_rocm", False):
        raise RuntimeError("microduck_rl ROCm patch is not active")
    return f"mujoco {mujoco.__version__}, mjlab {md.version('mjlab')}, ROCm patch active"


def _tasks() -> str:
    from mjlab.tasks.registry import list_tasks

    duck = [t for t in list_tasks() if "MicroDuck" in t]
    for task in ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Velocity-Flat-MicroDuck-HD1910"):
        if task not in duck:
            raise RuntimeError(f"{task} is not registered")
    hd1910 = sum("HD1910" in t for t in duck)
    return f"{len(duck)} MicroDuck tasks registered, {hd1910} of them HD-1910"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="also write the report to this Markdown file")
    args = ap.parse_args()

    rows: list[tuple[str, str, str]] = []

    def check(name, fn):
        try:
            rows.append((name, "ok", fn()))
        except Exception as e:  # report every failure, then exit non-zero
            rows.append((name, "FAIL", f"{type(e).__name__}: {e}"))

    for dist in DISTS:
        check(f"version {dist}", lambda d=dist: md.version(d))
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    check("torch device", _torch_device)
    check("warp device", _warp_device)
    check("warp kernel", _warp_kernel)
    check("simulation stack", _stack)
    check("tasks", _tasks)

    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{name:<{width}}  {status:<4}  {detail}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# Environment check {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            f.write(f"Host: {platform.node()}, kernel {platform.release()}, Python {platform.python_version()}\n\n")
            f.write("| Check | Status | Detail |\n|---|---|---|\n")
            for name, status, detail in rows:
                f.write(f"| {name} | {status} | {detail} |\n")

    sys.exit(any(status == "FAIL" for _, status, _ in rows))


if __name__ == "__main__":
    main()
