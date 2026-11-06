"""Break one environment step down by phase on the GPU.

Builds a task's environment with N worlds and steps it with random actions, so
falls and resets happen as they do early in training. Each phase that
ManagerBasedRlEnv.step() calls (action processing; per substep the actuators,
the write to the simulation, the physics step and the scene update; then
terminations, rewards, resets, forward kinematics, commands, events, sensing
and observations) is wrapped with a device synchronisation on both sides, so
every phase is charged its own GPU time. Only the outermost wrapped call is
timed; nested calls count toward their caller.

Also reports the process's CPU time over the timed steps: CPU time close to
wall time means the step is host-bound, while CPU time well below it means the
host is waiting on the device. --alloc-stats counts the device allocations made
during the timed steps; MICRODUCK_HIP_CACHING_ALLOCATOR=0 turns off the fork's
caching allocator for comparison.

    python scripts/profile_step.py --num-envs 4096 --steps 20 --out <file.json>
    python scripts/profile_step.py --num-envs 4096 --steps 20 --no-sync   # wall time only
"""

import argparse
import json
import os
import resource
import statistics
import time
from collections import defaultdict

import torch  # before warp, so the ROCm libraries bundled with torch load first
import warp as wp

import mjlab  # noqa: F401  # loads the mjlab.tasks entry points, and with them the ROCm patch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from microduck_rl.patches.rocm import CachingAllocator

# (attribute of the env holding the object, method) in the order step() calls them.
PHASES = [
    ("action_manager", "process_action"),
    ("action_manager", "apply_action"),
    ("scene", "write_data_to_sim"),
    ("sim", "step"),
    ("scene", "update"),
    ("metrics_manager", "compute_substep"),
    ("termination_manager", "compute"),
    ("reward_manager", "compute"),
    ("metrics_manager", "compute"),
    (None, "_reset_idx"),
    ("sim", "forward"),
    ("command_manager", "compute"),
    ("event_manager", "apply"),
    ("sim", "sense"),
    ("observation_manager", "compute"),
]


class CountingAllocator:
    """Wraps a Warp allocator to count its calls and the time spent in them."""

    def __init__(self, inner):
        self.inner = inner
        self.memory_kind = inner.memory_kind
        self.deallocate_requires_context_guard = inner.deallocate_requires_context_guard
        self.allocs = self.frees = self.bytes = 0
        self.alloc_s = self.free_s = 0.0

    def allocate(self, size_in_bytes):
        t = time.perf_counter()
        ptr = self.inner.allocate(size_in_bytes)
        self.alloc_s += time.perf_counter() - t
        self.allocs += 1
        self.bytes += size_in_bytes
        return ptr

    def deallocate(self, ptr, size_in_bytes):
        t = time.perf_counter()
        self.inner.deallocate(ptr, size_in_bytes)
        self.free_s += time.perf_counter() - t
        self.frees += 1


def sync() -> None:
    torch.cuda.synchronize()
    wp.synchronize_device("cuda:0")


def summarize_trace(path: str, steps: int, wall_s: float) -> dict:
    """Device busy time (union of kernel intervals) and per-kernel device time from a Chrome trace."""
    with open(path) as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel" and "dur" in e]
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    busy, end = 0.0, float("-inf")
    for start, stop in spans:
        if stop > end:
            busy += stop - max(start, end)
            end = stop
    per_name: dict[str, list[float]] = defaultdict(list)
    for e in events:
        per_name[e["name"]].append(e["dur"])
    top = sorted(per_name.items(), key=lambda kv: -sum(kv[1]))
    return {
        "kernels_per_step": len(events) / steps,
        "device_busy_ms_per_step": busy / 1e3 / steps,
        "device_busy_fraction": busy / 1e6 / wall_s,
        "top": {name: {"ms_per_step": sum(d) / 1e3 / steps, "launches_per_step": len(d) / steps,
                       "mean_us": sum(d) / len(d)} for name, d in top[:25]},
    }


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--action-noise", type=float, default=0.5, help="std of the random actions")
    ap.add_argument("--seed", type=int, default=0, help="seeds the env and the action sequence, for A/B runs")
    ap.add_argument("--no-sync", action="store_true", help="time whole steps only, without per-phase syncs")
    ap.add_argument("--kernels", type=int, default=0, metavar="N",
                    help="also time every Warp kernel (wp.timing_begin) and list the top N; "
                         "rocprofv3 from the host ROCm cannot load into torch's bundled 7.13 runtime")
    ap.add_argument("--alloc-stats", action="store_true", help="count device allocations during the timed steps")
    ap.add_argument("--njmax", type=int, help="override the task's constraint rows per world")
    ap.add_argument("--jacobian", choices=("auto", "dense", "sparse"),
                    help="override the task's Jacobian and mass-matrix representation (auto: sparse above 32 DOFs)")
    ap.add_argument("--no-origin-sites", action="store_true",
                    help="skip the per-env origin marker sites mjlab's terrain adds to the shared model")
    ap.add_argument("--torch-trace", metavar="FILE",
                    help="record the timed steps with torch.profiler (Kineto/roctracer sees Warp's HIP launches "
                         "too), write a Chrome trace and report device busy time and per-kernel device time")
    ap.add_argument("--out", help="write the result as JSON")
    args = ap.parse_args()
    if args.torch_trace:
        # Kineto needs rocprofiler registration, which env/activate.sh turns off; HIP reads it when it starts.
        os.environ.pop("ROCPROFILER_REGISTER_ENABLED", None)

    wp.init()
    device = wp.get_device("cuda:0")

    if args.no_origin_sites:
        from mjlab.terrains.terrain_entity import TerrainEntity

        TerrainEntity._add_env_origin_sites = lambda self: None
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.njmax:
        env_cfg.sim.njmax = args.njmax
    if args.jacobian:
        env_cfg.sim.mujoco.jacobian = args.jacobian
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env.reset()
    shape = env.action_space.shape
    # The fork's ROCm patch installs it on HIP unless MICRODUCK_HIP_CACHING_ALLOCATOR=0.
    caching = isinstance(wp.get_device_allocator(device), CachingAllocator)

    totals: dict[str, list[float]] = defaultdict(list)
    depth = [0]
    if not args.no_sync:
        for owner, name in PHASES:
            obj = env if owner is None else getattr(env, owner)
            label = f"{owner or 'env'}.{name}"
            fn = getattr(obj, name)

            def timed(*a, _fn=fn, _label=label, **k):
                if depth[0]:
                    return _fn(*a, **k)
                depth[0] += 1
                sync()
                t = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    sync()
                    totals[_label].append(time.perf_counter() - t)
                    depth[0] -= 1

            setattr(obj, name, timed)

    gen = torch.Generator(device="cuda:0").manual_seed(args.seed)

    def step() -> None:
        env.step(args.action_noise * torch.randn(shape, device="cuda:0", generator=gen))

    for _ in range(args.warmup):
        step()
    sync()
    totals.clear()
    counter = None
    if args.alloc_stats:
        counter = CountingAllocator(wp.get_device_allocator(device))
        wp.set_device_allocator(device, counter)
    if args.kernels:
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
    profiler = None
    if args.torch_trace:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CUDA])
        profiler.__enter__()
    step_times = []
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    for _ in range(args.steps):
        t = time.perf_counter()
        step()
        sync()
        step_times.append(time.perf_counter() - t)
    cpu, wall = cpu_seconds() - cpu0, time.perf_counter() - wall0
    trace = {}
    if profiler is not None:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(args.torch_trace)
        trace = summarize_trace(args.torch_trace, args.steps, wall)
    kernels, device_ops = {}, {}
    if args.kernels:
        per_name: dict[str, list[float]] = defaultdict(list)
        per_kind: dict[str, list[float]] = defaultdict(list)
        kinds = {wp.TIMING_KERNEL: "kernel", wp.TIMING_MEMCPY: "memcpy", wp.TIMING_MEMSET: "memset"}
        for r in wp.timing_end():
            per_name[r.name].append(r.elapsed)  # ms
            per_kind[kinds.get(r.filter, "other")].append(r.elapsed)
        kernels = {
            name: {"ms_per_step": sum(v) / args.steps, "launches_per_step": len(v) / args.steps}
            for name, v in sorted(per_name.items(), key=lambda kv: -sum(kv[1]))
        }
        device_ops = {kind: {"ms_per_step": sum(v) / args.steps, "ops_per_step": len(v) / args.steps}
                      for kind, v in per_kind.items()}

    step_ms = statistics.mean(step_times) * 1e3
    phases = {
        label: {"ms_per_step": sum(v) / args.steps * 1e3, "calls_per_step": len(v) / args.steps}
        for label, v in totals.items()
    }
    timed_ms = sum(p["ms_per_step"] for p in phases.values())
    result = {
        "task": args.task, "num_envs": args.num_envs, "steps": args.steps, "synced": not args.no_sync,
        "njmax": env.sim._wp_data.njmax,
        "nsite": env.sim._wp_model.nsite,
        "step_ms": step_ms, "env_steps_per_s": args.num_envs / (step_ms / 1e3),
        "process_cpu_over_wall": cpu / wall, "untimed_ms_per_step": step_ms - timed_ms,
        "phases": dict(sorted(phases.items(), key=lambda kv: -kv[1]["ms_per_step"])),
        "warp_kernels": kernels,
        "warp_device_ops": device_ops,
        "torch_trace": trace,
        "caching_allocator": caching,
    }
    if counter is not None:
        result["allocations"] = {
            "allocs_per_step": counter.allocs / args.steps,
            "frees_per_step": counter.frees / args.steps,
            "mib_per_step": counter.bytes / args.steps / 2**20,
            "ms_in_alloc_per_step": counter.alloc_s / args.steps * 1e3,
            "ms_in_free_per_step": counter.free_s / args.steps * 1e3,
        }

    print(f"{args.task}, {args.num_envs} envs, {args.steps} steps: {step_ms:.1f} ms/step "
          f"({result['env_steps_per_s']:.0f} env steps/s), process CPU/wall {cpu / wall:.2f}"
          f", caching allocator {'on' if caching else 'off'}, njmax {result['njmax']}, nsite {result['nsite']}")
    if counter is not None:
        a = result["allocations"]
        print(f"  allocations per step: {a['allocs_per_step']:.0f} allocs ({a['mib_per_step']:.1f} MiB), "
              f"{a['frees_per_step']:.0f} frees; {a['ms_in_alloc_per_step']:.1f} ms in allocate, "
              f"{a['ms_in_free_per_step']:.1f} ms in deallocate")
    for label, p in result["phases"].items():
        print(f"  {label:<36} {p['ms_per_step']:8.2f} ms  {100 * p['ms_per_step'] / step_ms:5.1f}%  "
              f"x{p['calls_per_step']:.1f}")
    if phases:
        print(f"  {'(untimed glue)':<36} {result['untimed_ms_per_step']:8.2f} ms")
    if kernels:
        kernel_ms = sum(k["ms_per_step"] for k in kernels.values())
        launches = sum(k["launches_per_step"] for k in kernels.values())
        print(f"Warp device work: {kernel_ms:.1f} ms/step over {launches:.0f} operations/step "
              f"({100 * kernel_ms / step_ms:.0f}% of the step); by kind: " + ", ".join(
                  f"{k} {v['ms_per_step']:.1f} ms x{v['ops_per_step']:.0f}" for k, v in device_ops.items()))
        print(f"top {args.kernels}:")
        for name, k in list(kernels.items())[:args.kernels]:
            print(f"  {name[:60]:<60} {k['ms_per_step']:8.2f} ms  {100 * k['ms_per_step'] / step_ms:5.1f}%  "
                  f"x{k['launches_per_step']:.0f}")
    if trace:
        print(f"torch.profiler trace: {trace['kernels_per_step']:.0f} kernels/step, device busy "
              f"{trace['device_busy_ms_per_step']:.1f} ms/step ({100 * trace['device_busy_fraction']:.0f}% of wall); top:")
        for name, k in list(trace["top"].items())[:15]:
            print(f"  {name[:60]:<60} {k['ms_per_step']:8.2f} ms  x{k['launches_per_step']:.0f}  "
                  f"mean {k['mean_us']:.0f} us")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
