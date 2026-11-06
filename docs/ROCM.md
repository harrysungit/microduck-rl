# mjlab on ROCm: what had to change

mjlab and MuJoCo Warp run on AMD GPUs through
[AMD's Warp HIP port](https://github.com/AMD-Ecosystem/warp/tree/amd-integration-halo)
(branch `amd-integration-halo`, validated on gfx1151 only). A plain port runs,
and its physics matches: positions agree with Warp on the CPU to about 1e-6 m
and with MuJoCo C to about 3e-4 m. It is slow. These are the six bottlenecks
found on a Radeon 8060S (gfx1151), in the order they were found, and what
`src/microduck_rl/patches/` does about each.

| # | Bottleneck | How it showed | Fix |
|---|---|---|---|
| 1 | Eager MuJoCo Warp allocates and frees its scratch buffers every step: 286 allocations, 13 GiB, per environment step at 4096 worlds. Outside graph capture the HIP port uses plain `hipMalloc`/`hipFree`, because its async pool aliases memory still in use on gfx1151. 86% of the step | Warp kernel timing: the GPU was busy 14% of each step; counting allocator calls found the 286 | A caching allocator: freed blocks are kept by size and handed back out, stream-ordered like PyTorch's (`rocm.py`; `MICRODUCK_HIP_CACHING_ALLOCATOR=0` turns it off) |
| 2 | Constraint budget: mjlab's velocity default of 1500 rows per world; the walk uses at most about 50 | A budget probe counting `nefc` (`scripts/constraint_budget.py`) | `--env.sim.njmax 128` for the walk, with a guard: MuJoCo Warp drops rows past the budget silently, so peaks are tracked on the device and training stops at the limit (`perf.py`) |
| 3 | mjlab's terrain adds one origin marker site per environment to the shared model, so every world carries every marker: kinematics is O(N²), 3.5 ms per pass at 4096 worlds | `_site_local_to_global` topped the kernel times: 13 µs alone, 3.5 ms in the environment, with `nsite` = 4103 | Skip the markers, which only draw (`perf.py`; `MICRODUCK_ENV_ORIGIN_SITES=1` keeps them) |
| 4 | On ROCm builds of torch, the TorchScript tensor-expression fuser spends tens of ms of CPU per call of mjlab's scripted math: 39 ms per `quat_from_euler_xyz` in a reset, with unchanged shapes | Phase timing showed 108 ms resets; per-term timing and torch.profiler found the calls | Turn the fuser and the profiling executor off when `torch.version.hip` is set: 39 ms to 25 µs (`rocm.py`) |
| 5 | Eager kernel launches: about 3,200 kernels per step, with gaps that take 26% of the step | torch.profiler timeline: 75.8 ms per step, 56 ms of it busy | Capture mjlab's simulation graphs on HIP after all. Every capture first runs eagerly so the caching allocator holds every buffer; capture then allocates nothing and the graphs carry no allocation nodes, which ROCm 7.2 leaks on replay. Conditional graph nodes stay off (`rocm.py`, `MICRODUCK_HIP_GRAPHS=1`) |
| 6 | PPO's GEMMs: rocBLAS falls back to 32x32 tiles for its shapes | `scripts/profile_ppo.py` in the task's real runner, then kernel names | PyTorch TunableOp, cached under `.cache/tunableop`. mjlab sets TF32 through torch's new API while TunableOp reads the old one, which torch 2.12 rejects; gfx1151 has no TF32, so FP32 matmuls are set to IEEE on HIP (`rocm.py`) |

Results are in [benchmark/](../benchmark/README.md): 7.3x the plain port's
training throughput at 4096 environments, with the learning curves
overlapping by iteration.

## Gotchas

- **Exit codes.** torch's bundled Kineto attaches to rocprofiler-sdk when HIP
  starts and registers an exit handler that calls `_exit(0)`: every process
  that touched the GPU exits 0, failed or not. `env/activate.sh` sets
  `ROCPROFILER_REGISTER_ENABLED=0`. The profiling scripts turn it back on for
  themselves, since torch.profiler sees no GPU activity without it.
- **The device looks like CUDA.** The HIP port reports HIP devices as CUDA
  devices and the packed ROCm version as the driver version, so mjlab would
  capture graphs with conditional nodes on its own. `rocm.py` decides per
  device.
- **Profilers.** The host's rocprofv3 (ROCm 7.2) does not load into the ROCm
  runtime bundled with the torch nightly. Warp's `timing_begin` and
  torch.profiler work.
- **Graph memory.** With graph capture, the scratch buffers exist twice (the
  eager pool and the graph pool): GPU memory goes from 4.2 to 6.8 GiB at 4096
  environments, and stays flat over a run.
- **GPU runs are not bitwise repeatable.** Two runs with the same seed and code
  differ from the first iteration (about 6e-5 in the physics, a few points of
  mean reward over a run). Worlds on the CPU are bitwise equal. Compare runs
  against their own run-to-run spread.
