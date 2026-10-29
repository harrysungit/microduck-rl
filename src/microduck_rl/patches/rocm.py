"""ROCm/HIP adjustments for mjlab on AMD GPUs.

Applied from ``microduck_rl.tasks``, which mjlab imports through the
``mjlab.tasks`` entry point, so ``train``, ``play`` and the export all run
with it. Nothing changes on CUDA or CPU devices.

- mjlab captures CUDA graphs when the Warp device is CUDA, has a memory pool
  and reports a driver >= 12.4. AMD's Warp port presents HIP devices as CUDA
  devices and reports the packed ROCm version as the driver version (about
  (70226, 1) on ROCm 7.2), so mjlab would capture. On ROCm 7.2 every graph
  replay leaks memory, and HIP capture cannot hold the allocations
  mujoco_warp makes, so the simulation runs eagerly instead.
- mujoco_warp's solver loop uses conditional graph nodes while
  ``opt.graph_conditional`` is set, and HIP has none. Use the fixed-iteration
  loop instead.
- Running eagerly, every mujoco_warp step allocates and frees its scratch
  buffers (for the Microduck velocity task at 4096 worlds: 286 allocations,
  13 GiB, per environment step). Outside graph capture, AMD's Warp port serves
  them with plain hipMalloc/hipFree, since its async pool aliases in-use
  memory on gfx1151, and that took 86 % of the step. A caching allocator
  hands freed blocks back out instead (see CachingAllocator). Set
  MICRODUCK_HIP_CACHING_ALLOCATOR=0 to compare without it.
- With ROCm PyTorch, TorchScript's tensor-expression fuser spends tens of
  milliseconds of CPU time on every call of mjlab's ``torch.jit.script`` math
  helpers (39 ms per ``quat_from_euler_xyz`` in an environment reset, against
  25 us unfused), which tiny per-environment tensors never earn back. The
  fuser and the profiling executor are turned off when torch is a HIP build.
- mjlab sets ``torch.backends.cuda.matmul.fp32_precision = "tf32"``. RDNA has
  no TF32, so matmuls run in IEEE FP32 regardless, but with PyTorch TunableOp
  enabled, torch 2.12 raises when TunableOp reads the TF32 flag through the
  legacy getter ("mix of the legacy and new APIs"). On HIP mjlab sets "ieee",
  which is what the hardware does anyway.

Opt-in, MICRODUCK_HIP_GRAPHS=1: capture mjlab's step, forward, reset and sense
graphs on HIP after all. Eager steps spend about a fifth of their time between
kernel launches. The simulation first runs each captured function once eagerly,
so every scratch buffer it needs is in the caching allocator's cache; capture
then takes all its memory from the cache, the graphs hold no allocation nodes
(ROCm 7.2 leaks the memory those allocate on every replay), and blocks handed
out during capture stay with the graphs as their shared scratch.
"""

import functools
import inspect
import os
from collections import defaultdict

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.sim import sim as _sim

_applied = False


class CachingAllocator:
    """Warp allocator that keeps freed device blocks by size and reuses them.

    A block is handed out again only after Python has released it. Warp and
    mjlab's torch views of simulation data run on the same stream, so stream
    order keeps the previous user of a block ahead of the next one, as in
    PyTorch's caching allocator. Memory is never returned to the driver; the
    cache holds at most the peak working set of a step.

    Blocks handed out during graph capture belong to the graphs from then on.
    All of a simulation's graphs replay one after another on one stream, so
    they share these blocks as scratch: a block a capture frees goes to a
    graph pool that later allocations in the same or another capture draw from
    first, and that eager allocations never see. With the cache warmed up by
    an eager run of the same functions, capture then allocates in the pattern
    of that run and never reaches the inner allocator; ``capture_misses``
    counts the times it did.
    """

    def __init__(self, inner, device=None):
        self.inner = inner
        self.device = device
        self.memory_kind = inner.memory_kind
        self.deallocate_requires_context_guard = inner.deallocate_requires_context_guard
        self.blocks: dict[int, list[int]] = defaultdict(list)  # free, for eager use
        self.graph_blocks: dict[int, list[int]] = defaultdict(list)  # free graph scratch
        self.graph_owned: set[int] = set()
        self.capture_misses = 0

    def allocate(self, size_in_bytes):
        # Warp's Device.is_capturing returns the dict of active captures when one is on.
        if self.device is not None and bool(self.device.is_capturing):
            for pool in (self.graph_blocks, self.blocks):
                cached = pool.get(size_in_bytes)
                if cached:
                    ptr = cached.pop()
                    break
            else:
                ptr = self.inner.allocate(size_in_bytes)
                self.capture_misses += 1
            self.graph_owned.add(ptr)
            return ptr
        cached = self.blocks.get(size_in_bytes)
        return cached.pop() if cached else self.inner.allocate(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        pool = self.graph_blocks if ptr in self.graph_owned else self.blocks
        pool[size_in_bytes].append(ptr)


def _is_hip(simulation) -> bool:
    return bool(getattr(simulation.wp_device, "is_hip", False))


def _use_caching_allocator(device) -> None:
    device = wp.get_device(device)
    if not getattr(device, "is_hip", False) or os.environ.get("MICRODUCK_HIP_CACHING_ALLOCATOR") == "0":
        return
    current = wp.get_device_allocator(device)
    if not isinstance(current, CachingAllocator):
        wp.set_device_allocator(device, CachingAllocator(current, device))


def _graphs_requested() -> bool:
    return os.environ.get("MICRODUCK_HIP_GRAPHS") == "1"


def _warm_up(simulation) -> None:
    """Run each function mjlab captures once eagerly, so capture finds every buffer it needs cached."""
    with wp.ScopedDevice(simulation.wp_device):
        mjwarp.step(simulation.wp_model, simulation.wp_data)
        mjwarp.forward(simulation.wp_model, simulation.wp_data)
        mjwarp.reset_data(simulation.wp_model, simulation.wp_data, reset=simulation._reset_mask_wp)
        if simulation._sensor_context is not None:
            simulation._sense_kernel()
    wp.synchronize_device(simulation.wp_device)


def apply() -> None:
    """Patch mjlab's ``Simulation`` for HIP devices. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True

    if torch.version.hip:
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(False)

        from mjlab.utils import torch as _mjlab_torch

        configure_29 = _mjlab_torch._configure_29

        @functools.wraps(configure_29)
        def _configure_29_ieee(allow_tf32: bool) -> None:
            configure_29(False)

        _mjlab_torch._configure_29 = _configure_29_ieee

    should_use_cuda_graph = _sim.Simulation._should_use_cuda_graph
    init = _sim.Simulation.__init__
    init_signature = inspect.signature(init)
    create_graph = _sim.Simulation.create_graph

    @functools.wraps(should_use_cuda_graph)
    def _should_use_cuda_graph(self) -> bool:
        if _is_hip(self):
            if not _graphs_requested():
                print("[INFO] HIP device: CUDA graphs disabled, simulation runs eagerly")
            return False  # with MICRODUCK_HIP_GRAPHS=1, captured after warm-up in __init__
        return should_use_cuda_graph(self)

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        # Installed here rather than at import, so Warp initialises only once
        # mjlab has chosen the device.
        _use_caching_allocator(init_signature.bind(self, *args, **kwargs).arguments["device"])
        init(self, *args, **kwargs)
        if _is_hip(self):
            self._wp_model.opt.graph_conditional = False
            if _graphs_requested() and isinstance(wp.get_device_allocator(self.wp_device), CachingAllocator):
                self.use_cuda_graph = True
                self.create_graph()

    @functools.wraps(create_graph)
    def _create_graph(self) -> None:
        # Every capture, including the re-capture after expand_model_fields(),
        # starts from a warm cache, so the graphs allocate nothing.
        if _is_hip(self) and self.use_cuda_graph:
            _warm_up(self)
            allocator = wp.get_device_allocator(self.wp_device)
            misses = allocator.capture_misses
            create_graph(self)
            print(f"[INFO] HIP device: CUDA graphs captured after warm-up "
                  f"({allocator.capture_misses - misses} cache misses during capture)")
        else:
            create_graph(self)

    _should_use_cuda_graph._microduck_rocm = True
    _sim.Simulation._should_use_cuda_graph = _should_use_cuda_graph
    _sim.Simulation.__init__ = __init__
    _sim.Simulation.create_graph = _create_graph
