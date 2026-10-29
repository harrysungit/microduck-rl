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
"""

import functools
import inspect
import os
from collections import defaultdict

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
    """

    def __init__(self, inner):
        self.inner = inner
        self.memory_kind = inner.memory_kind
        self.deallocate_requires_context_guard = inner.deallocate_requires_context_guard
        self.blocks: dict[int, list[int]] = defaultdict(list)

    def allocate(self, size_in_bytes):
        cached = self.blocks.get(size_in_bytes)
        return cached.pop() if cached else self.inner.allocate(size_in_bytes)

    def deallocate(self, ptr, size_in_bytes):
        self.blocks[size_in_bytes].append(ptr)


def _is_hip(simulation) -> bool:
    return bool(getattr(simulation.wp_device, "is_hip", False))


def _use_caching_allocator(device) -> None:
    device = wp.get_device(device)
    if not getattr(device, "is_hip", False) or os.environ.get("MICRODUCK_HIP_CACHING_ALLOCATOR") == "0":
        return
    current = wp.get_device_allocator(device)
    if not isinstance(current, CachingAllocator):
        wp.set_device_allocator(device, CachingAllocator(current))


def apply() -> None:
    """Patch mjlab's ``Simulation`` for HIP devices. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True

    should_use_cuda_graph = _sim.Simulation._should_use_cuda_graph
    init = _sim.Simulation.__init__
    init_signature = inspect.signature(init)

    @functools.wraps(should_use_cuda_graph)
    def _should_use_cuda_graph(self) -> bool:
        if _is_hip(self):
            print("[INFO] HIP device: CUDA graphs disabled, simulation runs eagerly")
            return False
        return should_use_cuda_graph(self)

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        # Installed here rather than at import, so Warp initialises only once
        # mjlab has chosen the device.
        _use_caching_allocator(init_signature.bind(self, *args, **kwargs).arguments["device"])
        init(self, *args, **kwargs)
        if _is_hip(self):
            self._wp_model.opt.graph_conditional = False

    _should_use_cuda_graph._microduck_rocm = True
    _sim.Simulation._should_use_cuda_graph = _should_use_cuda_graph
    _sim.Simulation.__init__ = __init__
