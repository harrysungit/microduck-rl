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
"""

import functools

from mjlab.sim import sim as _sim

_applied = False


def _is_hip(simulation) -> bool:
    return bool(getattr(simulation.wp_device, "is_hip", False))


def apply() -> None:
    """Patch mjlab's ``Simulation`` for HIP devices. Safe to call repeatedly."""
    global _applied
    if _applied:
        return
    _applied = True

    should_use_cuda_graph = _sim.Simulation._should_use_cuda_graph
    init = _sim.Simulation.__init__

    @functools.wraps(should_use_cuda_graph)
    def _should_use_cuda_graph(self) -> bool:
        if _is_hip(self):
            print("[INFO] HIP device: CUDA graphs disabled, simulation runs eagerly")
            return False
        return should_use_cuda_graph(self)

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if _is_hip(self):
            self._wp_model.opt.graph_conditional = False

    _should_use_cuda_graph._microduck_rocm = True
    _sim.Simulation._should_use_cuda_graph = _should_use_cuda_graph
    _sim.Simulation.__init__ = __init__
