"""Runtime patches for mjlab_microduck (Pollen Robotics' microduck_rl), applied on import.

mjlab imports this module through the ``mjlab.tasks`` entry point, so ``train``, ``play`` and
export all run with them.
"""

# HIP/ROCm adjustments for mjlab's simulation; a no-op on CUDA and CPU (see rocm.py).
from microduck_rl.patches.rocm import apply as _apply_rocm

_apply_rocm()
