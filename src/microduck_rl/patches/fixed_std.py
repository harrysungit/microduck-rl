"""Fix the PPO exploration noise at a given standard deviation: MICRODUCK_FIXED_STD=<std>, off without it.

rsl_rl's Gaussian policy learns its std, and a skill whose reward is earned with that noise can come to need it: the
SitStand policy's mean never started the descent to the sit, the noise did.
Fine-tuning with a small fixed std makes the mean policy do the work. The std is held at the value on
every distribution update and frozen, so neither the entropy bonus can raise it nor the loss lower it; the stored std_param,
the logged Policy/mean_std and the checkpoints carry the fixed value. Applied from ``microduck_rl.tasks``.
"""

import functools
import os

import torch

_applied = False


def apply() -> None:
    """Patch ``GaussianDistribution.update`` when MICRODUCK_FIXED_STD is set. Safe to call repeatedly."""
    global _applied
    value = os.environ.get("MICRODUCK_FIXED_STD")
    if _applied or not value:
        return
    std = float(value)
    if not std > 0.0:
        raise ValueError(f"MICRODUCK_FIXED_STD must be positive, got {value!r}")
    from rsl_rl.modules.distribution import GaussianDistribution

    _applied = True
    update = GaussianDistribution.update

    @functools.wraps(update)
    def fixed_update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type != "scalar":
            raise ValueError("MICRODUCK_FIXED_STD needs the scalar std parameterization")
        if self.std_param.requires_grad:
            self.std_param.requires_grad_(False)  # an optimizer skips a parameter without a gradient
        with torch.no_grad():
            self.std_param.fill_(std)  # also what a checkpoint resumed with a different std comes back as
        update(self, mlp_output)

    GaussianDistribution.update = fixed_update
