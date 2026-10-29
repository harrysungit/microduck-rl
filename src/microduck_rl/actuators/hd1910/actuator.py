"""HD-1910 actuator: BAM's Feetech control law with HD-1910 M6 parameters.

BAM's Feetech actuator (``sts3215``) slews the firmware goal through
``q_target_smooth``, which bam 62bd8ce only creates when it replays a test-bench
log (``load_log``). mjlab never calls that, so the first step would fail. These
subclasses keep one smoothed goal per world and seed it from the measured
position on the first control step after a reset: ``reset()`` runs before the
reset events write the start pose, so seeding there would start from qpos0.
The per-world mask needs no host sync and leaves the other worlds' goals alone.

The goal-state reset follows fanhao375/microduck-replica (Apache-2.0),
software/training/src/mjlab_microduck/actuator/feetech_bam.py @ 25791f0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.actuator.actuator import ActuatorCmd

from mjlab_microduck.actuator import (
    BacklashEncoderBamActuator,
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)

PARAMS_JSON = Path(__file__).parent / "params" / "1910_m6.json"
PARAMS_SHA256 = "ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb"

# The XL330 cfg's randomization (microduck_constants._BAM_ACTUATOR_KWARGS) at
# the HD-1910's operating point: firmware P = 5, the gain the robot writes to
# the servos; 2S battery at 7.4-8.0 V with load sag, floored at 7.0 V.
HD1910_ACTUATOR_KWARGS = dict(
    json_path=str(PARAMS_JSON),
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=5.0,
    vin_range=(7.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=7.0,
    delay_min_lag=3,
    delay_max_lag=6,
)


class Hd1910Actuator(FrictionDRBamActuator):
    """FrictionDRBamActuator with a per-world firmware goal for the Feetech law."""

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._bam_model.actuator.q_target_smooth = torch.zeros(
            (data.nworld, len(self._target_names)), device=device
        )
        self._goal_reset_pending = torch.ones((data.nworld, 1), dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._goal_reset_pending.fill_(True)
        else:
            self._goal_reset_pending[env_ids] = True

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        actuator = self._bam_model.actuator
        actuator.q_target_smooth = torch.where(
            self._goal_reset_pending, cmd.pos, actuator.q_target_smooth
        )
        self._goal_reset_pending.zero_()
        return super().compute(cmd)


@dataclass(kw_only=True)
class Hd1910ActuatorCfg(FrictionDRBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> Hd1910Actuator:
        return Hd1910Actuator(self, entity, target_ids, target_names)


class Hd1910BacklashActuator(Hd1910Actuator, BacklashEncoderBamActuator):
    """Hd1910Actuator whose firmware loop reads the encoder through the backlash.

    The goal is seeded from that reading too: ``cmd.pos`` is motor + backlash.
    """


@dataclass(kw_only=True)
class Hd1910BacklashActuatorCfg(Hd1910ActuatorCfg, BacklashEncoderBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> Hd1910BacklashActuator:
        return Hd1910BacklashActuator(self, entity, target_ids, target_names)
