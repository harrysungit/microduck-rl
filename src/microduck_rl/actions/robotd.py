"""robotd's runtime target processing as an mjlab action term.

The targets the duck's servos receive are not ``home + action``: robotd scales
the action with the battery voltage and low-pass filters the targets
(robotd_chain.py has the arithmetic and the robotd source lines). The training
tasks built from the official configs have none of this. This term does it per
environment; the observation's last action stays the raw policy output
(mjlab's ``ActionManager.action``), as on the robot.

The battery voltage is the HD-1910 actuator's per-env ``vin_tensor``, so the
action scale and the servo torque budget see one supply. Two simplifications,
both small next to what the board reported on 2026-11-26 (scale 0.922-0.929
over one sit at about 8.0 V): the scale uses that no-load voltage, not the
load-sagged reading robotd sees; and an episode starts with the filter empty,
where the robot carries the walk policy's last target into sit-stand (robotd
does not reset between the two nets).

The filter state is reset with the episode: ``reset(env_ids)`` makes those
worlds' next tick unfiltered, as robotd's ``Controller::reset()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from microduck_rl.actions import robotd_chain

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RobotdJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets as robotd sends them to the servos."""

    voltage_adapt: bool = True
    nominal_voltage: float = robotd_chain.NOMINAL_VOLTAGE
    head_lowpass: float = robotd_chain.HEAD_LOWPASS
    legs_lowpass: float = robotd_chain.LEGS_LOWPASS

    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        if self.clip is not None:
            raise ValueError("robotd does not clip targets; RobotdJointPositionActionCfg takes no clip")

    def build(self, env: ManagerBasedRlEnv) -> RobotdJointPositionAction:
        return RobotdJointPositionAction(self, env)


class RobotdJointPositionAction(JointPositionAction):
    """JointPositionAction with robotd's voltage scale and target low-pass."""

    cfg: RobotdJointPositionActionCfg

    def __init__(self, cfg: RobotdJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        alpha = robotd_chain.joint_alphas(self._target_names, cfg.head_lowpass, cfg.legs_lowpass)
        self._alpha = torch.tensor(alpha, device=self.device)
        self._previous = torch.zeros_like(self._raw_actions)
        self._fresh = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._supply = None
        if cfg.voltage_adapt:
            supplies = [a for a in self._entity.actuators if hasattr(a, "vin_tensor")]
            if len(supplies) != 1:
                raise ValueError(
                    f"voltage_adapt needs exactly one actuator with a per-env vin_tensor, found {len(supplies)}"
                )
            self._supply = supplies[0]

    @property
    def runtime_scale(self) -> torch.Tensor | float:
        """robotd's ``applied_action_scale`` per env, (N, 1), or 1.0 without voltage adaptation."""
        if self._supply is None:
            return 1.0
        # Read every tick: tools such as bam.mjlab.Simulator rewrite vin_tensor in place.
        return robotd_chain.voltage_scale(self._supply.vin_tensor, self.cfg.nominal_voltage)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions = robotd_chain.targets(
            self._raw_actions,
            self._offset,
            self._scale * self.runtime_scale,
            self._previous,
            self._fresh,
            self._alpha,
        )
        self._previous[:] = self._processed_actions
        self._fresh[:] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._fresh[slice(None) if env_ids is None else env_ids] = True
