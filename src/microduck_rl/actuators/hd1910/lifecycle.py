"""Opt-in HD1910 state updates owned by physical action writes.

Nothing registers or alters a task on import. Use a new task/source identity.
The phase gate assumes the installed serial Python apply_action/write/step loop;
it does not support capturing that whole loop or concurrent control writers.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from .actuator import Hd1910Actuator, Hd1910ActuatorCfg

class PhysicsWriteGate:
    """Mixin; parent owns all mathematical computation and parameter values.

    Host flags describe Python call phase, not per-world physics state. The
    installed loop calls apply_actions before every physics substep, for all
    worlds. Do not use with a captured whole Python loop or concurrent writers.
    """
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._lifecycle_cached_ctrl = torch.zeros_like(self._prev_motor_torque)
        assert self._lifecycle_cached_ctrl.shape == (data.nworld, len(self.ctrl_ids))
        self._lifecycle_armed = False
        self._lifecycle_delayed = False
        self.lifecycle_counts = dict(arms=0, advancing_delays=0, advancing_computes=0,
                                     blocked_delays=0, blocked_computes=0)

    def arm_physics_write(self):
        assert not self._lifecycle_armed, 'Previous physical control write was not consumed'
        self._lifecycle_armed = True
        self._lifecycle_delayed = False
        self.lifecycle_counts['arms'] += 1

    def apply_delay(self, cmd):
        if not self._lifecycle_armed:
            self.lifecycle_counts['blocked_delays'] += 1
            return cmd
        assert not self._lifecycle_delayed, 'One delay append per armed write'
        result = super().apply_delay(cmd)
        self._lifecycle_delayed = True
        self.lifecycle_counts['advancing_delays'] += 1
        return result

    def compute(self, cmd):
        if not self._lifecycle_armed:
            self.lifecycle_counts['blocked_computes'] += 1
            return self._lifecycle_cached_ctrl
        assert self._lifecycle_delayed, 'Physical write must apply delay before compute'
        try:
            output = super().compute(cmd)
            assert isinstance(output, torch.Tensor) and output.shape == self._lifecycle_cached_ctrl.shape
            with torch.no_grad():
                self._lifecycle_cached_ctrl.copy_(output.detach())
            self.lifecycle_counts['advancing_computes'] += 1
            return output  # Exact parent tensor, no recalculation or rescaling.
        finally:
            self._lifecycle_armed = False
            self._lifecycle_delayed = False

    def reset(self, env_ids=None):
        assert not getattr(self, '_lifecycle_armed', False), 'Reset cannot interrupt an armed write'
        super().reset(env_ids)  # Preserve original delay, lazy seed, and torque reset.
        if hasattr(self, '_lifecycle_cached_ctrl'):
            self._lifecycle_cached_ctrl[slice(None) if env_ids is None else env_ids] = 0


class LifecycleHd1910Actuator(PhysicsWriteGate, Hd1910Actuator):
    pass


@dataclass(kw_only=True)
class LifecycleHd1910ActuatorCfg(Hd1910ActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return LifecycleHd1910Actuator(self, entity, target_ids, target_names)


class PhysicsWriteActionMixin:
    """Explicit mixin for an action owning the whole gated HD1910 actuator.

    Put this before the existing action class in the MRO. Its original
    apply_actions must still set targets (including its encoder-bias handling).
    Multiple action terms sharing this actuator are deliberately unsupported.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bind_lifecycle_actuator()

    def _bind_lifecycle_actuator(self):
        actuators = self._entity.actuators
        if len(actuators) != 1 or not isinstance(actuators[0], LifecycleHd1910Actuator):
            raise TypeError('Action requires exactly one explicit LifecycleHd1910Actuator')
        action_ids = self._target_ids.detach().cpu().tolist()
        actuator_ids = actuators[0].target_ids.detach().cpu().tolist()
        if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(actuator_ids):
            raise ValueError('One action must own all gated actuator targets')
        self._lifecycle_owned_actuator = actuators[0]

    def apply_actions(self):
        super().apply_actions()  # Preserve the original target/bias transformation.
        self._lifecycle_owned_actuator.arm_physics_write()


class LifecycleJointPositionAction(PhysicsWriteActionMixin, JointPositionAction):
    """Standard JointPositionAction with explicit physical-write ownership."""


@dataclass(kw_only=True)
class LifecycleJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return LifecycleJointPositionAction(self, env)


def _copy_fields(cfg, cls):
    return cls(**{field.name: deepcopy(getattr(cfg, field.name)) for field in fields(cfg)})


def with_lifecycle_actuator(robot_cfg):
    """Copy one non-backlash HD1910 robot cfg; preserve every parameter value."""
    actuators = robot_cfg.articulation.actuators
    if len(actuators) != 1 or type(actuators[0]) is not Hd1910ActuatorCfg:
        raise TypeError('Only the reviewed single non-backlash Hd1910ActuatorCfg can be converted')
    robot = deepcopy(robot_cfg)
    robot.articulation.actuators = (_copy_fields(actuators[0], LifecycleHd1910ActuatorCfg),)
    return robot


def with_lifecycle_joint_position_action(action_cfg):
    """Copy a standard cfg; custom actions require an explicit mixin/Cfg build.

    In particular this never silently replaces a custom runtime transform,
    calibration validator, or action reset/process implementation.
    """
    if type(action_cfg) is not JointPositionActionCfg:
        raise TypeError('Custom action cfg: explicitly subclass PhysicsWriteActionMixin and preserve its build')
    return _copy_fields(action_cfg, LifecycleJointPositionActionCfg)


