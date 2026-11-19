"""CPU-only real HD1910/BAM methods with tensor-backed simulator storage.

No physics/environment step, GPU/Warp bridge, or learning is constructed here.
"""
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg

from microduck_rl.actuators.hd1910.actuator import (
    HD1910_ACTUATOR_KWARGS, Hd1910Actuator, Hd1910ActuatorCfg,
)
from microduck_rl.actuators.hd1910.lifecycle import (
    LifecycleHd1910Actuator, LifecycleHd1910ActuatorCfg,
    LifecycleJointPositionAction, LifecycleJointPositionActionCfg,
    PhysicsWriteActionMixin, with_lifecycle_actuator,
    with_lifecycle_joint_position_action,
)
from microduck_rl.actuators.hd1910.robot import (
    MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG,
    MICRODUCK_ALLCOLLISIONS_BACKLASH_HD1910_ROBOT_CFG,
)

N, J = 8, 14


@pytest.fixture(autouse=True)
def cpu_only_scope():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def make(cls=LifecycleHd1910Actuator):
    """Real initialize/compute/reset; only Warp storage replaced by CPU tensors."""
    torch.manual_seed(271828)
    cfg = Hd1910ActuatorCfg(**HD1910_ACTUATOR_KWARGS)
    entity = NS(indexing=NS(ctrl_ids=torch.arange(J), joint_ids=torch.arange(J)))
    actuator = cls(cfg, entity, list(range(J)), [str(i) for i in range(J)])
    actuator._mjs_actuators = [NS(id=i) for i in range(J)]
    model = NS(dof_frictionloss=torch.zeros(N, J), dof_damping=torch.zeros(N, J))
    data = NS(nworld=N, qfrc_bias=torch.arange(N*J).reshape(N, J)*.0001,
              qfrc_constraint=torch.zeros(N, J), qfrc_actuator=torch.zeros(N, J),
              nefc=torch.zeros(N, dtype=torch.int64),
              efc=NS(type=torch.zeros(N, 1, dtype=torch.int64),
                     id=torch.zeros(N, 1, dtype=torch.int64), force=torch.zeros(N, 1)))
    actuator.initialize(NS(jnt_dofadr=np.arange(J), opt=NS(timestep=.005)), model, data, 'cpu')
    return actuator


def command(k=0):
    pos = torch.sin(torch.arange(N*J).reshape(N, J)*.013+k*.1)*.8
    return ActuatorCmd(pos+.2, torch.zeros_like(pos), torch.zeros_like(pos), pos, torch.cos(pos)*.05)


def snapshot(actuator):
    delay = actuator._delay_buffer
    buffer = delay._buffer
    values = {
        'smooth':actuator._bam_model.actuator.q_target_smooth,
        'pending':actuator._goal_reset_pending, 'prev':actuator._prev_motor_torque,
        'lag':delay._current_lags, 'steps':delay._step_count,
        'phase':delay._phase_offsets, 'pushes':buffer._num_pushes,
        'friction':actuator._mjwarp_model.dof_frictionloss,
        'damping':actuator._mjwarp_model.dof_damping,
        'vin':actuator.vin_tensor, 'gain':actuator.kp_scale,
        'friction_scale':actuator.friction_scale,
    }
    if buffer._buffer is not None:
        values['buffer'] = buffer._buffer
    if hasattr(actuator, '_lifecycle_cached_ctrl'):
        values['cache'] = actuator._lifecycle_cached_ctrl
    return {key:value.clone() for key,value in values.items()} | {'pointer':buffer._pointer}


def assert_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_armed_path_is_exact_real_parent_math_state_and_rng():
    parent, gate = make(Hd1910Actuator), make()
    for k in range(12):
        cmd = command(k)
        rng = torch.random.get_rng_state().clone()
        expected = parent.compute(parent.apply_delay(cmd))
        after_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng)
        gate.arm_physics_write()
        actual = gate.compute(gate.apply_delay(cmd))
        assert torch.equal(actual, expected)
        assert torch.equal(after_rng, torch.random.get_rng_state())
        actual_state = snapshot(gate)
        actual_state.pop('cache')
        assert_equal(snapshot(parent), actual_state)
        assert not gate._lifecycle_armed and not gate._lifecycle_delayed
    assert gate.lifecycle_counts['arms'] == gate.lifecycle_counts['advancing_computes'] == 12


def test_static_writes_do_not_advance_state_rng_or_friction():
    gate = make()
    for initialized in (False, True):
        if initialized:
            gate.arm_physics_write()
            gate.compute(gate.apply_delay(command()))
        old, rng = snapshot(gate), torch.random.get_rng_state().clone()
        for _ in range(5):
            cmd = command()
            assert gate.apply_delay(cmd) is cmd
            assert torch.equal(gate.compute(cmd), old['cache'])
        assert_equal(old, snapshot(gate))
        assert torch.equal(rng, torch.random.get_rng_state())


def test_partial_reset_and_extra_scene_write_leave_other_worlds_unchanged():
    gate = make()
    for k in range(4):
        gate.arm_physics_write()
        gate.compute(gate.apply_delay(command(k)))
    ids = torch.tensor([0, 3])
    keep = torch.ones(N, dtype=torch.bool)
    keep[ids] = False
    old, rng = snapshot(gate), torch.random.get_rng_state().clone()
    gate.reset(ids)
    after = snapshot(gate)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key in old:
        if key == 'pointer':
            assert old[key] == after[key]
        elif key == 'buffer':
            assert torch.equal(old[key][:, keep], after[key][:, keep])
        else:
            assert torch.equal(old[key][keep], after[key][keep]), key
    assert not after['cache'][ids].any() and not after['prev'][ids].any()
    assert not after['steps'][ids].any() and not after['pushes'][ids].any()
    assert after['pending'][ids].all()
    # Preserve original parent reset: no invented friction-budget or smooth reset.
    for key in ('friction', 'damping', 'friction_scale', 'smooth'):
        assert torch.equal(old[key], after[key]), key
    gate.compute(gate.apply_delay(command(5)))
    assert_equal(after, snapshot(gate))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_cache_is_independent_and_full_reset_clears_only_cached_control():
    gate = make()
    gate.arm_physics_write()
    output = gate.compute(gate.apply_delay(command()))
    saved = gate._lifecycle_cached_ctrl.clone()
    assert output.data_ptr() != gate._lifecycle_cached_ctrl.data_ptr()
    gate._prev_motor_torque.add_(7.)
    assert torch.equal(saved, gate._lifecycle_cached_ctrl)
    gate.reset()
    assert not gate._lifecycle_cached_ctrl.any() and not gate._prev_motor_torque.any()
    assert gate._goal_reset_pending.all() and not gate._delay_buffer._buffer._num_pushes.any()


def test_phase_errors_fail_explicitly():
    gate = make()
    gate.arm_physics_write()
    with pytest.raises(AssertionError):
        gate.arm_physics_write()
    with pytest.raises(AssertionError):
        gate.reset([0])
    with pytest.raises(AssertionError):
        gate.compute(command())
    delayed = gate.apply_delay(command())
    with pytest.raises(AssertionError):
        gate.apply_delay(command())
    gate.compute(delayed)


def test_standard_action_keeps_real_target_bias_before_arm():
    gate = make()
    action = LifecycleJointPositionAction.__new__(LifecycleJointPositionAction)
    bias = torch.linspace(-.0075, .0075, J).repeat(N, 1)
    action._target_ids = torch.arange(J)
    action._processed_actions = torch.ones(N, J)*.7
    seen = []
    def set_target(target, joint_ids):
        assert not gate._lifecycle_armed
        assert torch.equal(target, action._processed_actions-bias)
        seen.append(target.clone())
    action._entity = NS(actuators=[gate], data=NS(encoder_bias=bias), set_joint_position_target=set_target)
    action._bind_lifecycle_actuator()
    action.apply_actions()
    assert seen and gate._lifecycle_armed
    gate.compute(gate.apply_delay(command()))


def test_action_binding_rejects_partial_ownership_and_unknown_actuators():
    gate = make()
    action = LifecycleJointPositionAction.__new__(LifecycleJointPositionAction)
    action._entity = NS(actuators=[gate])
    action._target_ids = torch.arange(J-1)
    with pytest.raises(ValueError):
        action._bind_lifecycle_actuator()
    action._target_ids = torch.arange(J)
    action._entity.actuators = [make(Hd1910Actuator)]
    with pytest.raises(TypeError):
        action._bind_lifecycle_actuator()
    action._entity.actuators = [gate, gate]
    with pytest.raises(TypeError):
        action._bind_lifecycle_actuator()


def test_explicit_custom_action_mixin_preserves_constructor_process_and_reset():
    gate = make()
    class ExistingAction:
        def __init__(self, entity):
            self._entity = entity
            self._target_ids = torch.arange(J)
            self.trace = ['init']
        def process_actions(self, raw):
            self.trace.append(('process', raw))
        def reset(self, env_ids):
            self.trace.append(('reset', env_ids))
        def apply_actions(self):
            assert not gate._lifecycle_armed
            self.trace.append('original_apply')
    class ExplicitAction(PhysicsWriteActionMixin, ExistingAction):
        pass
    action = ExplicitAction(NS(actuators=[gate]))
    action.process_actions('raw')
    action.reset('selected')
    action.apply_actions()
    assert action.trace == ['init', ('process', 'raw'), ('reset', 'selected'), 'original_apply']
    assert gate._lifecycle_armed
    gate.compute(gate.apply_delay(command()))


def test_config_helpers_preserve_parameters_and_never_convert_unknown_custom_actions():
    shared = MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG
    original = deepcopy(shared)
    robot = with_lifecycle_actuator(shared)
    assert robot is not shared and type(shared.articulation.actuators[0]) is Hd1910ActuatorCfg
    old, new = shared.articulation.actuators[0], robot.articulation.actuators[0]
    assert type(new) is LifecycleHd1910ActuatorCfg
    for field in fields(old):
        assert getattr(old, field.name) == getattr(new, field.name)
        assert getattr(old, field.name) == getattr(original.articulation.actuators[0], field.name)
    built = new.build(NS(), list(range(J)), [str(i) for i in range(J)])
    assert type(built) is LifecycleHd1910Actuator
    action = JointPositionActionCfg(entity_name='robot', actuator_names=('.*',))
    converted = with_lifecycle_joint_position_action(action)
    assert type(converted) is LifecycleJointPositionActionCfg
    for field in fields(action):
        assert getattr(action, field.name) == getattr(converted, field.name)
    class CustomCfg(JointPositionActionCfg):
        pass
    with pytest.raises(TypeError):
        with_lifecycle_joint_position_action(CustomCfg(entity_name='robot', actuator_names=('.*',)))
    with pytest.raises(TypeError):
        with_lifecycle_actuator(MICRODUCK_ALLCOLLISIONS_BACKLASH_HD1910_ROBOT_CFG)
    with pytest.raises(TypeError):
        with_lifecycle_actuator(robot)
