"""robotd's target chain (actions/robotd_chain.py, actions/robotd.py) against robotd's own arithmetic."""

from types import SimpleNamespace

import mjlab  # noqa: F401  first, as in test_hd1910: importing mjlab loads microduck_rl.tasks
import numpy as np
import pytest
import torch

from microduck_rl.actions import robotd_chain
from microduck_rl.actions.robotd import RobotdJointPositionAction, RobotdJointPositionActionCfg

# duck_ipc_proto::JOINT_NAMES, robotd's order (mouth included); the sim has the same order without the mouth.
ROBOTD_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch",
                 "head_pitch", "head_yaw", "head_roll", "mouth", "right_hip_yaw", "right_hip_roll",
                 "right_hip_pitch", "right_knee", "right_ankle"]
SIM_JOINTS = [n for n in ROBOTD_JOINTS if n != "mouth"]


def robotd_loop(home, actions, scale, head, legs):
    """control.rs:745-771 transcribed: targets = home + scale * action, then the head and the leg filters."""
    previous, out = None, []
    for action in actions:
        targets = [home[j] + scale * action[j] for j in range(len(home))]
        if previous is not None:
            for j in range(5, 9):  # HEAD_JOINTS = 5..9
                targets[j] = head * targets[j] + (1.0 - head) * previous[j]
            for j in range(len(home)):
                if 5 <= j < 9 or j == 9:  # head, MOUTH_INDEX
                    continue
                targets[j] = legs * targets[j] + (1.0 - legs) * previous[j]
        previous = targets
        out.append(targets)
    return np.array(out)


def test_voltage_scale_is_robotds_with_its_clamp():
    v = np.array([5.0, 7.4, 8.0, 10.0])
    assert np.allclose(robotd_chain.voltage_scale(v), [7.4 / 6.0, 1.0, 0.925, 7.4 / 9.5])
    assert torch.allclose(robotd_chain.voltage_scale(torch.tensor([[8.0]])), torch.tensor([[0.925]]))


def test_joint_alphas_follow_robotds_joint_sets():
    alphas = robotd_chain.joint_alphas(ROBOTD_JOINTS)
    assert alphas == [0.7] * 5 + [0.5] * 4 + [1.0] + [0.7] * 5
    assert robotd_chain.joint_alphas(SIM_JOINTS) == [0.7] * 5 + [0.5] * 4 + [0.7] * 5
    with pytest.raises(ValueError, match="head filter joints"):
        robotd_chain.joint_alphas(["left_knee", "right_knee"])
    with pytest.raises(ValueError, match="low-pass"):
        robotd_chain.joint_alphas(SIM_JOINTS, head=0.0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_targets_match_robotds_loop(backend):
    rng = np.random.default_rng(0)
    home, actions, scale = rng.normal(size=15), rng.normal(size=(6, 15)), 7.4 / 8.03
    expected = robotd_loop(home, actions, scale, 0.5, 0.7)
    alpha = np.array(robotd_chain.joint_alphas(ROBOTD_JOINTS))
    def conv(x):
        x = np.asarray(x)
        if backend == "numpy":
            return x
        return torch.tensor(x, dtype=torch.bool if x.dtype == bool else torch.float64)

    previous, fresh, got = conv(np.zeros((1, 15))), conv(np.array([[True]])), []
    for action in actions:
        previous = robotd_chain.targets(conv(action[None]), conv(home), scale, previous, fresh, conv(alpha))
        fresh = conv(np.array([[False]]))
        got.append(np.asarray(previous)[0])
    assert np.allclose(got, expected)
    assert np.allclose(got[0], home + scale * actions[0])  # the first tick goes through unfiltered


def test_ticks_are_bit_identical_to_the_plain_expressions():
    """sim2sim's walk L4 used ``alpha * target + (1 - alpha) * previous``; its results must reproduce exactly."""
    rng = np.random.default_rng(1)
    home, alpha = rng.normal(size=14), np.array(robotd_chain.joint_alphas(SIM_JOINTS))
    for _ in range(200):
        action, previous = rng.normal(size=14), rng.normal(size=14)
        target = home + 0.9 * action
        filtered = robotd_chain.targets(action, home, 0.9, previous, np.array([False]), alpha)
        assert np.array_equal(filtered, alpha * target + (1 - alpha) * previous)
        assert np.array_equal(robotd_chain.targets(action, home, 0.9, previous, np.array([True]), alpha), target)


# -- the mjlab action term, on a stand-in entity (no simulation needed) --

class _Entity:
    def __init__(self, n_envs, vin):
        self.joint_names = SIM_JOINTS
        self.actuators = [SimpleNamespace(target_names=SIM_JOINTS, vin_tensor=vin)]
        default = torch.linspace(-0.5, 0.5, len(SIM_JOINTS)).repeat(n_envs, 1)
        self.data = SimpleNamespace(default_joint_pos=default, encoder_bias=torch.zeros_like(default))
        self.sent = None

    def find_joints_by_actuator_names(self, keys):
        return list(range(len(SIM_JOINTS))), list(SIM_JOINTS)

    def set_joint_position_target(self, target, joint_ids):
        self.sent = target.clone()


def make_term(vin=(8.0, 7.0), **cfg):
    vin = torch.tensor(vin).reshape(-1, 1)
    entity = _Entity(len(vin), vin)
    env = SimpleNamespace(scene={"robot": entity}, num_envs=len(vin), device="cpu")
    term = RobotdJointPositionActionCfg(entity_name="robot", actuator_names=(".*",), **cfg).build(env)
    return term, entity


def test_action_term_scales_by_each_worlds_battery_and_filters():
    term, entity = make_term()
    home = entity.data.default_joint_pos
    alpha = torch.tensor(robotd_chain.joint_alphas(SIM_JOINTS))
    scale = torch.tensor([[7.4 / 8.0], [7.4 / 7.0]])
    a0, a1 = torch.full((2, 14), 0.2), torch.full((2, 14), -0.1)
    term.process_actions(a0)
    first = home + scale * a0
    assert torch.allclose(term._processed_actions, first)
    term.process_actions(a1)
    assert torch.allclose(term._processed_actions, alpha * (home + scale * a1) + (1 - alpha) * first)
    assert torch.equal(term.raw_action, a1)  # the observation's last action stays raw
    term.apply_actions()
    assert torch.allclose(entity.sent, term._processed_actions)


def test_reset_makes_only_those_worlds_pass_through():
    term, entity = make_term()
    home = entity.data.default_joint_pos
    scale = torch.tensor([[7.4 / 8.0], [7.4 / 7.0]])
    term.process_actions(torch.full((2, 14), 0.3))
    before = term._processed_actions.clone()
    term.reset(torch.tensor([1]))
    a = torch.full((2, 14), -0.2)
    term.process_actions(a)
    alpha = torch.tensor(robotd_chain.joint_alphas(SIM_JOINTS))
    assert torch.allclose(term._processed_actions[1], (home + scale * a)[1])
    assert torch.allclose(term._processed_actions[0], (alpha * (home + scale * a) + (1 - alpha) * before)[0])


def test_the_chain_can_be_switched_off():
    term, entity = make_term(voltage_adapt=False, head_lowpass=1.0, legs_lowpass=1.0)
    home = entity.data.default_joint_pos
    for a in (torch.full((2, 14), 0.2), torch.full((2, 14), -0.4)):
        term.process_actions(a)
        assert torch.allclose(term._processed_actions, home + a)


def test_voltage_scale_follows_the_actuator_after_build():
    term, entity = make_term()
    entity.actuators[0].vin_tensor[:] = 7.4  # e.g. bam.mjlab.Simulator writing per-env voltages in place
    term.process_actions(torch.ones(2, 14))
    assert torch.allclose(term._processed_actions, entity.data.default_joint_pos + 1.0)


def test_cfg_rejects_clipping():
    with pytest.raises(ValueError, match="clip"):
        RobotdJointPositionActionCfg(entity_name="robot", actuator_names=(".*",), clip={".*": (-1.0, 1.0)})
