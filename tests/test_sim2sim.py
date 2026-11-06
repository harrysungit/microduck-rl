"""sim2sim.py on a small stand-in robot (CPU): independence, servo loop, observations."""

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sim2sim  # noqa: E402
from sim2sim import CommandQueue, Level, Observer, Servos, build_model, sample_params  # noqa: E402

ROBOT = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="trunk_base" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.5"/>
      <site name="imu"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="j1" axis="0 1 0" range="-1 1" damping="0.05" frictionloss="0.005"/>
        {backlash}
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="foot" pos="0 0 -0.1">
          <joint name="j2" axis="0 1 0" range="-1 1"/>
          <geom name="left_foot_collision" type="box" size="0.03 0.02 0.005" mass="0.02"/>
          <geom name="right_foot_collision" type="box" size="0.03 0.02 0.005" pos="0 0.05 0" mass="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j1" joint="j1" kp="0.55"/>
    <position name="j2" joint="j2" kp="0.55"/>
  </actuator>
  <sensor>
    <gyro name="imu_ang_vel" site="imu"/>
    <framequat name="orientation" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
BACKLASH = '<joint name="passive_j1_backlash" axis="0 1 0" range="-0.0175 0.0175"/>'
JOINTS = ["j1", "j2"]
NOMINAL = Level("T", "test", scene="walk", sensors=False, joint_vel_lag=1, timestep=0.005)


@pytest.fixture
def assets(tmp_path):
    (tmp_path / "scene_hd1910_walk.xml").write_text(ROBOT.format(backlash=""))
    (tmp_path / "scene_hd1910_allcollisions_backlash.xml").write_text(ROBOT.format(backlash=BACKLASH))
    return tmp_path


def make(assets, level=NOMINAL, rng=None):
    rng = rng or np.random.default_rng(0)
    p = sample_params(level, rng, len(JOINTS))
    model = build_model(level, assets, JOINTS, sim2sim.SERVO_PARAMS, p)
    data = mujoco.MjData(model)
    servos = Servos(model, JOINTS, sim2sim.SERVO_PARAMS, p, level.latency_ms, rng)
    return model, data, servos, p


def test_runner_does_not_import_the_training_code():
    code = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import sim2sim; "
            "print([m for m in sys.modules if m.split('.')[0] in ('mjlab', 'mjlab_microduck')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"


def test_command_queue_releases_targets_after_the_latency():
    q = CommandQueue((0.02, 0.02), np.random.default_rng(0))
    q.reset(np.array([0.0]))
    q.send(0.0, np.array([1.0]))
    assert q.at(0.019)[0] == 0.0 and q.at(0.02)[0] == 1.0
    q.send(0.02, np.array([2.0]))
    assert q.at(0.039)[0] == 1.0 and q.at(0.04)[0] == 2.0


def test_command_queue_with_a_random_latency_stays_within_its_bounds():
    q = CommandQueue((0.015, 0.03), np.random.default_rng(0))
    q.reset(np.array([0.0]))
    q.send(0.0, np.array([1.0]))
    assert all(q.at(t)[0] == 0.0 for t in np.arange(0.0, 0.0149, 0.001))
    assert all(q.at(t)[0] == 1.0 for t in np.arange(0.0301, 0.06, 0.001))


def test_servos_become_torque_motors_with_bam_joint_settings(assets):
    model, _, servos, _ = make(assets)
    kt, R = servos.bam.kt.value, servos.bam.R.value
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1.0)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], max(sim2sim.VIN_RANGE) * kt / R)
    np.testing.assert_allclose(model.dof_armature[servos.dadr], servos.act.get_extra_inertia())
    np.testing.assert_allclose(model.dof_damping[servos.dadr], 0.0)
    np.testing.assert_allclose(model.dof_frictionloss[servos.dadr], 0.0)
    np.testing.assert_allclose(model.dof_solref[servos.dadr], [sim2sim.STIFF_SOLREF_FRICTION] * 2)


def test_feetech_goal_starts_at_the_measured_position(assets):
    model, data, servos, _ = make(assets)
    data.qpos[servos.qadr] = [0.3, -0.2]
    mujoco.mj_forward(model, data)
    servos.reset(data, np.zeros(2))
    np.testing.assert_allclose(servos.act.q_target_smooth, [0.3, -0.2])


def test_friction_load_uses_dof_indices_past_the_free_joint(assets):
    model, data, servos, _ = make(assets)
    assert list(servos.dadr) == [6, 7]  # the free joint takes DOFs 0-5
    servos.reset(data, np.array([0.5, -0.5]))
    model.dof_frictionloss[servos.dadr] = 0.05
    for _ in range(20):  # load the friction constraints
        servos.step(model, data)
        mujoco.mj_step(model, data)
    seen = {}
    real = servos.bam.compute_frictions
    servos.bam.compute_frictions = lambda motor, external, dq: (seen.setdefault("external", external.copy()),
                                                                real(motor, external, dq))[1]
    mujoco.mj_forward(model, data)
    expected = -data.qfrc_bias[servos.dadr] + data.qfrc_constraint[servos.dadr]
    n = data.nefc
    rows = data.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    assert set(data.efc_id[:n][rows]) & {6, 7}
    for i, d in enumerate(servos.dadr):
        expected[i] -= data.efc_force[:n][rows & (data.efc_id[:n] == d)].sum()
    servos.step(model, data)
    np.testing.assert_allclose(seen["external"], expected)


def test_supply_sag_stops_at_the_floor(assets):
    model, data, servos, _ = make(assets)
    servos.reset(data, np.zeros(2))
    servos.prev_torque[:] = 100.0
    servos.step(model, data)
    assert servos.act.vin == sim2sim.VIN_MIN


def test_observation_layout(assets):
    model, data, servos, p = make(assets)
    home = np.array([0.1, -0.1])
    data.qpos[servos.qadr] = [0.3, -0.2]
    mujoco.mj_forward(model, data)
    observer = Observer(model, servos, home, NOMINAL, p, np.random.default_rng(0))
    observer.reset(data)
    obs = observer(data, np.array([0.7, -0.7]), np.array([0.3, 0.0, 0.5]))
    assert obs.shape == (3 + 3 + 2 + 2 + 2 + 3 + 4 + 6,)
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-9)  # upright
    np.testing.assert_allclose(obs[6:8], [0.2, -0.1])
    np.testing.assert_allclose(obs[10:12], [0.7, -0.7])
    np.testing.assert_allclose(obs[12:15], [0.3, 0.0, 0.5])
    np.testing.assert_allclose(obs[15:], 0.0)


def test_sensor_observations_quantize_the_encoders(assets):
    level = replace(NOMINAL, sensors=True, joint_vel_lag=0)
    model, data, servos, p = make(assets, level)
    data.qpos[servos.qadr] = [0.30001, -0.2]
    mujoco.mj_forward(model, data)
    observer = Observer(model, servos, np.zeros(2), level, p, np.random.default_rng(0))
    observer.reset(data)
    q = observer(data, np.zeros(2), np.zeros(3))[6:8]
    np.testing.assert_allclose(q / sim2sim.ENCODER_STEP, np.round(q / sim2sim.ENCODER_STEP), atol=1e-9)
    assert abs(q[0] - 0.30001) <= sim2sim.ENCODER_STEP / 2


def test_backlash_encoder_reads_through_the_gear_play(assets):
    level = replace(NOMINAL, scene="allcollisions_backlash")
    model, data, servos, _ = make(assets, level)
    data.qpos[servos.qadr] = [0.3, -0.2]
    data.qpos[model.jnt_qposadr[model.joint("passive_j1_backlash").id]] = 0.01
    np.testing.assert_allclose(servos.measured(data), [0.31, -0.2])
    assert math.isclose(servos.bl_mask.sum(), 1.0)


# -- sitstand
POSTURES = [(2.0, (0.0, 0.0, 0.0), "stand"), (3.0, (1.0, 0.0, 0.0), "sit"), (3.0, (0.0, 0.0, 0.0), "stand")]


def posture_traces(height, tilt=2.0, dt=0.02, duration=8.0):
    t = [(k + 1) * dt for k in range(round(duration / dt))]
    return {"t": t, "z": [height(x) for x in t], "tilt": [tilt] * len(t), "vx": [0.0] * len(t), "vy": [0.0] * len(t)}


def height_profile(t):
    """Stand; down to the sit over 1.5 s from 2 s; sit; back up over 1 s from 5 s."""
    if t <= 2.0:
        return 0.118
    if t <= 3.5:
        return 0.118 - 0.058 * (t - 2.0) / 1.5
    if t <= 5.0:
        return 0.060
    return min(0.118, 0.060 + 0.058 * (t - 5.0))


def test_command_at_follows_the_schedule_it_is_given():
    assert list(sim2sim.command_at(1.0, POSTURES)) == [0.0, 0.0, 0.0]
    assert list(sim2sim.command_at(2.5, POSTURES)) == [1.0, 0.0, 0.0]
    assert list(sim2sim.command_at(5.5, POSTURES)) == [0.0, 0.0, 0.0]
    assert list(sim2sim.command_at(99.0, POSTURES)) == [0.0, 0.0, 0.0]


def test_posture_segments_give_settle_times_and_transitions():
    segs = sim2sim.posture_segments(posture_traces(height_profile), POSTURES)
    assert [sgm["transition"] for sgm in segs] == ["hold", "sit-down", "stand-up"]
    assert all(sgm["ok"] for sgm in segs)
    assert segs[0]["settled_s"] == 0.0
    assert 1.0 < segs[1]["settled_s"] < 1.3  # 7.5 cm is reached 1.1 s after the sit command
    assert 0.6 < segs[2]["settled_s"] < 0.85  # 10 cm is reached 0.7 s after the stand command
    assert math.isclose(segs[1]["z_end"], 0.060, abs_tol=1e-6)
    assert math.isclose(segs[1]["tilt_end"], 2.0)


def test_a_posture_lost_before_the_end_of_the_segment_is_not_reached():
    def height(t):  # sits at 3.5 s, stands up again at 4.5 s while the command is still sit
        return 0.118 if t <= 2.0 or t > 4.5 else (0.118 - 0.058 * (t - 2.0) / 1.5 if t <= 3.5 else 0.060)

    segs = sim2sim.posture_segments(posture_traces(height), POSTURES)
    assert segs[1]["ok"] is False and segs[1]["settled_s"] is None


def test_a_posture_with_the_trunk_tipped_is_not_reached():
    segs = sim2sim.posture_segments(posture_traces(height_profile, tilt=50.0), POSTURES)
    assert not any(sgm["ok"] for sgm in segs)


def test_seated_start_follows_the_training_spawn():
    home = np.arange(14) * 0.01
    rng = np.random.default_rng(0)
    poses = [sim2sim.seated_pose(home, rng) for _ in range(400)]
    z = np.array([pose[0] for pose in poses])
    assert z.min() >= sim2sim.SEAT_Z[0] and z.max() <= sim2sim.SEAT_Z[1]
    joints = np.array([pose[2] for pose in poses])
    for i, angle in sim2sim.SIT_KEYFRAME.items():
        assert abs(joints[:, i].mean() - angle) < 0.03
    others = [i for i in range(14) if i not in sim2sim.SIT_KEYFRAME]
    np.testing.assert_allclose(joints[:, others].mean(axis=0), home[others], atol=0.03)
    assert abs(joints[:, 3].std() - sim2sim.SEAT_JOINT_NOISE) < 0.02
    max_tilt = math.acos(math.cos(sim2sim.SEAT_TILT) ** 2)  # pitch and roll both at their limit
    for _, quat, _ in poses[:100]:
        assert math.isclose(np.linalg.norm(quat), 1.0, abs_tol=1e-9)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array(quat))
        assert math.acos(rot[8]) <= max_tilt + 1e-9


def test_groundcontact_collisions_keep_the_geoms_the_model_authors(tmp_path):
    (tmp_path / "scene_hd1910_groundcontact.xml").write_text(ROBOT.format(backlash=""))
    for mode, contype in (("groundcontact", 1), ("feet", 0)):
        level = replace(NOMINAL, scene="groundcontact", collisions=mode)
        p = sample_params(level, np.random.default_rng(0), len(JOINTS))
        model = build_model(level, tmp_path, JOINTS, sim2sim.SERVO_PARAMS, p)
        leg = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("leg").id)
        assert model.geom_contype[leg] == contype, mode
        foot = model.geom("left_foot_collision").id
        assert model.geom_condim[foot] == 3 and model.geom_priority[foot] == 1


def test_sitstand_levels_and_schedules():
    keys = [lv.key for lv in sim2sim.SITSTAND_LEVELS]
    assert len(set(keys)) == len(keys)
    assert all(lv.task == "sitstand" for lv in sim2sim.SITSTAND_LEVELS)
    assert all(lv.task == "walk" for lv in sim2sim.LEVELS)
    seated = next(lv for lv in sim2sim.SITSTAND_LEVELS if lv.start == "seated")
    assert sim2sim.schedule_for(seated) is sim2sim.SIT_FROM_SEAT
    assert sim2sim.schedule_for(sim2sim.SITSTAND_LEVELS[0]) is sim2sim.SIT_FROM_STAND
    assert sim2sim.schedule_for(sim2sim.LEVELS[0]) is sim2sim.SCHEDULE
    for schedule in (sim2sim.SIT_FROM_STAND, sim2sim.SIT_FROM_SEAT):
        assert all(cmd[0] in (0.0, 1.0) and cmd[1:] == (0.0, 0.0) for _, cmd, _ in schedule)
