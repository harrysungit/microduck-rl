"""HD-1910 servo variant: parameters, mass model and task registration (CPU)."""

import hashlib

import numpy as np
import pytest
from bam.model import load_model
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from microduck_rl.actuators.hd1910.actuator import (
    PARAMS_JSON,
    PARAMS_SHA256,
    Hd1910ActuatorCfg,
    Hd1910BacklashActuatorCfg,
)
from microduck_rl.actuators.hd1910.mass import (
    PAYLOAD_BODY,
    SERVO_DELTA_KG,
    TOTAL_MASS_KG,
    apply_hd1910_mass,
    servo_housings,
)
from microduck_rl.actuators.hd1910 import robot as hd1910_robot
from mjlab_microduck.robot.microduck_constants import (
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_backlash_spec,
    get_standup_spec,
    get_walk_backlash_spec,
    get_walk_spec,
)
from microduck_rl.tasks import HD1910_SKILL_TASKS, HD1910_TASKS

TASK_IDS = [task_id for task_id, _, _ in HD1910_TASKS]


def test_params_are_luwu_m6():
    assert hashlib.sha256(PARAMS_JSON.read_bytes()).hexdigest() == PARAMS_SHA256


def test_params_load_with_feetech_control_law():
    assert type(load_model(str(PARAMS_JSON)).actuator).__name__ == "STS3215Actuator"


@pytest.mark.parametrize("spec_fn", [get_walk_spec, get_walk_backlash_spec, get_standup_spec, get_backlash_spec,
                                     get_allcollisions_spec, get_allcollisions_backlash_spec])
def test_mass_model(spec_fn):
    base = spec_fn().compile()
    model = apply_hd1910_mass(spec_fn()).compile()
    assert model.body_mass.sum() == pytest.approx(TOTAL_MASS_KG, abs=1e-9)

    housings = servo_housings(spec_fn())
    assert sum(len(points) for points in housings.values()) == 15
    servo_total = base.body_mass.sum() + SERVO_DELTA_KG * 15
    for name, points in housings.items():
        gained = model.body(name).mass[0] - base.body(name).mass[0]
        expected = SERVO_DELTA_KG * len(points)
        if name == PAYLOAD_BODY:
            expected += TOTAL_MASS_KG - servo_total
        assert gained == pytest.approx(expected, abs=1e-9), name
    # Bodies without housings keep their mass and inertia.
    for b in range(base.nbody):
        name = base.body(b).name
        if name not in housings:
            assert model.body(name).mass[0] == base.body(name).mass[0]
            np.testing.assert_allclose(model.body(name).inertia, base.body(name).inertia)


def test_mass_model_moves_com_toward_housings():
    spec = get_walk_spec()
    housings = servo_housings(spec)
    before = {name: np.array(spec.body(name).ipos) for name in housings}
    servo_total = get_walk_spec().compile().body_mass.sum() + SERVO_DELTA_KG * 15
    apply_hd1910_mass(spec, total_mass_kg=servo_total)  # no payload: housings only
    for name, points in housings.items():
        centroid = np.mean(points, axis=0)
        after = np.array(spec.body(name).ipos)
        assert np.linalg.norm(after - centroid) < np.linalg.norm(before[name] - centroid), name


def test_mass_model_rejects_a_lighter_total():
    with pytest.raises(ValueError):
        apply_hd1910_mass(get_walk_spec(), total_mass_kg=0.75)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_task_uses_the_hd1910_robot(task_id):
    assert task_id in list_tasks()
    for play in (False, True):
        robot = load_env_cfg(task_id, play=play).scene.entities["robot"]
        (actuator,) = robot.articulation.actuators
        expected = Hd1910BacklashActuatorCfg if "Backlash" in task_id else Hd1910ActuatorCfg
        assert type(actuator) is expected
        assert actuator.json_path == str(PARAMS_JSON) and actuator.motor_name is None
        assert actuator.kp_fw == 5.0 and actuator.vin_range == (7.4, 8.0)
        assert actuator.delay_min_lag == 3 and actuator.delay_max_lag == 6
    assert load_rl_cfg(task_id).experiment_name == "velocity_hd1910"


def test_xl330_tasks_are_unchanged():
    for task_id in ("Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Velocity-Flat-Backlash-MicroDuck"):
        (actuator,) = load_env_cfg(task_id).scene.entities["robot"].articulation.actuators
        assert actuator.motor_name == "xl330" and actuator.kp_fw == 200.0
        assert load_rl_cfg(task_id).experiment_name == "velocity"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_backlash_tasks_use_the_dense_jacobian(task_id):
    expected = "dense" if "Backlash" in task_id else "auto"
    for play in (False, True):
        assert load_env_cfg(task_id, play=play).sim.mujoco.jacobian == expected



SKILL_IDS = [task_id for task_id, *_ in HD1910_SKILL_TASKS]


@pytest.mark.parametrize("task_id", SKILL_IDS)
def test_skill_task_runs_on_the_hd1910_robot(task_id):
    assert task_id in list_tasks()
    backlash = "Backlash" in task_id
    for play in (False, True):
        cfg = load_env_cfg(task_id, play=play)
        (actuator,) = cfg.scene.entities["robot"].articulation.actuators
        assert type(actuator) is (Hd1910BacklashActuatorCfg if backlash else Hd1910ActuatorCfg)
        assert actuator.json_path == str(PARAMS_JSON) and actuator.kp_fw == 5.0
        assert cfg.sim.mujoco.jacobian == ("dense" if backlash else "auto")
        if "BallKick" in task_id:
            assert "ball" in cfg.scene.entities
    assert load_rl_cfg(task_id).experiment_name.endswith("_hd1910")


def test_velstand_hd1910_never_downloads_xl330_teachers():
    bc = load_rl_cfg("Mjlab-VelStand-Flat-MicroDuck-HD1910").algorithm.bc_cfg
    assert bc["wandb_run_path"] is None and bc["anchor_wandb_run_path"] is None
    assert bc["checkpoint_path"] and bc["anchor_checkpoint_path"]


def test_two_stage_pairs_share_an_experiment():
    for first, second in (("Mjlab-GroundPick-Flat-MicroDuck-HD1910", "Mjlab-GroundPick-Flat-Backlash-MicroDuck-HD1910"),
                          ("Mjlab-BallKickLeft-Flat-MicroDuck-HD1910", "Mjlab-BallKickLeft-Flat-Backlash-MicroDuck-HD1910"),
                          ("Mjlab-VelStand-Flat-MicroDuck-HD1910", "Mjlab-VelStand-Rough-Backlash-MicroDuck-HD1910"),
                          ("Mjlab-Velocity-Flat-MicroDuck-HD1910", "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910")):
        assert load_rl_cfg(first).experiment_name == load_rl_cfg(second).experiment_name


@pytest.mark.parametrize("name", [n for n in dir(hd1910_robot) if n.startswith("MICRODUCK_") and n.endswith("_ROBOT_CFG")])
def test_every_hd1910_robot_has_the_build_mass(name):
    cfg = getattr(hd1910_robot, name)
    assert cfg.spec_fn().compile().body_mass.sum() == pytest.approx(TOTAL_MASS_KG, abs=1e-9)
    (actuator,) = cfg.articulation.actuators
    assert type(actuator) is (Hd1910BacklashActuatorCfg if "BACKLASH" in name else Hd1910ActuatorCfg)
