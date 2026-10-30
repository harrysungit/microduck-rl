"""Microduck tasks for servos other than the XL330, registered with mjlab.

mjlab imports this module through the ``mjlab.tasks`` entry point, so ``train``, ``play`` and
export see these tasks next to the XL330 ones from ``mjlab_microduck`` (Pollen Robotics'
microduck_rl), whose task configs they reuse unchanged. Importing it also applies the runtime
patches below.
"""

# HIP/ROCm adjustments for mjlab's simulation; a no-op on CUDA and CPU (see rocm.py).
from microduck_rl.patches.rocm import apply as _apply_rocm

_apply_rocm()

# Throughput fixes for mjlab at thousands of worlds, on any device (see perf.py).
from microduck_rl.patches.perf import apply as _apply_perf

_apply_perf()

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import (
    MicroduckOnPolicyRunner,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
    MicroduckStandUpRlCfg,
    make_microduck_standup_env_cfg,
    MicroduckSitStandRlCfg,
    make_microduck_sitstand_env_cfg,
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
    MicroduckVelStandRlCfg,
    make_microduck_velstand_env_cfg,
)
from mjlab_microduck.tasks.backlash import make_backlash_variant

# Feetech HD-1910 servo variant: the walk robot with the HD-1910
# actuator and mass model; rewards, randomization and curricula as the XL330
# tasks (see actuators/hd1910/).
from dataclasses import replace as _replace

from microduck_rl.actuators.hd1910.robot import (
    MICRODUCK_WALK_BACKLASH_HD1910_ROBOT_CFG,
    MICRODUCK_WALK_HD1910_ROBOT_CFG,
)

MicroduckHd1910RlCfg = _replace(
    MicroduckRlCfg, experiment_name="velocity_hd1910", run_name="velocity_hd1910"
)


def _make_hd1910_velocity_env_cfg(backlash: bool, **kwargs):
    cfg = make_microduck_velocity_env_cfg(**kwargs)
    if backlash:
        cfg = make_backlash_variant(cfg, MICRODUCK_WALK_BACKLASH_HD1910_ROBOT_CFG)
        # 34 DOFs: mujoco_warp picks its sparse path above 32; the dense one is the same physics and
        # 15% faster on gfx1151 (177 against 209 ms per 4096-world step).
        cfg.sim.mujoco.jacobian = "dense"
        return cfg
    cfg.scene.entities = {**cfg.scene.entities, "robot": MICRODUCK_WALK_HD1910_ROBOT_CFG}
    return cfg


HD1910_TASKS = (
    ("Mjlab-Velocity-Flat-MicroDuck-HD1910", {}, False),
    ("Mjlab-Velocity-Rough-MicroDuck-HD1910", {"rough": True}, False),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910", {}, True),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck-HD1910", {"rough": True}, True),
)
for _task_id, _kw, _backlash in HD1910_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_make_hd1910_velocity_env_cfg(_backlash, **_kw),
        play_env_cfg=_make_hd1910_velocity_env_cfg(_backlash, play=True, **_kw),
        rl_cfg=MicroduckHd1910RlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )



# The official skills on the HD-1910 duck: each family's env cfg on the HD-1910 robot of the model it
# uses, backlash twins on the dense Jacobian, and an experiment of its own. A stage that continues
# another (Flat -> Backlash) shares that stage's experiment so a plain resume finds it.
import os as _os

from microduck_rl.actuators.hd1910.robot import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_HD1910_ROBOT_CFG,
    MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG,
    MICRODUCK_GROUNDCONTACT_BACKLASH_HD1910_ROBOT_CFG,
    MICRODUCK_GROUNDCONTACT_HD1910_ROBOT_CFG,
)


def _on_hd1910(cfg, robot=None, backlash_robot=None):
    if backlash_robot is not None:
        cfg = make_backlash_variant(cfg, backlash_robot)
        cfg.sim.mujoco.jacobian = "dense"
        return cfg
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot}
    return cfg


def _hd1910_rl(rl_cfg, name):
    return _replace(rl_cfg, experiment_name=name, run_name=name)


def _hd1910_velstand_rl():
    """VelStand's behaviour cloning from local HD-1910 teachers, never the XL330 ones on wandb.

    HD1910_STAND_EXPERT is a StandUp-HD1910 checkpoint and HD1910_WALK_ANCHOR a Velocity-HD1910
    one; without them, building the algorithm fails on the placeholder path.
    """
    rl = _hd1910_rl(MicroduckVelStandRlCfg, "velstand_hd1910")
    bc = rl.algorithm.bc_cfg
    if bc is not None:
        bc = {**bc, "wandb_run_path": None, "anchor_wandb_run_path": None,
              "checkpoint_path": _os.environ.get("HD1910_STAND_EXPERT", "<set HD1910_STAND_EXPERT>"),
              "anchor_checkpoint_path": _os.environ.get("HD1910_WALK_ANCHOR", "<set HD1910_WALK_ANCHOR>")}
        rl = _replace(rl, algorithm=_replace(rl.algorithm, bc_cfg=bc))
    return rl


_GC, _GC_BL = MICRODUCK_GROUNDCONTACT_HD1910_ROBOT_CFG, MICRODUCK_GROUNDCONTACT_BACKLASH_HD1910_ROBOT_CFG
_AC, _AC_BL = MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG, MICRODUCK_ALLCOLLISIONS_BACKLASH_HD1910_ROBOT_CFG
_GROUND_PICK_RL = _hd1910_rl(MicroduckGroundPickRlCfg, "ground_pick_hd1910")
_KICK_RIGHT_RL = _hd1910_rl(MicroduckBallKickRlCfg, "ball_kick_right_hd1910")
_KICK_LEFT_RL = _hd1910_rl(MicroduckBallKickRlCfg, "ball_kick_left_hd1910")
_VELSTAND_RL = _hd1910_velstand_rl()

# (task id, env cfg factory, factory kwargs, rl cfg, robot, backlash robot)
HD1910_SKILL_TASKS = (
    ("Mjlab-StandUp-Flat-MicroDuck-HD1910", make_microduck_standup_env_cfg, {},
     _hd1910_rl(MicroduckStandUpRlCfg, "microduck_stand_hd1910"), _GC, None),
    ("Mjlab-SitStand-Flat-MicroDuck-HD1910", make_microduck_sitstand_env_cfg, {},
     _hd1910_rl(MicroduckSitStandRlCfg, "microduck_sitstand_hd1910"), _GC, None),
    ("Mjlab-GroundPick-Flat-MicroDuck-HD1910", make_microduck_ground_pick_env_cfg, {}, _GROUND_PICK_RL, _GC, None),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck-HD1910", make_microduck_ground_pick_env_cfg, {}, _GROUND_PICK_RL, None, _GC_BL),
    ("Mjlab-BallKick-Flat-MicroDuck-HD1910", make_microduck_ball_kick_env_cfg, {"kick_foot": "right"}, _KICK_RIGHT_RL, _GC, None),
    ("Mjlab-BallKickLeft-Flat-MicroDuck-HD1910", make_microduck_ball_kick_env_cfg, {"kick_foot": "left"}, _KICK_LEFT_RL, _GC, None),
    ("Mjlab-BallKickLeft-Flat-Backlash-MicroDuck-HD1910", make_microduck_ball_kick_env_cfg, {"kick_foot": "left"},
     _KICK_LEFT_RL, None, _GC_BL),
    ("Mjlab-Roulade-Flat-MicroDuck-HD1910", make_microduck_roulade_env_cfg, {},
     _hd1910_rl(MicroduckRouladeRlCfg, "microduck_roulade_hd1910"), _GC, None),
    ("Mjlab-VelStand-Flat-MicroDuck-HD1910", make_microduck_velstand_env_cfg, {}, _VELSTAND_RL, _AC, None),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck-HD1910", make_microduck_velstand_env_cfg, {}, _VELSTAND_RL, None, _AC_BL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck-HD1910", make_microduck_velstand_env_cfg, {"rough": True},
     _VELSTAND_RL, None, _AC_BL),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot, _bl_robot in HD1910_SKILL_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_on_hd1910(_make_cfg(**_kw), _robot, _bl_robot),
        play_env_cfg=_on_hd1910(_make_cfg(play=True, **_kw), _robot, _bl_robot),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
