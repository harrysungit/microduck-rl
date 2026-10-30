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
