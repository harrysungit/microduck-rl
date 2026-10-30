"""HD-1910 robots: the XL330 robot models with the HD-1910 actuator and mass."""

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from microduck_rl.actuators.hd1910.actuator import (
    HD1910_ACTUATOR_KWARGS,
    Hd1910ActuatorCfg,
    Hd1910BacklashActuatorCfg,
)
from microduck_rl.actuators.hd1910.mass import apply_hd1910_mass
from mjlab_microduck.robot.microduck_constants import (
    BACKLASH_HOME_FRAME,
    FULL_COLLISION,
    HOME_FRAME,
    get_allcollisions_backlash_spec,
    get_allcollisions_spec,
    get_backlash_spec,
    get_standup_spec,
    get_walk_backlash_spec,
    get_walk_spec,
)


def get_walk_hd1910_spec():
    return apply_hd1910_mass(get_walk_spec())


def get_walk_backlash_hd1910_spec():
    return apply_hd1910_mass(get_walk_backlash_spec())


MICRODUCK_WALK_HD1910_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_hd1910_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(Hd1910ActuatorCfg(**HD1910_ACTUATOR_KWARGS),),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# ±1° serial play per servo, encoder on the output side (tasks/backlash.py).
MICRODUCK_WALK_BACKLASH_HD1910_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_hd1910_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(Hd1910BacklashActuatorCfg(**HD1910_ACTUATOR_KWARGS),),
        soft_joint_pos_limit_factor=0.9,
    ),
)


def _robot(spec_fn, backlash: bool) -> EntityCfg:
    cfg = Hd1910BacklashActuatorCfg if backlash else Hd1910ActuatorCfg
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=BACKLASH_HOME_FRAME if backlash else HOME_FRAME,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(cfg(**HD1910_ACTUATOR_KWARGS),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def get_groundcontact_hd1910_spec():
    return apply_hd1910_mass(get_standup_spec())


def get_groundcontact_backlash_hd1910_spec():
    return apply_hd1910_mass(get_backlash_spec())


def get_allcollisions_hd1910_spec():
    return apply_hd1910_mass(get_allcollisions_spec())  # keeps the named servo collision geoms


def get_allcollisions_backlash_hd1910_spec():
    return apply_hd1910_mass(get_allcollisions_backlash_spec())


# Ground-contact model: StandUp, SitStand, GroundPick, BallKick, Roulade.
MICRODUCK_GROUNDCONTACT_HD1910_ROBOT_CFG = _robot(get_groundcontact_hd1910_spec, backlash=False)
MICRODUCK_GROUNDCONTACT_BACKLASH_HD1910_ROBOT_CFG = _robot(get_groundcontact_backlash_hd1910_spec, backlash=True)
# Every part colliding: VelStand.
MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG = _robot(get_allcollisions_hd1910_spec, backlash=False)
MICRODUCK_ALLCOLLISIONS_BACKLASH_HD1910_ROBOT_CFG = _robot(get_allcollisions_backlash_hd1910_spec, backlash=True)
