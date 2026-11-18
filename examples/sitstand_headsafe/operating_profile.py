"""Explicit observation/physical uncertainty profiles; no step-based ramp."""
from copy import deepcopy
import math
from mjlab.tasks.registry import load_env_cfg

BASE='Mjlab-SitStand-Flat-MicroDuck-HD1910'
# Preserve the archived five-body DR distribution, without calling the right hip head anatomy.
LEGACY_HEAD_DR_BODIES=('neck','neck_pitch','yaw_roll_motion','jaw_soft','bearing_roll')


def apply(cfg,profile='nominal'):
    assert profile in ('nominal','operating')
    angle=2. if profile=='operating' else 0.
    changed=[]
    for name,term in cfg.observations['actor'].terms.items():
        if 'max_angle_deg' in term.params:
            term.params['max_angle_deg']=angle
            changed.append(name)
    assert len(changed)==2,changed
    record={'profile':profile,'imu_misalignment_max_deg':angle,'pushes':False,'curriculum':'fixed_by_declared_profile_no_iteration_ramp','observation_imu_terms':changed}
    if profile=='nominal':
        return record
    base=load_env_cfg(BASE)
    keep=('randomize_com','randomize_head_com','randomize_mass_inertia','randomize_joint_friction','randomize_armature','encoder_bias','foot_friction')
    for name in keep:
        assert name in base.events,name
        cfg.events[name]=deepcopy(base.events[name])
    cfg.events['randomize_com'].params['ranges']=(-.003,.003)
    cfg.events['randomize_head_com'].params['ranges']=(-.003,.003)
    cfg.events['randomize_head_com'].params['asset_cfg'].body_names=LEGACY_HEAD_DR_BODIES
    cfg.events['randomize_mass_inertia'].params['alpha_range']=(math.log(.95)/2,math.log(1.05)/2)
    cfg.events['randomize_joint_friction'].params['scale_range']=(.9,1.1)
    cfg.events['randomize_armature'].params['ranges']=(.9,1.1)
    cfg.events['encoder_bias'].params['bias_range']=(-.0075,.0075)
    cfg.events['foot_friction'].params['ranges']=(.7,1.3)
    assert not any('push' in key or 'tip' in key or 'jerk' in key for key in cfg.events)
    record.update(trunk_com_range_m=[-.003,.003],head_com_range_m=[-.003,.003],head_com_bodies=list(LEGACY_HEAD_DR_BODIES),head_com_scope_note='historical DR superset includes right hip bearing_roll; anatomical head is the neck subtree',trunk_mass_scale=[.95,1.05],joint_friction_scale=[.9,1.1],armature_scale=[.9,1.1],encoder_bias_rad=[-.0075,.0075],foot_friction=[.7,1.3],reset_randomization_baseline='stock dr uses compile-time defaults; BAM friction restores 1 before sampling')
    return record
