"""Measured full-collision endpoints and immutable training bounds.

The groundcontact-era SIT target .0615 is historical. These defaults describe
allcollisions fixed-control holds, not a relaxation of independent CPU criteria.
"""
from dataclasses import asdict,dataclass
import math

CPU_SIT_Z_MAX=.075
CPU_STAND_Z_MIN=.10

@dataclass(frozen=True)
class PostureGoals:
    stand_z:float=.11596
    sit_z:float=.06505
    sit_success_z_max:float=CPU_SIT_Z_MAX
    stand_success_z_min:float=CPU_STAND_Z_MIN
    spawn_clearance_m:float=.002
    endpoint_height_tolerance_m:float=.015

    def __post_init__(self):
        if not all(type(value) in (int,float) and math.isfinite(value) for value in asdict(self).values()):
            raise ValueError('Posture goals must be finite real numbers')
        if self.sit_success_z_max>CPU_SIT_Z_MAX or self.stand_success_z_min<CPU_STAND_Z_MIN:
            raise ValueError('Training bounds cannot relax the independent CPU posture limits')
        if not (0<self.sit_z<=self.sit_success_z_max<self.stand_success_z_min<=self.stand_z):
            raise ValueError('Endpoint targets and success bounds are inconsistent')
        if self.spawn_clearance_m<=0 or self.endpoint_height_tolerance_m<=0:
            raise ValueError('Spawn clearance and endpoint tolerance must be positive')

    def record(self):
        return {**asdict(self),'scene':'allcollisions','measurement_scope':'fixed-control HD1910 HOME/SIT endpoint holds',
                'measured_sit_equilibrium_range_m':[.064832,.065127],
                'measured_stand_equilibrium_range_m':[.11562,.11596],
                'initialization':'FK minimum floor-contact-compatible geometry + clearance; not reward goal + clearance',
                'cpu_limits_changed':False}

DEFAULT_GOALS=PostureGoals()

def target_height(alpha,goals=DEFAULT_GOALS):
    return goals.stand_z+alpha*(goals.sit_z-goals.stand_z)

def posture_height_masks(z,goals=DEFAULT_GOALS):
    return z>=goals.stand_success_z_min,z<=goals.sit_success_z_max

def endpoint_height_mask(z,command,goals=DEFAULT_GOALS):
    near=abs(z-target_height(command,goals))<goals.endpoint_height_tolerance_m
    stand,sit=posture_height_masks(z,goals)
    return near & (((command>.5)&sit)|((command<=.5)&stand))
