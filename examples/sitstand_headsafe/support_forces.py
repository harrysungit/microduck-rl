"""Flat-floor normal-load gate for native netforce world-frame sensor outputs."""
def loaded_support_bodies(found,force_world,threshold=.05):
    # Native mjSENS_CONTACT netforce Z can be negative for primary body;
    # abs(world Z) is the sum of floor normal magnitudes, without lateral friction.
    normal=abs(force_world[...,2])
    return (found>0)&(normal>threshold)&(normal<float('inf'))


def late_sit_support_mask(command,alpha,valid_history):
    """Gate only settled SIT endpoint reward; every current physics substep must qualify."""
    completed_sit=(command>.5)&(alpha>=.98)
    return (~completed_sit)|valid_history.all(-1)
