"""CPU FK endpoint reset heights from actual floor-compatible collision geometry."""
import mujoco
import numpy as np
if __package__:
    from .verify_geometry import support
else:
    from verify_geometry import support


def subtree(model,root):
    bodies=set()
    for body in range(1,model.nbody):
        ancestor=body
        while ancestor and ancestor!=root:ancestor=int(model.body_parentid[ancestor])
        if ancestor==root:bodies.add(body)
    return bodies


def ground_spawn(model,joint_targets,prefix='robot/',clearance=.002,floor_name=None):
    if not np.isfinite(clearance) or clearance<=0:raise ValueError('Positive finite initial clearance required')
    root=model.body(prefix+'trunk_base').id
    bodies=subtree(model,root)
    data=mujoco.MjData(model)
    free=[j for j in range(model.njnt) if model.jnt_bodyid[j]==root and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    if len(free)!=1:raise ValueError('A unique free trunk joint is required')
    adr=int(model.jnt_qposadr[free[0]])
    data.qpos[adr:adr+7]=[0,0,.15,1,0,0,0]
    for name,value in joint_targets.items():
        if not np.isfinite(value):raise ValueError('Nonfinite endpoint joint target')
        joint=model.joint(name if name.startswith(prefix) else prefix+name).id
        if model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError('Endpoint targets require scalar hinge/slide joints')
        data.qpos[model.jnt_qposadr[joint]]=value
    mujoco.mj_kinematics(model,data)
    if floor_name is None:
        floors=[g for g in range(model.ngeom) if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[g] not in bodies]
        if len(floors)!=1:raise ValueError('Exactly one external flat floor plane is required')
        floor=floors[0]
    else:floor=model.geom(floor_name).id
    if model.geom_type[floor]!=mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(data.geom_xmat[floor].reshape(3,3)[2],(0,0,1),atol=1e-8):
        raise ValueError('Endpoint spawn requires a horizontal plane')
    explicit=set()
    for first,second in zip(model.pair_geom1,model.pair_geom2):
        if floor in (first,second):explicit.add(int(second if first==floor else first))
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and
           (g in explicit or (int(model.geom_contype[g])&int(model.geom_conaffinity[floor])) or
            (int(model.geom_contype[floor])&int(model.geom_conaffinity[g])))]
    if not geoms:raise ValueError('No robot geometry can contact the actual floor')
    floor_z=float(data.geom_xpos[floor,2])
    minimum=min(support(model,data,g)-floor_z for g in geoms)
    data.qpos[adr+2]+=clearance-minimum
    mujoco.mj_forward(model,data)
    gaps={int(g):float(support(model,data,g)-floor_z) for g in geoms}
    final=min(gaps.values())
    if abs(final-clearance)>1e-7:raise ValueError('FK spawn clearance check failed')
    return {'root_z_above_floor_m':float(data.qpos[adr+2]-floor_z),'minimum_clearance_m':final,
            'minimum_geometry_ids':[g for g,gap in gaps.items() if abs(gap-final)<1e-7],
            'floor_geometry':model.geom(floor).name,'floor_contact_geometry_count':len(geoms),
            'all_floor_contact_geometry_gaps_m':gaps,'root_qpos_address':adr,'joint_targets':dict(joint_targets)}
