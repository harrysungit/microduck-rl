"""Independent CPU-forward comparison for full-head GPU support geometry."""
import mujoco
import numpy as np


def support(model,data,g):
    kind=model.geom_type[g]
    n=data.geom_xmat[g].reshape(3,3)[2]
    z=data.geom_xpos[g,2]
    s=model.geom_size[g]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=model.geom_dataid[g]
        start,count=model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        return z+(model.mesh_vert[start:start+count] @ n).min()
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
        return z-np.abs(n)@s
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        return z-s[0]
    if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
        return z-s[0]-s[1]*abs(n[2])
    if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        return z-s[0]*np.linalg.norm(n[:2])-s[1]*abs(n[2])
    if kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return z-np.linalg.norm(n*s)
    raise ValueError(kind)


def verify(env):
    m=env.sim.mj_model
    d=mujoco.MjData(m)
    ids=[row['geom_id'] for row in env.head_clearance.names]
    actual=env.head_clearance(env).cpu().numpy()
    errors=[]
    for i in range(min(env.num_envs,8)):
        d.qpos[:]=env.sim.data.qpos[i].cpu().numpy()
        mujoco.mj_forward(m,d)
        expected=min(support(m,d,g) for g in ids)-env.scene.terrain.env_origins[i,2].item()
        errors.append(abs(expected-actual[i]))
    assert max(errors)<1e-5,errors
    return {'cpu_forward_samples':len(errors),'geometry_count':len(ids),'max_support_abs_error_m':max(errors)}
