"""Isolated endpoint-reset SitStand experiment; no FK intermediate resets.

This task is a learning experiment, never an acceptance or deployment gate.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import itertools
import torch
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task
from mjlab_microduck.tasks.mdp import SitStandCommand, SitStandCommandCfg

BASE = 'Mjlab-SitStand-Flat-MicroDuck-HD1910'
TASK = BASE + '-HeadSafeEndpointsV2-20261118'
if __package__:
    from .runtime_actions import RuntimeActionCfg, NAMES, SIT
else:
    from runtime_actions import RuntimeActionCfg, NAMES, SIT

HEAD_BODIES = ('neck', 'neck_pitch', 'yaw_roll_motion', 'jaw_soft')
LEGACY_HIP_BODY = 'bearing_roll'  # right hip yaw; upstream DR list retains its historical mistake


def resolve_head_body_names(model, prefix='robot/'):
    root=model.body(prefix+'neck').id
    result=[]
    for body in range(1,model.nbody):
        parent=body
        while parent and parent!=root:parent=int(model.body_parentid[parent])
        if parent==root:result.append(model.body(body).name.removeprefix(prefix))
    if not set(HEAD_BODIES)<=set(result) or LEGACY_HIP_BODY in result:
        raise ValueError('Unexpected neck subtree anatomy; do not use the historical DR list as head scope')
    return tuple(result)


class LatchedContactSensor(ContactSensor):
    """Found (including zero-force contact) and force captured every substep."""
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.ever_found = torch.zeros_like(self.data.found[:, 0], dtype=torch.bool)
        self.substep_count = torch.zeros_like(self.ever_found, dtype=torch.long)

    def update(self, dt):
        super().update(dt)
        self.ever_found |= (self.data.found > 0).any(-1)
        self.substep_count += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'ever_found'):
            index = slice(None) if env_ids is None else env_ids
            self.ever_found[index] = False
            self.substep_count[index] = 0


@dataclass
class LatchedContactCfg(ContactSensorCfg):
    def build(self):
        return LatchedContactSensor(self)


class LegalSupportSensor(ContactSensor):
    """Both sides (hip/upper-leg/shank) or trunk, at every physics substep."""
    def initialize(self,*args,**kwargs):
        super().initialize(*args,**kwargs)
        self.body_order=list(dict.fromkeys(slot.primary_name for slot in self._slots))
        short=[name.split('/')[-1] for name in self.body_order]
        self.left=[short.index(name) for name in ('hip_l','upper_leg_left','leg')]
        self.right=[short.index(name) for name in ('bearing_roll','hip_l_2','upper_leg_right','leg_2')]
        self.trunk=short.index('trunk_base')
        self.valid_history=torch.zeros((self.data.found.shape[0],self.cfg.history_length),dtype=torch.bool,device=self.data.found.device)

    def update(self,dt):
        super().update(dt)
        loaded=(self.data.found>0) & (self.data.force[...,0].abs()>.05)
        valid=loaded[:,self.trunk] | (loaded[:,self.left].any(-1) & loaded[:,self.right].any(-1))
        self.valid_history.copy_(torch.cat((valid[:,None],self.valid_history[:,:-1]),-1))

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'valid_history'):
            self.valid_history[slice(None) if env_ids is None else env_ids]=False


@dataclass
class LegalSupportCfg(ContactSensorCfg):
    def build(self):
        return LegalSupportSensor(self)


class CycleCommand(SitStandCommand):
    """13s hold/opposite/return cycle, binary deployment command unchanged."""
    def compute(self, dt):
        env = self._env_ref
        if not hasattr(env, 'initial_posture'):
            return super().compute(dt)
        t = env.episode_length_buf * env.step_dt
        flag = torch.where((t >= 3.) & (t < 8.), 1-env.initial_posture, env.initial_posture)
        self.vel_command_b.zero_()
        self.vel_command_b[:, 0] = flag
        fresh = env.episode_length_buf <= 1
        self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        step = dt / self._ramp_s
        self._alpha += (flag-self._alpha).clamp(-step, step)


@dataclass(kw_only=True)
class CycleCommandCfg(SitStandCommandCfg):
    def build(self, env):
        return CycleCommand(self, env)


class GeometryClearance:
    """All authored head/neck mesh collision supports against flat ground."""
    def __init__(self, env, body_names=None):
        m = env.sim.mj_model
        robot = env.scene['robot']
        self.parts = []
        self.primitives = []
        self.names = []
        for name in (resolve_head_body_names(m) if body_names is None else body_names):
            body = m.body('robot/'+name).id
            vertices = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != body:
                    continue
                r = np.empty(9)
                mujoco.mju_quat2Mat(r, m.geom_quat[g])
                kind=m.geom_type[g]
                if kind==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=m.geom_dataid[g]
                    start,count=m.mesh_vertadr[mesh],m.mesh_vertnum[mesh]
                    v=m.mesh_vert[start:start+count]
                    vertices.append(v @ r.reshape(3,3).T+m.geom_pos[g])
                elif kind==mujoco.mjtGeom.mjGEOM_BOX:
                    v=np.array(list(itertools.product((-1.,1.),repeat=3)))*m.geom_size[g]
                    vertices.append(v @ r.reshape(3,3).T+m.geom_pos[g])
                elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    self.primitives.append((robot.body_names.index(name),int(kind),*[torch.tensor(a.copy(),dtype=torch.float32,device=env.device) for a in (m.geom_pos[g],r.reshape(3,3),m.geom_size[g])]))
                else:
                    raise ValueError('Unsupported head geometry: '+str(kind))
                self.names.append({'body':name,'geom_id':g,'geom_name':m.geom(g).name})
            if vertices:
                v = np.concatenate(vertices)
                v = v[ConvexHull(v).vertices]
                self.parts.append((robot.body_names.index(name), torch.tensor(v.T.copy(), dtype=torch.float32, device=env.device)))
        assert (self.parts or self.primitives) and self.names

    def __call__(self, env):
        d = env.scene['robot'].data
        values = []
        for body, points in self.parts:
            w,x,y,z = d.body_link_quat_w[:,body].unbind(-1)
            n = torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), -1)
            values.append(d.body_link_pos_w[:,body,2] + (n @ points).amin(-1))
        for body,kind,pos,rotation,size in self.primitives:
            w,x,y,z=d.body_link_quat_w[:,body].unbind(-1)
            n=torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
            local=n @ rotation
            center=d.body_link_pos_w[:,body,2]+n @ pos
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
                support=size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
                support=size[0]+size[1]*local[:,2].abs()
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                support=size[0]*local[:,:2].norm(dim=-1)+size[1]*local[:,2].abs()
            else:
                support=(local*size).norm(dim=-1)
            values.append(center-support)
        return torch.stack(values,-1).amin(-1)-env.scene.terrain.env_origins[:,2]


def state(env):
    """Update once per control step; termination executes before reward."""
    if getattr(env, '_safety_tick', -1) == env.common_step_counter:
        return env.safety
    env._safety_tick = env.common_step_counter
    d = env.scene['robot'].data
    s = env.safety
    command = env.command_manager.get_command('twist')[:,0]
    z = d.root_link_pos_w[:,2]-env.scene.terrain.env_origins[:,2]
    upright = (-d.projected_gravity_b[:,2]).clamp(-1,1)
    slow = (d.root_link_lin_vel_w[:,2].abs()<.02) & (d.root_link_ang_vel_b.norm(dim=-1)<.5)
    # Endpoints use measured heights; classification also requires uprightness.
    stand = (z-.1158).abs()<.012
    sit = (z-.0615).abs()<.012
    joints=d.joint_pos[:,env.motor_ids]
    home=d.default_joint_pos[:,env.motor_ids]
    sitting=home.clone()
    for idx,value in SIT.items():
        sitting[:,idx]=value
    legs=[0,1,2,3,4,9,10,11,12,13]
    stand &= (joints[:,legs]-home[:,legs]).abs().amax(-1)<.3
    sit &= (joints[:,legs]-sitting[:,legs]).abs().amax(-1)<.3
    legal_support=env.scene['sit_support'].valid_history.all(-1)
    sit &= legal_support
    env.extras.setdefault('log',{})['Task/legal_sit_support_rate']=legal_support.float().mean()
    posture = torch.where(sit, 1., torch.where(stand, 0., -1.))
    stable = (posture>=0) & (upright>np.cos(np.deg2rad(15))) & slow
    changed = command != s['command']
    # Capture pre-switch evidence before resetting the stable counter.
    s['eligible'] = torch.where(changed, (s['stable_s']>=.5) & (s['posture']==1-command), s['eligible'])
    s['credited'] &= ~changed
    same = stable & (posture==s['posture'])
    s['stable_s'] = torch.where(same, s['stable_s']+env.step_dt, torch.zeros_like(z))
    s['posture'] = posture
    s['command'] = command.clone()
    s['failed'] |= env.scene['head_ground'].ever_found
    s['success_pulse'] = s['eligible'] & ~s['credited'] & (s['stable_s']>=1.) & (posture==command) & ~s['failed']
    s['credited'] |= s['success_pulse']
    s['success_count'] += s['success_pulse'].long()
    s['z'], s['upright'] = z, upright
    env.extras.setdefault('log', {})['Safety/head_contact_latched_rate'] = s['failed'].float().mean()
    env.extras['log']['Task/valid_transition_count'] = s['success_count'].float().mean()
    return s


def head_failure(env):
    return state(env)['failed']


def height_tracking(env):
    s = state(env)
    alpha = env.command_manager.get_term('twist').alpha
    target = .1158 + alpha*(.0615-.1158)
    # No interpolated joint reference: an infeasible FK blend is not rewarded.
    return torch.exp(-((s['z']-target)/.025).square()) * s['upright'].clamp(min=0).pow(4) * (~s['failed'])


def endpoint_pose(env):
    s = state(env)
    d = env.scene['robot'].data
    ids = env.motor_ids
    target = d.default_joint_pos[:,ids].clone()
    for idx,value in SIT.items():
        target[:,idx] += s['command']*(value-target[:,idx])
    legs = [0,1,2,3,4,9,10,11,12,13]
    error = (d.joint_pos[:,ids][:,legs]-target[:,legs]).square().mean(-1)
    target_z = .1158+s['command']*(.0615-.1158)
    near = (s['z']-target_z).abs()<.015
    return torch.exp(-error/.16)*near*s['upright'].clamp(min=0).pow(4)*(~s['failed'])


def transition_bonus(env):
    s=state(env)
    return (s['success_pulse'] & ~s['failed']).float()


def failure_cost(env):
    return state(env)['failed'].float()


def clearance_cost(env):
    gap = env.head_clearance(env)
    env.extras.setdefault('log', {})['Safety/full_head_clearance_min_m'] = gap.min()
    legacy=env.legacy_hip_clearance(env)
    env.extras['log']['Safety/legacy_right_hip_clearance_min_m']=legacy.min()
    env.extras['log']['Safety/historical_head_superset_clearance_min_m']=torch.minimum(gap,legacy).min()
    return ((.012-gap).clamp(min=0,max=.04)/.012).square()


class HeadSafeEnv(ManagerBasedRlEnv):
    def _reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        robot = self.scene['robot']
        if not hasattr(self, 'initial_posture'):
            self.motor_ids = [robot.joint_names.index(n) for n in NAMES]
            self.initial_posture = torch.zeros(self.num_envs,device=self.device)
            z = torch.zeros(self.num_envs,device=self.device)
            self.safety = {k:z.clone() for k in ('stable_s','posture','command','z','upright')}
            self.safety.update({k:torch.zeros_like(z,dtype=torch.bool) for k in ('eligible','credited','failed','success_pulse')})
            self.safety['success_count'] = torch.zeros_like(z,dtype=torch.long)
            self.head_clearance = GeometryClearance(self)
            self.legacy_hip_clearance = GeometryClearance(self,body_names=(LEGACY_HIP_BODY,))
        n = len(env_ids)
        posture = (torch.rand(n,device=self.device)<.5).float()
        if hasattr(self,'initial_pose_override'):
            posture.fill_(self.initial_pose_override)
        self.initial_posture[env_ids] = posture
        q = robot.data.default_joint_pos[env_ids].clone()
        for idx,value in SIT.items():
            joint = self.motor_ids[idx]
            q[:,joint] += posture*(value-q[:,joint])
        robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
        root = torch.zeros(n,13,device=self.device)
        root[:,:3] = self.scene.terrain.env_origins[env_ids]
        root[:,2] += .1158+posture*(.0615-.1158)+.002
        root[:,3] = 1.
        robot.write_root_state_to_sim(root,env_ids=env_ids)
        for value in self.safety.values():
            value[env_ids] = 0
        self.safety['posture'][env_ids] = posture
        self.safety['command'][env_ids] = posture
        self._safety_tick = -1


def register(num_envs=64, calibration=None):
    if calibration is None:
        raise ValueError('An external device calibration is required')
    cfg = load_env_cfg(BASE)
    from microduck_rl.actuators.hd1910.robot import MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG
    cfg.scene.entities['robot']=MICRODUCK_ALLCOLLISIONS_HD1910_ROBOT_CFG
    cfg.seed = 181126
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = num_envs
    cfg.episode_length_s = 13.
    cfg.sim.njmax, cfg.sim.nconmax = 320,256
    cfg.scene.sensors = (*cfg.scene.sensors, LatchedContactCfg(name='head_ground',primary=ContactMatch(mode='subtree',pattern='neck',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force','dist'),reduce='maxforce',num_slots=1,history_length=cfg.decimation))
    cfg.scene.sensors=(*cfg.scene.sensors,LatchedContactCfg(name='legacy_hip_ground',primary=ContactMatch(mode='body',pattern='bearing_roll',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force','dist'),reduce='maxforce',num_slots=1,history_length=cfg.decimation))
    cfg.scene.sensors=(*cfg.scene.sensors,LegalSupportCfg(name='sit_support',primary=ContactMatch(mode='body',pattern=r'^(trunk_base|hip_l|upper_leg_left|leg|bearing_roll|hip_l_2|upper_leg_right|leg_2)$',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='maxforce',num_slots=1,history_length=cfg.decimation))
    cfg.commands['twist'] = CycleCommandCfg(**vars(cfg.commands['twist']))
    cfg.actions['joint_pos'] = RuntimeActionCfg(**asdict(cfg.actions['joint_pos']), calibration=str(Path(calibration).resolve()))
    # First bounded experiment: nominal dynamics, fixed neutral head. Capability
    # evidence, not update count, is required before adding disturbances.
    keep = {'expand_bam_friction_fields','reset_action_history'}
    cfg.events = {k:v for k,v in cfg.events.items() if k in keep}
    cfg.curriculum = {}
    head = cfg.commands['head_pose']
    if hasattr(head,'ranges'):
        head.ranges = ((0.,0.),)*4
    cfg.rewards = {
        'height_tracking':RewardTermCfg(func=height_tracking,weight=4.),
        'endpoint_pose':RewardTermCfg(func=endpoint_pose,weight=1.),
        'valid_transition':RewardTermCfg(func=transition_bonus,weight=100.),
        'head_failure':RewardTermCfg(func=failure_cost,weight=-25.),
        'clearance':RewardTermCfg(func=clearance_cost,weight=-.3),
    }
    # Low regularization during action discovery; no iteration-based ramp.
    from mjlab.tasks.velocity import mdp
    cfg.rewards['action_rate'] = RewardTermCfg(func=mdp.action_rate_l2,weight=-.01)
    cfg.terminations['head_contact'] = TerminationTermCfg(func=head_failure,time_out=False)
    agent = load_rl_cfg(BASE)
    agent.seed = cfg.seed
    agent.logger='tensorboard'
    agent.upload_model=False
    agent.algorithm.learning_rate=3e-4
    agent.algorithm.schedule='adaptive'
    agent.save_interval=100
    agent.actor.distribution_cfg['init_std']=.3
    register_mjlab_task(TASK,cfg,cfg,agent,load_runner_cls(BASE))
    return cfg,agent
