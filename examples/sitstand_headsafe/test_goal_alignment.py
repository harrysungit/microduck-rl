"""CPU contracts for measured goals, collision-compatible FK reset, and force frames."""
from dataclasses import asdict,FrozenInstanceError
from pathlib import Path
import sys
import types
import numpy as np
import mujoco
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goals import DEFAULT_GOALS,PostureGoals,endpoint_height_mask,posture_height_masks,target_height
from endpoint_geometry import ground_spawn
from support_forces import loaded_support_bodies


def test_all_endpoint_height_consumers_use_one_immutable_measured_config(monkeypatch):
    import headsafe_task as task
    monkeypatch.setattr(task,'register_mjlab_task',lambda *args:None)
    goals=PostureGoals()
    cfg,_=task.register(2,'/external/synthetic-test-only.toml',goals)
    assert cfg.posture_goals==goals
    assert asdict(cfg)['posture_goals']==asdict(goals)
    assert cfg.commands['twist'].stand_z==goals.stand_z==.11596
    assert cfg.commands['twist'].sit_z==goals.sit_z==.06505
    assert target_height(0.,goals)==goals.stand_z and target_height(1.,goals)==goals.sit_z
    with pytest.raises(FrozenInstanceError):goals.sit_z=.0615
    source=Path(task.__file__).read_text()
    assert '.0615' not in source and '.1158' not in source
    assert 'root[:,2] += torch.where' in source
    sensors={sensor.name:sensor for sensor in cfg.scene.sensors}
    assert sensors['sit_support'].reduce=='netforce' and not sensors['sit_support'].global_frame
    assert sensors['sit_support_max_contact'].reduce=='maxforce'


def test_sit_success_and_endpoint_reward_never_drift_to_point077():
    z=np.array([.06505,.07499,.075,.075001,.076,.077,.10,.11596])
    stand,sit=posture_height_masks(z)
    assert sit.tolist()==[True,True,True,False,False,False,False,False]
    assert not endpoint_height_mask(z,np.ones_like(z))[3:].any()
    assert endpoint_height_mask(z,np.ones_like(z))[:3].all()
    assert stand[-2:].all()
    with pytest.raises(ValueError,match='cannot relax'):PostureGoals(sit_success_z_max=.077)
    with pytest.raises(ValueError,match='cannot relax'):PostureGoals(stand_success_z_min=.099)
    with pytest.raises(ValueError):PostureGoals(sit_z=float('nan'))


def test_fk_spawn_uses_floor_compatible_geometry_not_old_goal_or_visual_bounds():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1" contype="1" conaffinity="1"/>
      <body name="trunk_base"><freejoint/><geom type="sphere" size=".01" mass=".1"/>
      <geom name="visual_only" type="box" pos="0 0 -1" size=".1 .1 .1" contype="0" conaffinity="0"/>
      <body name="leg"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="contact_leg" type="box" pos="0 0 -.1" size=".02 .02 .03" mass=".1"/>
      </body></body></worldbody></mujoco>''')
    before=model.qpos0.copy()
    home=ground_spawn(model,{'hinge':0.},prefix='')
    sit=ground_spawn(model,{'hinge':1.},prefix='')
    assert home['root_z_above_floor_m']==pytest.approx(.132)
    assert home['minimum_clearance_m']==pytest.approx(.002)
    assert sit['minimum_clearance_m']==pytest.approx(.002)
    assert sit['root_z_above_floor_m']!=pytest.approx(DEFAULT_GOALS.sit_z+.002)
    assert model.geom('visual_only').id not in home['all_floor_contact_geometry_gaps_m']
    np.testing.assert_array_equal(model.qpos0,before)


def contact_model(mass=.2):
    spec=mujoco.MjSpec.from_string(f'''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="1 1 .1" friction="1 .005 .0001"/>
      <body name="leg" pos="0 0 .021"><freejoint/>
      <geom name="box" type="box" size=".02 .02 .02" mass="{mass}" friction="1 .005 .0001"/>
      </body></worldbody></mujoco>''')
    for name,reduce,bits in [('net',3,2),('max',2,2),('found',3,1)]:
        spec.add_sensor(name=name,type=mujoco.mjtSensor.mjSENS_CONTACT,objtype=mujoco.mjtObj.mjOBJ_BODY,
          objname='leg',reftype=mujoco.mjtObj.mjOBJ_GEOM,refname='floor',intprm=[bits,reduce,1])
    return spec.compile()


@pytest.mark.parametrize('velocity',[0.,.2])
def test_native_netforce_world_z_matches_sum_of_cpu_contact_normals(velocity):
    model=contact_model();data=mujoco.MjData(model)
    for _ in range(500):
        if velocity:data.qvel[0]=velocity
        mujoco.mj_step(model,data)
    normal=[]
    for c in range(data.ncon):
        force=np.zeros(6);mujoco.mj_contactForce(model,data,c,force);normal.append(max(0.,force[0]))
    net=data.sensor('net').data;maximum=data.sensor('max').data
    assert abs(net[2])==pytest.approx(sum(normal),abs=1e-9)
    assert maximum[0]==pytest.approx(max(normal),abs=1e-9)
    assert abs(net[2])>maximum[0]
    if velocity:assert np.linalg.norm(net)>abs(net[2])+.01
    assert loaded_support_bodies(np.array([data.sensor('found').data[0]]),net[None])[0]


def test_body_sum_load_can_qualify_while_each_contact_is_below_threshold():
    model=contact_model(.012);data=mujoco.MjData(model)
    for _ in range(500):mujoco.mj_step(model,data)
    assert data.sensor('max').data[0]<.05<abs(data.sensor('net').data[2])
    assert loaded_support_bodies(np.array([1.]),data.sensor('net').data[None])[0]
    assert not loaded_support_bodies(np.array([0.]),data.sensor('net').data[None])[0]
    assert not loaded_support_bodies(np.array([1.]),np.array([[3.,4.,.01]]))[0]


def test_late_sit_support_reward_gate_does_not_block_stand_or_early_transition():
    from support_forces import late_sit_support_mask
    import torch
    command=torch.tensor([0.,1.,1.,1.])
    alpha=torch.tensor([0.,.5,.99,1.])
    history=torch.tensor([[False,False],[False,False],[True,False],[True,True]])
    assert late_sit_support_mask(command,alpha,history).tolist()==[True,True,False,True]


def test_runtime_config_types_and_goal_records_match_constructed_implementations(monkeypatch):
    import headsafe_task as task
    from runtime_actions import RuntimeAction,RuntimeActionCfg
    from runtime_identity import object_identity
    monkeypatch.setattr(task,'register_mjlab_task',lambda *args:None)
    cfg,_=task.register(2,'/external/synthetic-test-only.toml')
    assert cfg.commands['twist'].class_type is task.CycleCommand
    assert cfg.actions['joint_pos'].class_type is RuntimeAction
    assert cfg.sit_endpoint_requires_continuous_support is True
    observed=object_identity(DEFAULT_GOALS)
    assert observed['class'].endswith('.PostureGoals')
    assert '/' not in observed['definition_source_file']
    assert len(observed['definition_source_sha256'])==64
