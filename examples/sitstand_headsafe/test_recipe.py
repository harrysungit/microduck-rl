"""CPU-only contract tests. Fixtures are synthetic, never a device calibration."""
from pathlib import Path
import importlib
import json
import sys
import types
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibration import NAMES,load_calibration
from runtime_state import TargetState,VoltageState
from bootstrap import bind_inputs,require_resume_identity,validate_anchor
from teacher_init import freeze_normalizer,assert_normalizer_unchanged


def calibration_file(tmp_path):
    path=tmp_path/'synthetic.toml'
    path.write_text('\n'.join(f'[[joints]]\nname="{name}"\nmin_rad=-1.0\nmax_rad=1.0\nzero_raw=2048\ndirection={1 if i%2 else -1}\n' for i,name in enumerate(NAMES)))
    return path


def test_import_task_does_not_read_calibration(monkeypatch):
    import calibration
    monkeypatch.setattr(calibration,'load_calibration',lambda *a:pytest.fail('Import read calibration'))
    module=importlib.import_module('headsafe_task')
    assert callable(module.register)


def test_calibration_rejects_duplicates_and_invalid_direction(tmp_path):
    path=calibration_file(tmp_path)
    assert len(load_calibration(path))==14
    original=path.read_text()
    path.write_text(original.replace('direction=-1','direction=0',1))
    with pytest.raises(ValueError):load_calibration(path)
    path.write_text(original+original)
    with pytest.raises(ValueError,match='Duplicate'):load_calibration(path)


def test_runtime_pipeline_matches_numpy_reference_with_partial_reset(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    from runtime_targets import TargetTransform
    path=calibration_file(tmp_path)
    home=np.linspace(-.2,.2,14)
    state=TargetState(2,home,[-1.]*14,[1.]*14)
    voltage=VoltageState(2)
    refs=[TargetTransform(NAMES,home,path) for _ in range(2)]
    rows=load_calibration(path)
    zero=torch.tensor([r['zero_raw'] for r in rows]);direction=torch.tensor([r['direction'] for r in rows])
    rng=np.random.default_rng(17)
    ticks=np.zeros(2,dtype=int)
    for i in range(200):
        if i==80:
            state.reset(torch.tensor([1]));voltage.reset(torch.tensor([1]));ticks[1]=0
            refs[1]=TargetTransform(NAMES,home,path)
        raw=rng.normal(0,.8,(2,14)).astype(np.float32)
        supply=np.array([7.2+np.sin(i)*.3,np.nan if i%7==0 else 7.7])
        voltage.update(torch.tensor(supply),torch.tensor(ticks%50==0))
        target=state.step(torch.tensor(raw),voltage.multiplier)
        encoder=zero+direction*target*(4096/(2*torch.pi))
        encoder=torch.sign(encoder)*torch.floor(encoder.abs()+.5)
        actual=((encoder-zero)*direction*(2*torch.pi/4096)).numpy()
        expected=np.array([ref(home+raw[j],supply[j]) for j,ref in enumerate(refs)])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        ticks+=1


def test_frozen_normalizer_survives_explicit_checkpoint_restore():
    from rsl_rl.modules import EmpiricalNormalization
    norm=EmpiricalNormalization(3);norm._mean.fill_(.2);norm._std.fill_(.7);norm._var.fill_(.49)
    policy=types.SimpleNamespace(obs_normalizer=norm)
    frozen=freeze_normalizer(policy)
    norm.train();norm.update(torch.randn(8,3));assert_normalizer_unchanged(policy,frozen)
    restored=EmpiricalNormalization(3);restored.load_state_dict(norm.state_dict())
    assert restored.until is None
    policy.obs_normalizer=restored;freeze_normalizer(policy)
    restored.train();restored.update(torch.randn(8,3));assert_normalizer_unchanged(policy,frozen)


def test_resume_binds_teacher_anchor_calibration_sources_and_settings(tmp_path):
    teacher=tmp_path/'teacher';anchor=tmp_path/'anchor';cal=tmp_path/'cal';source=tmp_path/'source.py'
    for path in (teacher,anchor,cal,source):path.write_bytes(path.name.encode())
    binding=bind_inputs(teacher,anchor,cal,[source],{'std':.05})
    assert require_resume_identity({'recipe':binding},binding)
    anchor.write_bytes(b'changed')
    changed=bind_inputs(teacher,anchor,cal,[source],{'std':.05})
    with pytest.raises(ValueError,match='identity'):require_resume_identity({'recipe':binding},changed)
    with pytest.raises(ValueError):require_resume_identity(None,binding)


def test_anchor_label_mismatch_cannot_bind_to_teacher(tmp_path,monkeypatch):
    import onnxruntime
    class Session:
        def __init__(self,*args,**kwargs):pass
        def get_inputs(self):return [types.SimpleNamespace(name='obs')]
        def run(self,*args):return [np.zeros((1,14),dtype=np.float32)]
    monkeypatch.setattr(onnxruntime,'InferenceSession',Session)
    path=tmp_path/'anchor.npz'
    np.savez(path,observations=np.zeros((3,61)),actions=np.zeros((3,14)))
    assert validate_anchor('teacher',path)[2]['samples']==3
    np.savez(path,observations=np.zeros((3,61)),actions=np.ones((3,14)))
    with pytest.raises(ValueError,match='do not match'):validate_anchor('teacher',path)
    np.savez(path,observations=np.zeros((0,61)),actions=np.zeros((0,14)))
    with pytest.raises(ValueError,match='nonempty'):validate_anchor('teacher',path)

def test_package_import_is_opt_in_and_does_not_require_device_files(monkeypatch):
    # Also support normal package imports, not only executing a script by path.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    package=importlib.import_module('examples.sitstand_headsafe')
    assert not hasattr(package,'CALIBRATION')
    task=importlib.import_module('examples.sitstand_headsafe.headsafe_task')
    assert callable(task.register)


def test_profile_changes_real_observation_and_event_parameters(monkeypatch,tmp_path):
    import headsafe_task as task
    from operating_profile import apply
    monkeypatch.setattr(task,'register_mjlab_task',lambda *args:None)
    nominal,_=task.register(2,calibration_file(tmp_path))
    record=apply(nominal,'nominal')
    terms=nominal.observations['actor'].terms
    assert len(record['observation_imu_terms'])==2
    assert all(terms[name].params['max_angle_deg']==0 for name in record['observation_imu_terms'])
    operating,_=task.register(2,calibration_file(tmp_path))
    record=apply(operating,'operating')
    assert all(operating.observations['actor'].terms[name].params['max_angle_deg']==2 for name in record['observation_imu_terms'])
    assert operating.events['randomize_com'].params['ranges']==(-.003,.003)
    assert operating.events['encoder_bias'].params['bias_range']==(-.0075,.0075)
    assert not any('push' in name for name in operating.events)
    sensors={sensor.name:sensor for sensor in operating.scene.sensors}
    assert sensors['head_ground'].primary.mode=='subtree' and sensors['head_ground'].primary.pattern=='neck'
    assert sensors['legacy_hip_ground'].primary.pattern=='bearing_roll'
    assert 'bearing_roll' in sensors['sit_support'].primary.pattern
    assert 'upper_leg_right' in sensors['sit_support'].primary.pattern


def test_actor_warmup_runs_after_anchor_and_prevents_adam_step():
    from teacher_init import actor_warmup
    actor=torch.nn.Linear(2,1);critic=torch.nn.Linear(2,1)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=.1)
    step=[24]
    def hook(*args):
        actor(torch.ones(1,2)).square().mean().backward()
        actor_warmup(actor,step[0],10,24)
    optimizer.register_step_pre_hook(hook)
    initial=[p.detach().clone() for p in actor.parameters()]
    (actor(torch.ones(1,2)).square().mean()+critic(torch.ones(1,2)).square().mean()).backward()
    optimizer.step()
    assert all(torch.equal(a,p) for a,p in zip(initial,actor.parameters()))
    assert all(p not in optimizer.state for p in actor.parameters())
    optimizer.zero_grad();step[0]=264
    (actor(torch.ones(1,2)).square().mean()+critic(torch.ones(1,2)).square().mean()).backward()
    optimizer.step()
    assert any(not torch.equal(a,p) for a,p in zip(initial,actor.parameters()))


def test_cli_help_is_valid_without_gpu_or_device_inputs():
    import subprocess
    for name in ('train_headsafe.py','collect_anchor.py'):
        process=subprocess.run([sys.executable,str(Path(__file__).with_name(name)),'--help'],capture_output=True,text=True)
        assert process.returncode==0,process.stderr
        assert '--calibration' in process.stdout and '--profile' in process.stdout


def test_anatomical_head_tree_excludes_historical_right_hip():
    import mujoco
    from headsafe_task import resolve_head_body_names
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="trunk_base">
      <body name="neck"><body name="neck_pitch"><body name="yaw_roll_motion"><body name="jaw_soft">
      <body name="future_head_part"/></body></body></body></body><body name="bearing_roll"/>
      </body></worldbody></mujoco>''')
    assert set(resolve_head_body_names(model,prefix=''))=={'neck','neck_pitch','yaw_roll_motion','jaw_soft','future_head_part'}


def test_recipe_true_head_and_legacy_hip_primitive_clearances_are_separate():
    import mujoco
    from headsafe_task import GeometryClearance
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="robot/trunk_base" pos="0 0 .2">
      <body name="robot/neck" pos="0 0 .15"><geom type="box" size=".02 .02 .02"/>
      <body name="robot/neck_pitch" pos="0 0 .03"><geom type="sphere" size=".01"/>
      <body name="robot/yaw_roll_motion" pos="0 0 .03"><geom type="cylinder" size=".01 .01"/>
      <body name="robot/jaw_soft" pos="0 0 .03"><geom type="ellipsoid" size=".01 .02 .03"/>
      </body></body></body></body>
      <body name="robot/bearing_roll" pos=".1 0 -.181"><geom type="box" size=".02 .02 .02"/></body>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    names=[model.body(i).name.removeprefix('robot/') for i in range(model.nbody)]
    robot=types.SimpleNamespace(body_names=names,data=types.SimpleNamespace(
        body_link_pos_w=torch.tensor(data.xpos[None],dtype=torch.float32),
        body_link_quat_w=torch.tensor(data.xquat[None],dtype=torch.float32)))
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.terrain=types.SimpleNamespace(env_origins=torch.zeros(1,3))
    env=types.SimpleNamespace(sim=types.SimpleNamespace(mj_model=model),scene=scene,device='cpu')
    head=GeometryClearance(env);legacy=GeometryClearance(env,body_names=('bearing_roll',))
    assert float(head(env)[0])==pytest.approx(.33)
    assert float(legacy(env)[0])==pytest.approx(-.001,abs=1e-7)
    assert 'bearing_roll' not in {row['body'] for row in head.names}
