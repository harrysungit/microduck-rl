"""Build a review-only model package from THIS run's params and unchanged ONNX.

Never imports a policy, edits firmware/settings, or selects a robot model. Python
YAML tags are decoded as inert data, never imported or constructed.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
import numpy as np
import onnx
import onnxruntime as ort
import yaml
if __package__:
    from .archive_sources import actual_task,training_adapter
else:
    from archive_sources import actual_task,training_adapter

JOINTS=('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle')
TERMS=('base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command')
BAM_SHA='ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb'
REJECTED_SHA='6f6c06e830d32023556f66d38c495e9b54ff6625448246896221875269e1dc20'

class DataLoader(yaml.SafeLoader):pass

def python_data(loader,suffix,node):
    if suffix.startswith('name:'):return {'callable':suffix[5:]}
    if isinstance(node,yaml.SequenceNode):
        value=loader.construct_sequence(node,deep=True)
        return value if suffix=='tuple' else {'type':suffix,'args':value}
    if isinstance(node,yaml.MappingNode):return loader.construct_mapping(node,deep=True)
    return {'type':suffix,'value':loader.construct_scalar(node)}
DataLoader.add_multi_constructor('tag:yaml.org,2002:python/',python_data)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def reject_nested_failure(value,path='evaluation'):
    """No passing subset may hide a failed stress/held-out/switch row elsewhere."""
    if isinstance(value,dict):
        for key,item in value.items():
            here=path+'.'+str(key)
            if key=='automatic_pass' and item is not True:
                raise ValueError('Non-passing automatic check at '+here)
            if key.endswith('_pass') and item is False:
                raise ValueError('Failed check at '+here)
            if key in ('head_contact','head_contact_latched'):
                if not (item is False or (type(item) in (int,float) and math.isfinite(item) and item==0)):
                    raise ValueError('Head contact or unknown contact evidence at '+here)
            if key=='head_contact_substeps' and item!=0:
                raise ValueError('Head contact substeps at '+here)
            if key in ('error','errors') and item:
                raise ValueError('Evaluation error at '+here)
            reject_nested_failure(item,here)
    elif isinstance(value,list):
        for index,item in enumerate(value):reject_nested_failure(item,f'{path}[{index}]')
    elif isinstance(value,str) and value.strip().upper() in ('FAIL','FAILED','UNVERIFIED'):
        raise ValueError('Failed/unverified result at '+path)

def check_identity(row):
    if isinstance(row,str) and row.strip():return row
    if not isinstance(row,dict):raise ValueError('Declared checks and rows need explicit identities')
    if isinstance(row.get('check_id'),str) and row['check_id'].strip():return row['check_id']
    if isinstance(row.get('condition'),str) and row['condition'].strip() and type(row.get('seed')) is int:
        return f"{row['condition']}:{row['seed']}"
    raise ValueError('Each check needs check_id or condition plus integer seed')

def require_complete_row(row):
    identity=check_identity(row)
    if row.get('automatic_pass') is not True or row.get('legal_sit_support_pass') is not True:
        raise ValueError('Missing automatic/loaded-sit success at '+identity)
    audit=row.get('safety')
    if not isinstance(audit,dict):raise ValueError('Missing physics audit at '+identity)
    if audit.get('head_scope_version')!='neck-subtree-v2' or audit.get('head_scope_source')!='parent-link neck subtree':
        raise ValueError('Anatomical neck-subtree audit required at '+identity)
    bodies=audit.get('head_bodies')
    if not isinstance(bodies,list) or not all(isinstance(name,str) for name in bodies) or not {'neck','neck_pitch','yaw_roll_motion','jaw_soft'}<=set(bodies) or 'bearing_roll' in bodies:
        raise ValueError('True head body coverage is missing at '+identity)
    if type(audit.get('head_geometry_count')) is not int or audit['head_geometry_count']<=0:
        raise ValueError('Missing head geometry coverage at '+identity)
    if audit.get('head_contact') is not False or audit.get('head_contact_substeps')!=0:
        raise ValueError('True head contact audit missing or failed at '+identity)
    if 'first_contact_s' not in audit or audit['first_contact_s'] is not None:
        raise ValueError('First-contact evidence missing or inconsistent at '+identity)
    steps,expected=audit.get('physics_substeps'),audit.get('expected_physics_substeps')
    if type(steps) is not int or type(expected) is not int or expected<=0 or steps!=expected:
        raise ValueError('Incomplete physics substep coverage at '+identity)
    coverage=audit.get('coverage')
    if type(coverage) not in (int,float) or not math.isfinite(coverage) or coverage!=1.:
        raise ValueError('Missing complete audit coverage at '+identity)
    gap=audit.get('min_clearance_m')
    if type(gap) not in (int,float) or not math.isfinite(gap) or gap<0:
        raise ValueError('Missing/invalid true head clearance at '+identity)

def require_evidence(summary,policy_sha,checkpoint_sha,diagnostic=False):
    if not isinstance(summary,dict):raise ValueError('Evaluation summary must be an object')
    if summary.get('policy_sha256')!=policy_sha:raise ValueError('Evaluation policy SHA256 does not match ONNX')
    if policy_sha==REJECTED_SHA:raise ValueError('Known head-impact regression is forbidden')
    if summary.get('checkpoint_sha256',checkpoint_sha)!=checkpoint_sha:raise ValueError('Evaluation checkpoint hash mismatch')
    if diagnostic:
        if summary.get('diagnostic_only') is not True or summary.get('operating_pass') is not False:
            raise ValueError('Format-only evidence must declare diagnostic_only=true and operating_pass=false')
        return 'diagnostic_only_not_a_candidate'
    if summary.get('diagnostic_only') is True or summary.get('operating_pass') is not True:
        raise ValueError('Final evaluation must explicitly declare operating_pass=true, not diagnostic_only')
    if summary.get('automatic_pass') is not True:raise ValueError('Final evaluation must explicitly declare automatic_pass=true')
    reject_nested_failure(summary)
    declared,rows=summary.get('declared_checks'),summary.get('rows')
    if not isinstance(declared,list) or not declared or not isinstance(rows,list) or not rows:
        raise ValueError('Final evaluation requires nonempty declared_checks and rows')
    declared_ids=[check_identity(item) for item in declared]
    row_ids=[check_identity(row) for row in rows]
    if len(set(declared_ids))!=len(declared_ids) or len(set(row_ids))!=len(row_ids) or set(declared_ids)!=set(row_ids):
        raise ValueError('Declared check scope and rows must match exactly, with no missing, omitted or duplicate checks')
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Every row must include its full physics audit')
        require_complete_row(row)
    return 'automatic_checks_complete_pending_human_video_review'

def no_calibration_values(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('calibration','zero_raw','min_rad','max_rad','direction','device_calibration','joints_calibration'):
                raise ValueError('Device calibration values must not enter the contract: '+key)
            no_calibration_values(item)
    elif isinstance(value,list):
        for item in value:no_calibration_values(item)

def calibration_locator(action,initialization,summary,source,provided=None,diagnostic=False):
    """Bind an external legacy-global locator without pretending it was serialized."""
    original=action.get('calibration')
    present='calibration' in action
    if provided is not None and not diagnostic:
        raise ValueError('Explicit external calibration locators are diagnostic-only')
    if present:
        if not isinstance(original,str) or not original:raise ValueError('Invalid serialized calibration locator')
        selected=Path(provided) if provided is not None else Path(original)
        mode='explicit_external_for_serialized_cfg' if provided is not None else 'serialized_action_cfg'
    else:
        if not diagnostic or provided is None:
            raise ValueError('Legacy-global calibration requires diagnostic-only and explicit --calibration')
        declared=any(isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CALIBRATION' for t in node.targets)
                     for node in ast.parse(Path(source).read_text()).body)
        if not declared:raise ValueError('Archived source does not establish a module-global CALIBRATION locator')
        selected=Path(provided);mode='explicit_external_legacy_global'
    digest=sha(selected)
    if initialization.get('calibration_sha256')!=digest:
        raise ValueError('External calibration differs from initialization SHA256')
    if not present and summary.get('calibration_sha256') is None:
        raise ValueError('Legacy-global calibration requires explicit evaluation calibration_sha256')
    if summary.get('calibration_sha256',digest)!=digest:
        raise ValueError('External calibration differs from evaluation SHA256')
    record={'mode':mode,'configuration_field_present':present,'calibration_sha256':digest,
            'configuration_note':'Original action cfg serialized a locator' if present else
                'Original RuntimeActionCfg had no calibration field; archived module-global CALIBRATION was external',
            'archive_global_symbol':None if present else 'runtime_actions.CALIBRATION',
            'external_locator_recorded_without_path_or_values':True}
    clean=deepcopy(action);clean.pop('calibration',None)
    return clean,digest,record

def verify_pipeline_source(directory):
    path=directory/'runtime_state.py'
    if not path.exists():path=directory/'alignment_state.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('_ids','TargetState','VoltageState')]
    state_digest=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    tree=ast.parse((directory/'runtime_actions.py').read_text())
    action=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RuntimeAction')
    process=next(n for n in action.body if isinstance(n,ast.FunctionDef) and n.name=='process_actions')
    action_digest=hashlib.sha256(ast.dump(process,include_attributes=False).encode()).hexdigest()
    if state_digest!='a03117125f5106e63db1184c4a3f9d8f75eee638a2e5f0c04a0ed709bc9bd52a' or action_digest!='035e7613cb6f5eae36beed42dbc7bbb80ea0f35011dd7caefebde09b4ea17468':
        raise ValueError('Runtime target algorithm differs from the reviewed recipe; review a new mapping first')
    return {'state_ast_sha256':state_digest,'action_process_ast_sha256':action_digest}

def build(args):
    digest=sha(args.onnx); checkpoint_digest=sha(args.checkpoint)
    summary=json.loads(args.evaluation_summary.read_text(encoding='utf-8'))
    state=require_evidence(summary,digest,checkpoint_digest,args.diagnostic_only)
    if not args.task.startswith('Mjlab-SitStand-'):raise ValueError('This recipe packages only its real SitStand task')
    env_path=args.params/'env.yaml';agent_path=args.params/'agent.yaml'
    env=yaml.load(env_path.read_text(),Loader=DataLoader);agent=yaml.load(agent_path.read_text(),Loader=DataLoader)
    model=onnx.load(args.onnx);onnx.checker.check_model(model)
    if len(model.graph.input)!=1 or len(model.graph.output)!=1:raise ValueError('Requires one-input one-output feed-forward ONNX')
    if model.graph.input[0].name!='obs':raise ValueError('Firmware requires input name obs')
    for tensor,width in ((model.graph.input[0],61),(model.graph.output[0],14)):
        dims=tensor.type.tensor_type.shape.dim
        if len(dims)!=2 or dims[1].dim_value!=width or (dims[0].dim_value not in (0,1)):
            raise ValueError('Requires obs[1 or dynamic,61] -> actions[1 or dynamic,14]')
    if not any(n.op_type=='Sub' and any('obs_normalizer' in i for i in n.input) for n in model.graph.node) or not any(n.op_type=='Div' for n in model.graph.node):
        raise ValueError('Baked observation normalization was not established')
    metadata={p.key:p.value for p in model.metadata_props}
    if len(metadata)!=len(model.metadata_props):raise ValueError('Duplicate ONNX metadata keys')
    if metadata.get('joint_names')!=','.join(JOINTS):raise ValueError('Motor order differs from firmware')
    if metadata.get('observation_names')!=','.join(TERMS):raise ValueError('Observation order differs from firmware')
    if metadata.get('command_names')!='twist,head_pose':raise ValueError('Unexpected command slots')
    scales=[float(x) for x in metadata['action_scale'].split(',')]
    if not all(x==1. for x in scales) or len(scales) not in (1,14):raise ValueError('Requires scale-1 metadata')
    if agent['actor']['obs_normalization'] is not True or agent['clip_actions'] is not None:raise ValueError('Unexpected actor normalization/action clipping')
    session=ort.InferenceSession(str(args.onnx),providers=['CPUExecutionProvider'])
    prediction=session.run(None,{'obs':np.zeros((1,61),dtype=np.float32)})[0]
    if prediction.shape!=(1,14) or not np.isfinite(prediction).all():raise ValueError('Non-finite/wrong ONNX action')
    action=deepcopy(env['actions']['joint_pos'])
    if action.get('use_default_offset') is not True or action.get('offset')!=0. or action.get('clip') is not None or action.get('scale')!=1.:
        raise ValueError('Unexpected trained joint position mapping')
    initialization=json.loads((args.params.parent/'initialization.json').read_text())
    action,calibration_digest,locator=calibration_locator(action,initialization,summary,
        args.params.parent/'source/runtime_actions.py',getattr(args,'calibration',None),args.diagnostic_only)
    if 'RuntimeAction' not in (args.params.parent/'source/runtime_actions.py').read_text():raise ValueError('Archived RuntimeAction source required')
    action['implementation']={'callable':'runtime_actions.RuntimeAction','source_sha256':sha(args.params.parent/'source/runtime_actions.py')}
    robot=env['scene']['entities']['robot']
    joints=[]
    for name in JOINTS:
        values=[v for pattern,v in robot['init_state']['joint_pos'].items() if re.fullmatch(pattern,name)]
        if len(values)!=1 or not np.isfinite(values[0]):raise ValueError('Missing/ambiguous finite Home: '+name)
        joints.append({'name':name,'home_rad':values[0]})
    exported_home=[float(x) for x in metadata['default_joint_pos'].split(',')]
    if len(exported_home)!=14 or max(abs(row['home_rad']-value) for row,value in zip(joints,exported_home))>.00051:
        raise ValueError('ONNX Home metadata differs from actual config')
    actuators=deepcopy(robot['articulation']['actuators'])
    if len(actuators)!=1:raise ValueError('Expected one BAM actuator')
    parameter_path=Path(actuators[0]['json_path'])
    if sha(parameter_path)!=BAM_SHA or actuators[0]['kp_fw']!=5.:raise ValueError('Unexpected BAM/P configuration')
    actuators[0]['json_path']=parameter_path.name
    actuators[0]['parameter_file']={'name':parameter_path.name,'sha256':BAM_SHA}
    hz=1/(env['decimation']*env['sim']['mujoco']['timestep'])
    if abs(hz-50)>1e-9:raise ValueError('Requires 50Hz')
    if tuple(env['observations']['actor']['terms'])!=TERMS:raise ValueError('Actual observation order differs')
    source_dir=args.params.parent/'source'
    source_hashes={p.name:sha(p) for p in sorted(source_dir.glob('*.py'))}
    recorded_sources={Path(name).name:value for name,value in initialization['source_sha256'].items()}
    if source_hashes!=recorded_sources:raise ValueError('Archived source differs from training initialization hashes')
    adapter=training_adapter(source_dir,args.task)
    pipeline_identity=verify_pipeline_source(source_dir)
    commands=deepcopy(env['commands'])
    commands['twist']['serialized_cfg_class_type']=commands['twist'].get('class_type')
    commands['twist']['class_type']={'callable':adapter['command_callable']}
    commands['twist']['class_type_evidence_kind']='static_archived_source_ast_only'
    commands['twist']['class_type_runtime_instance_verified']=False
    provenance={'training_task':args.task,'checkpoint_sha256':checkpoint_digest,'env_yaml_sha256':sha(env_path),
                'agent_yaml_sha256':sha(agent_path),'source_sha256':source_hashes,'evaluation_summary_sha256':sha(args.evaluation_summary),
                'policy_sha256':digest,'calibration_sha256':calibration_digest,'calibration_locator':locator,
                'archived_task_adapter':adapter,'review_only':True,'human_review':'pending',
                'deployment_approved':False,'review_state':state,'onnx_weights_unchanged':True,
                'command_builder_note':'Static archived-source builder path only, not an observed historical runtime instance; original serialized class_type retained separately',
                'training_cycle':'See actual archived command builder and serialized command configuration; no timing inferred from another task',
                'packager_source_sha256':sha(Path(__file__)),'source_resolver_sha256':sha(Path(__file__).with_name('archive_sources.py'))}
    pipeline={'implementation':'runtime_actions.RuntimeAction','source_sha256':source_hashes['runtime_actions.py'],
              'stage':'external_not_baked_into_onnx','previous_action_observation':'raw_policy_output',
              'reviewed_algorithm_identity':pipeline_identity,
              'action_scale':1.,'voltage_adapt':True,'nominal_voltage':7.4,'voltage_ema_new_weight':.1,
              'voltage_update_policy_steps':50,'voltage_clamp_V':[6.,9.5],
              'head_lowpass':.5,'legs_lowpass':.7,'target_reset':'first_target_passthrough',
              'runtime_reset_scope':'controller episode reset; slot switches can preserve filter history, requiring separate switch tests',
              'limit_source':'external_device_calibration_hash','quantization':'4096 ticks per 2pi; ties away from zero',
              'required_runtime_settings_note':'hd1910-v1 parser does not enforce these external runtime settings; verify device profile separately'}
    contract={'schema':'microduck-deployment-contract/v1','task_adapter':'hd1910-v1','task':args.task,'slot':'sitstand',
              'robot_mode':'feet','target_firmware':{'version':'0.15.1'},'control_hz':50.,'input_dim':61,'output_dim':14,
              'obs_normalizer':'baked_in_onnx','onnx_metadata':metadata,'model':{'joints':joints,'actuators':actuators},
              'actions':{'joint_pos':action},'action_ema_old_weight':0.,'inference_action_scale':{'value':1.,'baked_into_onnx':False},
              'observations':env['observations']['actor'],'commands':commands,'runtime_target_pipeline':pipeline,'provenance':provenance,
              'review_only':True,'human_review':'pending','deployment_approved':False}
    no_calibration_values(contract)
    args.out.mkdir(parents=True,exist_ok=False)
    # Keep every supplied failure/detail outside the four-file import ZIP. The
    # contract binds this unchanged evidence rather than selecting passing rows.
    shutil.copyfile(args.evaluation_summary,args.out/'evaluation-summary.json')
    package=args.out/'package';package.mkdir()
    shutil.copyfile(args.onnx,package/'policy.onnx')
    save(package/'deployment-contract.json',contract)
    save(package/'manifest.json',{'task':args.task,'onnx_sha256':digest,'provenance':provenance,'review_only':True,'deployment_approved':False})
    label=('DIAGNOSTIC ONLY - ' if args.diagnostic_only else '')+args.name
    save(package/'model-info.json',{'name':label,'policy_sha256':digest,'review_state':state,'human_review':'pending','review_only':True,'deployment_approved':False})
    validation={'performed':False,'scope':'native contract + ONNX shape and one zero-observation inference, not action safety'}
    if args.validator:
        process=subprocess.run([str(args.validator.resolve()),'validate-policy',str((package/'policy.onnx').resolve())],capture_output=True,text=True,timeout=120)
        validation.update(performed=True,validator_sha256=sha(args.validator),returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
        save(args.out/'offline-validation.json',validation)
        if process.returncode:raise ValueError('Native offline validator rejected package; see offline-validation.json')
    archive=args.out/('diagnostic-format-only.zip' if args.diagnostic_only else 'review-only.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.iterdir()):bundle.write(path,path.name)
    result={'package_sha256':sha(archive),'policy_sha256':digest,'checkpoint_sha256':checkpoint_digest,
            'evaluation_summary_file':'evaluation-summary.json','evaluation_summary_sha256':sha(args.evaluation_summary),
            'calibration_locator':locator,'archived_task_adapter':adapter,
            'review_state':state,'review_only':True,'human_review':'pending','deployment_approved':False,'offline_validation':validation}
    save(args.out/'package-review.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('onnx','checkpoint','params','evaluation-summary','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--task',required=True,help='actual new training task; no base-task aliases')
    parser.add_argument('--name',required=True)
    parser.add_argument('--validator',type=Path,help='optional x86 robotd; ORT_DYLIB_PATH supplied by caller')
    parser.add_argument('--calibration',type=Path,help='diagnostic-only external locator for legacy-global or moved calibration; hashes must match init/evidence')
    parser.add_argument('--diagnostic-only',action='store_true',help='format probe only; evidence must explicitly deny operating_pass')
    args=parser.parse_args()
    print(json.dumps(build(args),indent=2),flush=True)

if __name__=='__main__':main()
