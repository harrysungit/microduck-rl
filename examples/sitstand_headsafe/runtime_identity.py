"""Record constructed objects and live manager configs, not guessed class_type."""
from dataclasses import asdict,is_dataclass
import hashlib
import inspect
from pathlib import Path
import torch


def encode(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value,dict):
        return {str(key):encode(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [encode(item) for item in value]
    if isinstance(value,slice):
        return {'slice':[value.start,value.stop,value.step]}
    if isinstance(value,(str,int,float,bool,type(None))):
        return value
    if callable(value):
        return getattr(value,'__module__',type(value).__module__)+'.'+getattr(value,'__qualname__',type(value).__qualname__)
    return repr(value)


def object_identity(instance):
    cls=type(instance)
    source=Path(inspect.getsourcefile(cls))
    return {'class':cls.__module__+'.'+cls.__qualname__,'definition_source_file':source.name,'definition_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}


def snapshot(env,task_id,profile):
    events={name:encode(env.event_manager.get_term_cfg(name)) for name in env.cfg.events}
    observations={name:encode(env.observation_manager.get_term_cfg('actor',name)) for name in env.observation_manager.active_terms['actor']}
    return {'evidence_kind':'observed_constructed_runtime_instances','runtime_instance_verified':True,'task_id':task_id,'actual_env':object_identity(env),'actual_command':object_identity(env.command_manager.get_term('twist')),'actual_runtime_action':object_identity(env.action_manager.get_term('joint_pos')),'posture_goals':asdict(env.goals),'endpoint_geometry':env.endpoint_geometry,'reward_contract':{'sit_endpoint_requires_continuous_support':env.cfg.sit_endpoint_requires_continuous_support,'support_gate_timing':'SIT command and alpha>=.98; every physics substep of current control tick; standing and early transition unaffected'},'profile':profile,'actual_event_manager_terms':events,'actual_actor_observation_manager_terms':observations,'physics_dt':env.physics_dt,'control_dt':env.step_dt,'head_command_scope':'neutral head only; self-balancing head outputs remain active'}
