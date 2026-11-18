"""Input identity binding and audited teacher-label checks. No simulator dependency."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_inputs(teacher, anchor, calibration, sources, settings):
    record={'schema':1,'teacher_sha256':sha(teacher),'anchor_data_sha256':sha(anchor),
            'calibration_sha256':sha(calibration),'source_sha256':{p.name:sha(p) for p in sources},
            'settings':settings,'normalizer_frozen':True}
    record['identity_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record


def require_resume_identity(infos, expected):
    if not isinstance(infos,dict) or infos.get('recipe',{}).get('identity_sha256')!=expected['identity_sha256']:
        raise ValueError('Checkpoint recipe identity differs from teacher/anchor/calibration/source/settings')
    if infos.get('recipe',{}).get('normalizer_frozen') is not True:
        raise ValueError('Checkpoint does not declare frozen teacher normalization')
    return infos


def validate_anchor(teacher, anchor):
    import onnxruntime as ort
    with np.load(anchor,allow_pickle=False) as data:
        obs=np.asarray(data['observations'],dtype=np.float32)
        actions=np.asarray(data['actions'],dtype=np.float32)
    if obs.ndim!=2 or obs.shape[1]!=61 or actions.shape!=(len(obs),14) or len(obs)==0:
        raise ValueError('Anchor requires nonempty (N,61) observations and (N,14) raw actor labels')
    if not np.isfinite(obs).all() or not np.isfinite(actions).all():raise ValueError('Non-finite anchor')
    session=ort.InferenceSession(str(teacher),providers=['CPUExecutionProvider'])
    name=session.get_inputs()[0].name
    error=0.
    for row,label in zip(obs,actions):
        predicted=session.run(None,{name:row[None]})[0]
        error=max(error,float(np.abs(predicted[0]-label).max()))
    if error>1e-3:raise ValueError(f'Anchor labels do not match this teacher: {error}')
    return obs,actions,{'samples':len(obs),'all_labels_teacher_max_abs_error':error,
                        'collection_safety':'not_proven_by_label_matching; requires external contact/video evidence'}
