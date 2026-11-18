"""Exact compatible ONNX actor initialization, no optimizer/counters inherited."""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch



def initialize(policy,path,actual_obs=None):
    graph=onnx.load(path).graph
    arrays={t.name:numpy_helper.to_array(t).copy() for t in graph.initializer}
    parameters=dict(policy.named_parameters())
    loaded=[]
    with torch.no_grad():
        for name,weight in arrays.items():
            if name.startswith('mlp.'):
                assert name in parameters,(name,list(parameters))
                target=parameters[name]
                assert tuple(target.shape)==weight.shape
                target.copy_(torch.as_tensor(weight,device=target.device))
                loaded.append(name)
        assert len(loaded)==8,loaded
        norm=policy.obs_normalizer
        norm._mean.copy_(torch.as_tensor(arrays['obs_normalizer._mean'],device=norm._mean.device))
        div=[n for n in graph.node if n.op_type=='Div' and 'obs_normalizer' in n.name]
        assert len(div)==1 and div[0].input[1] in arrays
        divisor=arrays[div[0].input[1]]
        norm._std.copy_(torch.as_tensor(divisor,device=norm._std.device)-norm.eps)
        assert (norm._std>=0).all()
        norm._var.copy_(norm._std.square())
        norm.count.zero_()
        freeze_normalizer(policy)
    policy.eval()
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(8126)
    samples=rng.normal(0,.5,(256,61)).astype(np.float32)
    if actual_obs is not None:
        samples=np.concatenate([samples,actual_obs.detach().cpu().numpy()])
    device=next(policy.parameters()).device
    with torch.no_grad():
        prediction=policy({'actor':torch.tensor(samples,device=device)}).cpu().numpy()
    expected=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0] for row in samples])
    error=float(np.abs(prediction-expected).max())
    assert error<=1e-3,error
    return {'initialization':'teacher_actor_and_frozen_normalizer','teacher_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'imported_weights':loaded,'parity_samples':len(samples),'teacher_import_max_abs_error':error,'normalizer_frozen':True,'normalizer_count':0,'optimizer_inherited':False,'critic_inherited':False,'counter_inherited':False}

def freeze_normalizer(policy):
    """Call again after every checkpoint load; until is not a state_dict buffer."""
    policy.obs_normalizer.until = 0
    return {key: value.detach().clone() for key, value in policy.obs_normalizer.state_dict().items()}


def assert_normalizer_unchanged(policy, snapshot):
    assert policy.obs_normalizer.until == 0
    for key, value in snapshot.items():
        assert torch.equal(policy.obs_normalizer.state_dict()[key], value), key


def actor_warmup(policy, common_steps, warmup_updates, steps_per_update):
    """Apply after ALL actor losses; setting grad=None prevents Adam momentum updates too."""
    frozen = common_steps <= warmup_updates*steps_per_update
    if frozen:
        for parameter in policy.parameters():
            parameter.grad = None
    return frozen
