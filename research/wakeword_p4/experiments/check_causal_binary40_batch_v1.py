"""Actual original C161 batch compatibility; no optimizer or development audio."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from causal_source_balanced_batch_v1 import Sampler, digest
sys.path.insert(0, str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_v1 import Binary40


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(726)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    (x, p, n), descriptors = Sampler(726).batch()
    assert x.shape == (128, 846, 40) and p.shape == n.shape == (128, 282)
    assert np.all((p == 0) | (n == 0))
    model = Binary40().cuda()
    before = {name: digest(v.detach().cpu().numpy()) for name,v in model.state_dict().items()}
    features = torch.from_numpy(x).permute(0,2,1).cuda()
    pw, nw = torch.from_numpy(p).cuda(), torch.from_numpy(n).cuda()
    z, _ = model(features, model.initial_state(128, 'cuda'))
    z = z[:,0,:]
    pos = torch.nn.functional.softplus(-z)
    neg = torch.nn.functional.softplus(z)
    loss = (pos*pw+20*neg*nw).sum(dim=1).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    assert before == {name:digest(v.detach().cpu().numpy()) for name,v in model.state_dict().items()}
    files = [Path(__file__), Path('research/wakeword_p4/training/causal_binary40_v1.py'),
             Path('research/wakeword_p4/experiments/causal_source_balanced_batch_v1.py'),
             Path('research/wakeword_p4/experiments/causal_sequence_supervision_v1.py')]
    out = dict(schema='causal-binary40-original-batch-v1', gpu=torch.cuda.get_device_name(),
               files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
               batch_hashes=dict(features=digest(x),positive_weights=digest(p),negative_weights=digest(n)),
               descriptors=descriptors,shape=list(x.shape),output_shape=list(z.shape),
               loss=float(loss.detach()),all_parameter_gradients_finite=True,
               no_optimizer_updates=True,model_parameters_unchanged=True,
               maximum_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
               policy='One actual original sampler batch. Original clipped features and positive/20xnegative presentation-normalized BCE; no mask/source edits. Untrained compatibility only, no recognition or convergence evidence; no new synthesis or cache.')
    path=Path('research/wakeword_p4/results/4090-causal-binary40-original-batch-v1.json')
    with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('descriptors','files')}))


if __name__=='__main__':main()
