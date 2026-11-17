"""Two fixed training batches; actual136 CUDA backward, zero optimizer updates."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from causal_source_balanced_batch_v1 import Sampler as Original
from causal_additive_carrier_batch_v1 import Sampler
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_span_v1 import Binary40Span,CONTROL
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
model=Binary40Span(CONTROL).cuda()
parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
assert all(torch.equal(v.cpu(),parent[k]) for k,v in model.state_dict().items())
a,b=Sampler(726),Original(726)
for step in range(2):
 (x,p,n),d=a.batch();base,bd=b.batch()
 assert d[:128]==bd and len(d)==136
 assert all(np.array_equal(v[:128],w) for v,w in zip((x,p,n),base,strict=True))
 assert np.allclose(p.sum(1),[r['label'] for r in d]) and np.allclose(n.sum(1),1)
 model.zero_grad(set_to_none=True)
 z,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(136,'cuda'));z=z[:,0,:]
 loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(p).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(n).cuda()).sum()/128
 assert torch.isfinite(loss);loss.backward()
 assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
 assert all(torch.equal(v.cpu(),parent[k]) for k,v in model.state_dict().items())
out=dict(all_passed=True,batches=2,original_rows_verified=256,actual_backward_batch_size=136,optimizer_updates=0,weights_unchanged=True,gpu=torch.cuda.get_device_name())
with Path('research/wakeword_p4/results/4090-additive-carrier-preflight-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
