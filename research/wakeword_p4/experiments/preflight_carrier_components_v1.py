"""Independent component reconstruction and actual CUDA backward, zero updates."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from causal_source_balanced_batch_v1 import Sampler as Original
from causal_paired_carrier_batch_v1 import Sampler as Replacement
from causal_carrier_components_batch_v1 import Sampler
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_span_v1 import Binary40Span,CONTROL
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
model=Binary40Span(CONTROL).cuda()
parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
assert all(torch.equal(v.cpu(),parent[k]) for k,v in model.state_dict().items())
for arm,kind in [('positive_only','paired_carrier_positive'),('negative_only','paired_carrier_nick_negative')]:
 sampler,original,replacement=Sampler(726,arm),Original(726),Replacement(726)
 for step in range(2):
  (x,p,n),d=sampler.batch();base,bd=original.batch();other,od=replacement.batch()
  indices=[i for i,t in enumerate(od) if t['kind']==kind];assert len(indices)==4
  expected=tuple(np.concatenate([a,b[indices]],axis=0) for a,b in zip(base,other,strict=True))
  assert d==bd+[od[i] for i in indices] and len(d)==132
  assert all(np.array_equal(v,w) for v,w in zip((x,p,n),expected,strict=True))
  assert np.allclose(p.sum(1),[t['label'] for t in d]) and np.allclose(n.sum(1),1)
  model.zero_grad(set_to_none=True)
  z,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(132,'cuda'));z=z[:,0,:]
  loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(p).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(n).cuda()).sum()/128
  assert torch.isfinite(loss);loss.backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
  assert all(torch.equal(v.cpu(),parent[k]) for k,v in model.state_dict().items())
out=dict(all_passed=True,batches_per_arm=2,original_rows_verified=512,independent_reconstruction=True,actual_backward_batch_size=132,optimizer_updates=0,weights_unchanged=True,gpu=torch.cuda.get_device_name())
with Path('research/wakeword_p4/results/4090-carrier-components-preflight-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
