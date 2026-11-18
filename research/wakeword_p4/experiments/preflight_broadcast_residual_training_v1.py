"""Original full batch identity and unchanged temporal initialization, zero updates."""
import sys,json,gzip,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_broadcast_residual_v1 import CausalBroadcastResidual
from causal_source_balanced_batch_v1 import Sampler,digest
import subprocess
lock=Path('research/wakeword_p4/results/4090-broadcast-residual-feasibility-untrained-export-plan-v1.json')
assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(lock)])==lock.read_bytes()
for path,h in json.loads(lock.read_text())['files'].items():
 with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,path
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
m=CausalBroadcastResidual().cuda();initial={k:v.detach().clone() for k,v in m.state_dict().items()}

s=Sampler(726);rows=[]
with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:
 for step in [1,2]:
  ref=json.loads(next(f));(x,pw,nw),ds=s.batch();assert ds==ref['descriptors']
  for k,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(a)==ref[k]
  m.zero_grad(set_to_none=True);z,_=m(torch.from_numpy(x).permute(0,2,1).cuda(),m.initial_state(128,'cuda'));z=z[:,0]
  loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(pw).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(nw).cuda()).sum(1).mean();assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());rows.append(dict(step=step,loss=float(loss.detach()),features_sha256=digest(x),positive_sha256=digest(pw),negative_sha256=digest(nw)))
assert all(torch.equal(v,initial[k]) for k,v in m.state_dict().items())
out=dict(all_passed=True,optimizer_updates=0,original_full_rows_verified=256,weights_unchanged=True,gpu=torch.cuda.get_device_name(),rows=rows,geometry=m.geometry(),limits='Two original batches CUDA backward only, not recognition/optimizer arithmetic or independent pronunciation truth.')
with open('research/wakeword_p4/results/4090-broadcast-residual-training-preflight-v1.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
