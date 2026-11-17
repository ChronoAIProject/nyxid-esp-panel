"""Original training-only first128 batch, gradients without updates."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-multiscale-batch-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def plan():
 files=dict(json.loads((R/'4090-binary40-training-fit-plan-v1.json').read_text())['files'])
 for p in [Path(__file__),Path('research/wakeword_p4/training/causal_binary40_multiscale_v1.py'),Path('research/wakeword_p4/training/causal_binary40_span_v1.py'),R/'4090-causal-binary40-original-batch-v1.json',R/'4090-multiscale-structural-original-v2.json']:files[str(p)]=sha(p)
 with PLAN.open('x') as f:json.dump(dict(files=files,seed=726,batch=128,optimizer_updates=0,scope='Original first128training batch only; random full multiscale head (not embedded zero-shallow probe); descriptor/mask identity, finite loss/all gradients, per-scale head gradient nonzero, weights unchanged. No efficacy/fitting/INT8/P4 claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from causal_source_balanced_batch_v1 import Sampler
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_multiscale_v1 import Binary40Multiscale
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert '4090' in torch.cuda.get_device_name()
 (x,pw,nw),desc=Sampler(726).batch();ref=json.loads((R/'4090-causal-binary40-original-batch-v1.json').read_text());hashes=dict(features=digest(x),positive_weights=digest(pw),negative_weights=digest(nw));assert hashes==ref['batch_hashes'] and desc==ref['descriptors']
 torch.manual_seed(726);m=Binary40Multiscale().cuda();before={k:digest(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
 z,_=m(torch.from_numpy(x).permute(0,2,1).cuda(),m.initial_state(128,'cuda'));logits=z[:,0];loss=(torch.nn.functional.softplus(-logits)*torch.from_numpy(pw).cuda()+20*torch.nn.functional.softplus(logits)*torch.from_numpy(nw).cuda()).sum(1).mean();loss.backward();assert torch.isfinite(loss)
 assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());norms=[float(m.head.weight.grad[:,i*64:(i+1)*64].norm()) for i in range(6)];assert all(v>0 for v in norms)
 assert before=={k:digest(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
 out=dict(plan_sha256=sha(PLAN),all_passed=True,gpu=torch.cuda.get_device_name(),batch_hashes=hashes,descriptors_exact=True,loss=float(loss.detach()),head_scale_gradient_norms=norms,all_gradients_finite=True,weights_unchanged=True,optimizer_updates=0,scope=p['scope'])
 with (R/'4090-multiscale-batch-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
