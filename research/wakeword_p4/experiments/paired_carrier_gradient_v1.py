"""Actual batch128 CUDA gradient gate, zero optimizer updates."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-paired-carrier-gradient-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 base=R/'4090-paired-carrier-preflight-plan-v1.json';files=dict(json.loads(base.read_text())['files'])
 for p in [base,Path(__file__),R/'4090-paired-carrier-preflight-original-v1.json',Path('research/wakeword_p4/training/causal_binary40_span_v1.py')]:files[str(p)]=sha(p)
 with LOCK.open('x') as f:json.dump({'files':files,'seed':726,'batch_size':128,'optimizer_updates':0},f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_span_v1 import Binary40Span,CONTROL
 from causal_paired_carrier_batch_v1 import Sampler
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name();model=Binary40Span(CONTROL).cuda();before={k:v.detach().clone() for k,v in model.state_dict().items()};(x,pw,nw),d=Sampler(726).batch();assert x.shape==(128,846,40) and pw.shape==nw.shape==(128,282)
 logits,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(128,'cuda'));z=logits[:,0];z.retain_grad();pos=torch.from_numpy(pw).cuda();neg=torch.from_numpy(nw).cuda();loss=(torch.nn.functional.softplus(-z)*pos+20*torch.nn.functional.softplus(z)*neg).sum(1).mean();loss.backward();assert torch.isfinite(loss)
 norms={}
 for k,v in model.named_parameters():assert v.grad is not None and torch.isfinite(v.grad).all();norms[k]=float(v.grad.norm())
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());paired=[]
 for i,row in enumerate(d):
  if row['kind'].startswith('paired_carrier_'):
   mask=pos[i]>0 if row['label'] else neg[i]>0;g=z.grad[i][mask];assert len(g)>0 and torch.isfinite(g).all() and torch.count_nonzero(g)==len(g);assert bool((g<0).all()) if row['label'] else bool((g>0).all());paired.append({'slot':i,'descriptor':row,'supervised_calls':len(g),'logit_gradient_norm':float(g.norm())})
 assert len(paired)==8
 out={'schema':'paired-carrier-gradient-original-v1','all_passed':True,'plan_sha256':sha(LOCK),'gpu':torch.cuda.get_device_name(),'loss':float(loss),'parameter_count':sum(v.numel() for v in model.parameters()),'parameter_gradient_norms':norms,'paired_rows':paired,'features_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'weights_unchanged':True,'optimizer_updates':0,'limits':'One fresh original Binary40 batch gradient gate only, no learned recognition/export/P4 evidence.'};(R/'4090-paired-carrier-gradient-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
