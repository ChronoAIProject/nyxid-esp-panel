"""Disposable optimizer wiring and original-checkpoint norms; no saved candidate."""
import gzip,json,subprocess,sys
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');P=R/'4090-binary40-regularization-preflight-plan-v1.json'
CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def plan():
 files=dict(json.loads((R/'4090-frozen-history-teacher-training-plan-v1.json').read_text())['files'])
 for f in [Path(__file__),Path(__file__).with_name('binary40_regularization_hypothesis_v1.md'),R/'4090-frozen-history-teacher-training-review-v1.json',*[Path(f'.local/wakeword/frozen-history-teacher-training-v1/{a}/final.pt') for a in ['control','teacher']]]:files[str(f)]=sha(f)
 with P.open('x') as f:json.dump(dict(files=files,steps=2,rtol=1e-5,atol=1e-6,policy='Disposable first2originalmain128/pair4batches; authenticate originals, Adam versusAdamW0 and independent float64 AdamW formula for decay.01 matrix-only. Frozen independent teacher, loss main+pair+KL1. No checkpoint reuse, formal fitting, scored-source input, threshold/calibration choice or recognition claim.'),f,indent=2)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from causal_source_balanced_batch_v1 import Sampler,digest
 from frozen_history_teacher_pair_v1 import Pairs,tensor,bce,pair_components
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load(CP,map_location='cpu',weights_only=False)
 def model():
  m=Binary40().cuda().eval();m.load_state_dict(cp['model'],strict=True);return m
 teacher=model();teacher.requires_grad_(False);models={a:model() for a in ['adam','adamw0','adamw01']};opts={};group_names={}
 for arm,m in models.items():
  decay=[(n,v) for n,v in m.named_parameters() if v.ndim>=2];other=[(n,v) for n,v in m.named_parameters() if v.ndim<2];assert set(n for n,v in decay).isdisjoint(n for n,v in other);assert len(decay)+len(other)==len(list(m.parameters()))
  group_names[arm]=dict(decay=[n for n,v in decay],no_decay=[n for n,v in other])
  if arm=='adam':opts[arm]=torch.optim.Adam(m.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
  else:opts[arm]=torch.optim.AdamW([dict(params=[v for n,v in decay],weight_decay=.01 if arm=='adamw01' else 0),dict(params=[v for n,v in other],weight_decay=0)],lr=.001,betas=(.9,.999),eps=1e-8,foreach=False)
 norms={}
 for arm,path in [('original',CP),*[(a,Path(f'.local/wakeword/frozen-history-teacher-training-v1/{a}/final.pt')) for a in ['control','teacher']]]:
  state=torch.load(path,map_location='cpu',weights_only=False)['model'];norms[arm]={n:dict(l2=float(v.double().norm()),drift_l2=float((v.double()-cp['model'][n].double()).norm())) for n,v in state.items()}
 sampler=Sampler(726);pairs=Pairs();records=[]
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:
  for step in [1,2]:
   (x,pw,nw),desc=sampler.batch();old=json.loads(next(f));assert old['descriptors']==desc
   hashes={k:digest(v) for k,v in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]};assert all(old[k]==h for k,h in hashes.items());ids=pairs.schedule[step-1]['inventory_indices'];batch=pairs.batch(ids);row=dict(step=step,main_hashes=hashes,pair_indices=ids,pair_hashes=[digest(v) for v in batch],arms={})
   for arm,m in models.items():
    opt=opts[arm];opt.zero_grad(set_to_none=True);z,_=m(tensor(x).permute(0,2,1),m.initial_state(128,'cuda'));main=bce(z[:,0],tensor(pw),tensor(nw));paired,kl=pair_components(m,teacher,batch);loss=main+paired+kl;loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters());assert all(v.grad is None for v in teacher.parameters())
    expected={}
    if arm=='adamw01':
     for n,v in m.named_parameters():
      g=v.grad.detach().double();state=opt.state[v];first=state['exp_avg'].double() if state else torch.zeros_like(g);second=state['exp_avg_sq'].double() if state else torch.zeros_like(g);first=.9*first+.1*g;second=.999*second+.001*g*g;rate=.01 if v.ndim>=2 else 0;expected[n]=v.detach().double()*(1-.001*rate)-.001*(first/(1-.9**step))/(torch.sqrt(second/(1-.999**step))+1e-8)
    opt.step();assert all(torch.isfinite(v).all() for v in m.parameters());error=0.
    if expected:
     for n,v in m.named_parameters():
      error=max(error,float((v.detach().double()-expected[n]).abs().max()));assert torch.allclose(v.detach().double(),expected[n],rtol=p['rtol'],atol=p['atol']),n
    row['arms'][arm]=dict(main=float(main.detach()),paired=float(paired.detach()),kl=float(kl.detach()),formula_max_abs_error=error)
   error=max(float((a-b).abs().max()) for a,b in zip(models['adam'].parameters(),models['adamw0'].parameters(),strict=True));assert all(torch.allclose(a,b,rtol=p['rtol'],atol=p['atol']) for a,b in zip(models['adam'].parameters(),models['adamw0'].parameters(),strict=True));row['adam_adamw0_max_abs_error']=error
   assert all(torch.equal(v.cpu(),cp['model'][n]) for n,v in teacher.state_dict().items());records.append(row)
 assert all(float(s['step'])==2 for opt in opts.values() for s in opt.state.values())
 out=dict(all_passed=True,plan_sha256=sha(P),gpu=torch.cuda.get_device_name(),group_names=group_names,parameter_norms=norms,records=records,teacher_unchanged=True,teacher_disconnected=True,checkpoint_saved=False,policy=p['policy']);dest=R/'4090-binary40-regularization-preflight-original-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,records=records)))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
