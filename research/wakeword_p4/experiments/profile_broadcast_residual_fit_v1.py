"""Prelocked two-original-batch activation/gradient diagnostic; zero updates."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments')
LOCK=R/'4090-broadcast-residual-fit-profile-plan-v1.json'
T=Path('.local/wakeword/broadcast-residual-training-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 files=dict(read(R/'4090-broadcast-residual-training-plan-v1.json')['files'])
 for p in [Path(__file__),T/'initial.pt',T/'final.pt',R/'4090-broadcast-residual-training-plan-v1.json',R/'4090-broadcast-residual-broad-fit-review-v1.json']:
  files[str(p)]=sha(p)
 spec=dict(files=files,seed=726,batches=2,rows_per_batch=128,checkpoints=['initial','final'],optimizer_updates=0,policy='First two original full training batches, unchanged features/masks, no evaluation audio. Original weighted BCE backward for descriptive gradient norms only; no optimizer, rescue, threshold/seed/checkpoint choice. Module preactivation <=0 fraction is a ReLU-input diagnostic only for stem/spectral/temporal; pointwise/head are signed and not directly ReLU outputs. Two batches cannot establish general inactivity or causal explanation. No normalization variant or new training authorized by these statistics alone.')
 with LOCK.open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 spec=read(LOCK)
 for p,h in spec['files'].items():assert sha(p)==h,p
 import torch
 import numpy as np
 from causal_source_balanced_batch_v1 import Sampler,digest
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_broadcast_residual_v1 import CausalBroadcastResidual
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 sampler=Sampler(726);batches=[sampler.batch() for _ in range(2)];rows=[]
 for name in spec['checkpoints']:
  cp=torch.load(T/(name+'.pt'),map_location='cpu',weights_only=False)
  assert cp['step']=={'initial':0,'final':20000}[name]
  model=CausalBroadcastResidual().cuda().eval();model.load_state_dict(cp['model'],strict=True)
  original={k:v.detach().clone() for k,v in model.state_dict().items()}
  for batch,((x,pw,nw),descriptors) in enumerate(batches):
   activations={};hooks=[]
   def capture(key):
    def hook(module,args,out):
     a=out.detach();axes=tuple(i for i in range(a.ndim) if i!=1)
     activations[key]=dict(shape=list(a.shape),mean=float(a.mean()),rms=float(a.square().mean().sqrt()),max_abs=float(a.abs().max()),nonpositive_fraction=float((a<=0).float().mean()),channels_all_nonpositive=int((a.amax(dim=axes)<=0).sum()))
    return hook
   for key,module in model.named_modules():
    if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d)):hooks.append(module.register_forward_hook(capture(key)))
   model.zero_grad(set_to_none=True)
   z,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(128,'cuda'));z=z[:,0]
   pos=torch.from_numpy(pw).cuda();neg=torch.from_numpy(nw).cuda()
   positive=(torch.nn.functional.softplus(-z)*pos).sum(1).mean()
   negative=(20*torch.nn.functional.softplus(z)*neg).sum(1).mean()
   loss=positive+negative;loss.backward()
   gradients={}
   for key,module in model.named_modules():
    if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d)):
     parts=[p.grad.detach().flatten() for p in module.parameters()];g=torch.cat(parts)
     assert torch.isfinite(g).all();gradients[key]=dict(l2=float(g.norm()),max_abs=float(g.abs().max()),zero_fraction=float((g==0).float().mean()))
   for h in hooks:h.remove()
   assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
   prob=torch.sigmoid(z.detach())
   rows.append(dict(checkpoint=name,batch=batch,descriptors=descriptors,hashes=dict(features=digest(x),positive=digest(pw),negative=digest(nw)),positive_loss=float(positive.detach()),weighted_negative_loss=float(negative.detach()),loss=float(loss.detach()),positive_mask_probability=float((prob*pos).sum()/pos.sum()),negative_mask_probability=float((prob*neg).sum()/neg.sum()),activations=activations,gradients=gradients,weights_unchanged=True))
 out=dict(plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name(),optimizer_updates=0,rows=rows,limits=spec['policy'])
 with (R/'4090-broadcast-residual-fit-profile-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps([dict(checkpoint=r['checkpoint'],batch=r['batch'],loss=r['loss']) for r in rows]))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
