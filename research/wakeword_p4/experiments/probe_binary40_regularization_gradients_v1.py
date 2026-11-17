"""Fixed50 actual training batches at unchanged initialization, no optimizer steps."""
import gzip,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');P=R/'4090-binary40-regularization-gradient-plan-v1.json';CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def plan():
 files={f:sha(f) for f in json.loads((R/'4090-binary40-regularization-preflight-plan-v1.json').read_text())['files']}
 for f in [Path(__file__),R/'4090-binary40-regularization-preflight-original-v1.json']:files[str(f)]=sha(f)
 with P.open('x') as f:json.dump(dict(files=files,batches=50,weight_decay=.01,lr=.001,eps=1e-8,policy='Original first50Sampler726main128 and same4pairs perstep; initial Binary40final weights fixed for every probe, independent frozen teacher. Save actual main,pair+KL,total gradients and analytic fresh-zero-moment Adam update versus decoupled matrix decay, no optimizer step. No coefficient choice, resampling, scored-source input, model fitting or recognition claim. Fresh-first-update ratios do not represent mature optimizer trajectory.'),f,indent=2)
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
 cp=torch.load(CP,map_location='cpu',weights_only=False);m=Binary40().cuda().eval();m.load_state_dict(cp['model'],strict=True);teacher=Binary40().cuda().eval();teacher.load_state_dict(cp['model'],strict=True);teacher.requires_grad_(False);params=list(m.parameters());names=[n for n,v in m.named_parameters()];flat=torch.cat([v.detach().reshape(-1) for v in params]);decay=torch.cat([(p['lr']*p['weight_decay']*v.detach() if v.ndim>=2 else torch.zeros_like(v)).reshape(-1) for v in params]);sampler=Sampler(726);pairs=Pairs();records=[];arrays={};start=time.monotonic()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:
  for step in range(1,51):
   (x,pw,nw),desc=sampler.batch();old=json.loads(next(f));assert old['step']==step and old['descriptors']==desc
   hashes={k:digest(a) for k,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]};assert all(old[k]==h for k,h in hashes.items());ids=pairs.schedule[step-1]['inventory_indices'];batch=pairs.batch(ids)
   z,_=m(tensor(x).permute(0,2,1),m.initial_state(128,'cuda'));main=bce(z[:,0],tensor(pw),tensor(nw));paired,kl=pair_components(m,teacher,batch)
   def flatten(loss):return torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(loss,params)])
   a=flatten(main);b=flatten(paired+kl);g=a+b;update=p['lr']*g/(g.abs()+p['eps']);assert torch.isfinite(g).all() and torch.isfinite(update).all()
   for name,value in [('main',a),('paired_kl',b),('total',g)]:arrays[f'{step:03d}_{name}']=value.cpu().numpy()
   records.append(dict(step=step,main_hashes=hashes,pair_indices=ids,pair_hashes=[digest(a) for a in batch],main_loss=float(main.detach()),paired_loss=float(paired.detach()),kl=float(kl.detach()),main_gradient_norm=float(a.norm()),paired_kl_gradient_norm=float(b.norm()),total_gradient_norm=float(g.norm()),main_pair_cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)),fresh_adam_update_norm=float(update.norm()),decay_norm=float(decay.norm()),decay_to_fresh_adam_ratio=float(decay.norm()/update.norm().clamp_min(1e-30))))
   assert all(torch.equal(v.detach().cpu(),cp['model'][n]) for n,v in m.state_dict().items()) and all(torch.equal(v.detach().cpu(),cp['model'][n]) for n,v in teacher.state_dict().items());assert all(v.grad is None for v in teacher.parameters())
 arrays['initial_parameter']=flat.cpu().numpy();arrays['decay_update']=decay.cpu().numpy();trace=R/'4090-binary40-regularization-gradient-traces-v1.npz';assert not trace.exists();np.savez_compressed(trace,**arrays)
 summary={k:dict(min=float(min(r[k] for r in records)),median=float(np.median([r[k] for r in records])),max=float(max(r[k] for r in records))) for k in ['main_gradient_norm','paired_kl_gradient_norm','total_gradient_norm','main_pair_cosine','decay_to_fresh_adam_ratio']}
 out=dict(all_passed=True,plan_sha256=sha(P),records=records,summary=summary,parameter_layout=[dict(name=n,shape=list(v.shape),elements=v.numel(),decayed=v.ndim>=2) for n,v in m.named_parameters()],trace=dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size),optimizer_updates=0,student_unchanged=True,teacher_unchanged=True,teacher_disconnected=True,elapsed_seconds=time.monotonic()-start,policy=p['policy']);dest=R/'4090-binary40-regularization-gradient-original-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(summary))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
