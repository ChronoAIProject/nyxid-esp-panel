"""All90 training-only supervised quiet/history gradient compatibility, no step."""
import hashlib,json,sys,subprocess,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-binary40-history-gradient-plan-v1.json';SOURCE=R/'4090-binary40-context-stratified-original-v1.json';CHECK=Path('.local/wakeword/binary40-training-v1/final.pt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 d=json.loads(SOURCE.read_text());paths=[Path(__file__),SOURCE,Path(d['trace']['path']),CHECK,Path('research/wakeword_p4/training/causal_binary40_v1.py')]
 PLAN.write_text(json.dumps(dict(files={str(p):sha(p) for p in paths},policy='All90fixed original training strata, no row score selection. Original final weights, complete saved physical diagonal features from both histories, zero start. Positive support automatic/native endpoint+480..4320; negative support before target starts including complete preceding negative. Original independently normalized positiveBCE+20negativeBCE. Mean over90sources separately per history. Record aggregate/perlayer exact gradients and dot/cosine without any optimizer, projection, coefficient fitting, mask changes or evaluation access. This is a diagnostic surrogate of proposed physical histories, not the original full128 sampler loss or proof of training/generalization benefit.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 model=Binary40().cuda().eval();cp=torch.load(CHECK,map_location='cpu',weights_only=False);assert cp['step']==20000;model.load_state_dict(cp['model'],strict=True)
 before={k:v.detach().clone() for k,v in model.state_dict().items()};d=json.loads(SOURCE.read_text());grads={};losses={};start=time.monotonic()
 with np.load(d['trace']['path']) as z:
  for history in ['silence','speech']:
   model.zero_grad(set_to_none=True);values=[]
   for row in d['records']:
    x=z[f'{row["index"]}_{history}_features'];logits,_=model(torch.from_numpy(x.T.copy()[None]).cuda(),model.initial_state(1,'cuda'));v=logits[0,0]
    support=(3*np.arange(1,len(v)+1)+2)*160;end=row['prefix_samples']+row['source']['prefix_reference_samples'];pos=(support>=end+480)&(support<=end+4320);neg=support<=row['prefix_samples'];assert pos.any() and neg.any() and not np.any(pos&neg)
    loss=torch.nn.functional.softplus(-v[torch.from_numpy(pos).cuda()]).mean()+20*torch.nn.functional.softplus(v[torch.from_numpy(neg).cuda()]).mean();values.append(float(loss.detach()));(loss/90).backward()
   grads[history]={name:v.grad.detach().cpu().numpy().copy() for name,v in model.named_parameters()};losses[history]=values
 assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 layers={};dots=qn=sn=0.
 for name,q in grads['silence'].items():
  s=grads['speech'][name];q=q.astype(np.float64);s=s.astype(np.float64);assert np.isfinite(q).all() and np.isfinite(s).all()
  a=float(np.sum(q*q));b=float(np.sum(s*s));dot=float(np.sum(q*s));dots+=dot;qn+=a;sn+=b;layers[name]=dict(silence_norm=a**.5,speech_norm=b**.5,dot=dot,cosine=dot/(a*b)**.5 if a*b>0 else None)
 out=dict(plan_sha256=sha(PLAN),rows=90,optimizer_updates=0,weights_unchanged=True,losses=losses,layers=layers,aggregate=dict(silence_norm=qn**.5,speech_norm=sn**.5,dot=dots,cosine=dots/(qn*sn)**.5),elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),policy=p['policy'])
 trace=R/'4090-binary40-history-gradient-traces-v1.npz';np.savez_compressed(trace,**{h+'__'+n:g for h,gs in grads.items() for n,g in gs.items()});out['trace']=dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size)
 (R/'4090-binary40-history-gradient-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['aggregate']))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
