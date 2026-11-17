"""Structural random-input preflight only; no recognition or updates."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-multiscale-structural-plan-v2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 paths=[Path(__file__),Path('research/wakeword_p4/training/causal_binary40_multiscale_v1.py'),Path('research/wakeword_p4/training/causal_binary40_span_v1.py'),Path('research/wakeword_p4/training/gpu_python.py'),Path('research/wakeword_p4/experiments/binary40_multiscale_readout_proposal_v1.md')]
 with LOCK.open('x') as f:json.dump(dict(files={str(p):sha(p) for p in paths},seed=726,optimizer_updates=0,scope='Random-input structural CUDA4090 only; training-batch and INT8 feasibility gates remain separate.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_span_v1 import Binary40Span,CONTROL
 from causal_binary40_multiscale_v1 import Binary40Multiscale
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert '4090' in torch.cuda.get_device_name()
 torch.manual_seed(726);base=Binary40Span(CONTROL).cuda().eval();torch.manual_seed(726);model=Binary40Multiscale().cuda().eval()
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if not k.startswith('head.'))
 assert model.geometry()['parameters']==31233 and sum(s.numel() for s in model.initial_state())==6864
 # Independent integer support enumeration from causal convolution offsets.
 offsets={0,1,2};supports=[]
 for d in [0,1,2,4,8,10]:
  if d:offsets={a+3*b*d for a in offsets for b in range(5)}
  head_offsets=offsets|{a+3 for a in offsets};supports.append(max(head_offsets)-min(head_offsets)+1)
 assert supports==[6,18,42,90,186,306]
 with torch.no_grad():model.head.weight.zero_();model.head.weight[:,320:,:].copy_(base.head.weight);model.head.bias.copy_(base.head.bias)
 before={k:v.clone() for k,v in model.state_dict().items()};x=torch.randn(1,40,1800,device='cuda');checks={}
 with torch.inference_mode():
  target,_=base(x,base.initial_state(device='cuda'));full,fs=model(x,model.initial_state(device='cuda'));checks['embedded_original_max_error']=float((target-full).abs().max());assert torch.allclose(target,full,atol=1e-5,rtol=1e-5)
  for chunk in [3,6,111,300,1800]:
   states=model.initial_state(device='cuda');parts=[]
   for i in range(0,1800,chunk):
    y,states=model(x[:,:,i:i+chunk],states);parts.append(y)
   err=float((full-torch.cat(parts,2)).abs().max());checks[f'chunk_{chunk}_max_error']=err;assert err<1e-4
   assert all(torch.allclose(a,b,atol=1e-4,rtol=1e-5) for a,b in zip(fs,states))
  changed=x.clone();changed[:,:,900:]+=torch.randn_like(changed[:,:,900:]);y,_=model(changed,model.initial_state(device='cuda'));assert torch.equal(full[:,:,:300],y[:,:,:300]);checks['future_invariance']=True
  changed=x.clone();changed[:,:,:1800-308]+=torch.randn_like(changed[:,:,:1800-308]);y,_=model(changed,model.initial_state(device='cuda'));assert torch.equal(full[:,:,-1],y[:,:,-1]);checks['outside_support_invariance']=True
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
 out=dict(plan_sha256=sha(LOCK),geometry=model.geometry(),checks=checks,optimizer_updates=0,gpu=torch.cuda.get_device_name(),scope=p['scope'],training_batch_gate_pending=True,int8_gate_pending=True)
 with (R/'4090-multiscale-structural-original-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
