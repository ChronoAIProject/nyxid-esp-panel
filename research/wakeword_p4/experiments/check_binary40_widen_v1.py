"""Capacity proposal functional gates; no optimizer update or recognition replay."""
import json,sys
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
import torch
from causal_binary40_v1 import Binary40
from causal_binary40_wide_v1 import widen_from,geometry
from causal_source_balanced_batch_v1 import Sampler,digest

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
 original=Binary40().cuda().eval();original.load_state_dict(cp['model'],strict=True)
 assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in original.state_dict().items())
 wide=widen_from(original).eval();(x,p,n),desc=Sampler(726).batch();t=torch.from_numpy(x).permute(0,2,1).cuda()
 with torch.inference_mode():
  a,_=original(t,original.initial_state(128,'cuda'));b,_=wide(t,wide.initial_state(128,'cuda'))
  error=float((torch.sigmoid(a)-torch.sigmoid(b)).abs().max());logit_error=float((a-b).abs().max());assert error<1e-4,error
  stream=t[:2];full,final=wide(stream,wide.initial_state(2,'cuda'));checks=[]
  for size in [3,6,37,128,846]:
   state=wide.initial_state(2,'cuda');pending=stream[:,:,:0];outputs=[]
   for off in range(0,846,size):
    pending=torch.cat((pending,stream[:,:,off:off+size]),dim=2);count=pending.shape[2]//3*3
    if count:
     y,state=wide(pending[:,:,:count],state);outputs.append(y);pending=pending[:,:,count:]
   actual=torch.cat(outputs,dim=2);e=float((actual-full).abs().max());assert torch.allclose(actual,full,atol=1e-4,rtol=1e-5)
   assert all(torch.allclose(v,w,atol=1e-4,rtol=1e-5) for v,w in zip(state,final));checks.append(dict(chunk_frames=size,max_logit_error=e))
  changed=stream.clone();changed[:,:,423:]+=1;y,_=wide(changed,wide.initial_state(2,'cuda'));assert torch.equal(y[:,:,:141],full[:,:,:141])
 wide.zero_grad();z,_=wide(t,wide.initial_state(128,'cuda'));loss=(torch.nn.functional.softplus(-z[:,0])*torch.from_numpy(p).cuda()+20*torch.nn.functional.softplus(z[:,0])*torch.from_numpy(n).cuda()).sum(1).mean();loss.backward()
 assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in wide.parameters())
 gradient_asymmetry=float((wide.stem.weight.grad[:64]-wide.stem.weight.grad[64:]).abs().max());assert gradient_asymmetry>0
 out=dict(schema='binary40-widen-functional-v1',gpu=torch.cuda.get_device_name(),checkpoint_sha256=sha('.local/wakeword/binary40-training-v1/final.pt'),source_sha256={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/causal_binary40_wide_v1.py')]},geometry=geometry(),parameters=sum(p.numel() for p in wide.parameters()),batch_features_sha256=digest(x),initial_max_probability_error=error,initial_max_logit_error=logit_error,chunk_checks=checks,causality_prior141_exact=True,finite_gradients=True,stem_duplicate_gradient_difference=gradient_asymmetry,optimizer_updates=0,policy='Fixed128channels and zero-sum split noise0.01RMS/seed727. Engineering gate only; no training benefit, INT8 feasibility or P4 claim. Future comparison must include equal continuation64control with same data/optimizer/final-only protocol; not a width/noise/seed sweep.')
 dest=Path('research/wakeword_p4/results/4090-binary40-widen-functional-v1.json');assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
