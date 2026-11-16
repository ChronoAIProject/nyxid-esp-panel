"""Functional gradient witnesses for the fixed allocation-weighted objective."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from contextual_allocation_margin_v1 import allocation_margin_loss
from contextual_generator_contrast_v1 import contextual_contrast_loss

def main():
 torch.set_num_threads(1);assert torch.cuda.is_available()
 z=torch.zeros(2,3,device='cuda',requires_grad=True);t=torch.zeros(3,device='cuda',dtype=torch.long);n=torch.ones_like(t);a=torch.tensor([1.,3.,0.],device='cuda')
 loss=allocation_margin_loss(z,t,n,a);loss.backward();assert torch.equal(z.grad[0],torch.tensor([-.25,-.75,0.],device='cuda'));assert torch.equal(z.grad[1],-z.grad[0]);assert torch.isfinite(z.grad).all()
 uniform=allocation_margin_loss(z,t,n,torch.ones_like(a));old=contextual_contrast_loss(z,t,n);assert torch.equal(uniform,old)
 assert torch.equal(loss,allocation_margin_loss(z,t,n,7*a))
 rejected=0
 for bad in [torch.zeros_like(a),-a,torch.full_like(a,float('nan'))]:
  try:allocation_margin_loss(z,t,n,bad)
  except ValueError:rejected+=1
 assert rejected==3
 paths=[Path(__file__),Path('research/wakeword_p4/training/contextual_allocation_margin_v1.py')]
 out=dict(schema='allocation-margin-functional-v1',gpu=torch.cuda.get_device_name(),loss=float(loss.detach()),uniform_equivalence=True,scale_invariance=True,zero_mass_zero_gradient=True,expected_gradients=True,invalid_weights_rejected=3,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},limits='Functional tensors only; no training updates, waveform or heldout access; no recognition claim.')
 p=Path('research/wakeword_p4/results/4090-allocation-margin-functional-v1.json')
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
