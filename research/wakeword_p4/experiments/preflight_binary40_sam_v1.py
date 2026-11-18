"""Independent SAM perturbation/second-gradient check, zero optimizer updates."""
import copy,hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_span_v1 import Binary40Span,CONTROL
from causal_source_balanced_batch_v1 import Sampler,digest
from sam_step_v1 import sam_gradients
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
m=Binary40Span(CONTROL).cuda();original=copy.deepcopy(m.state_dict())
parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
assert all(torch.equal(v.cpu(),parent[k]) for k,v in original.items())
s=Sampler(726);o=torch.optim.Adam(m.parameters(),lr=.001);records=[]
for step in range(2):
 (x,p,n),descriptors=s.batch();x=torch.from_numpy(x).permute(0,2,1).cuda();p=torch.from_numpy(p).cuda();n=torch.from_numpy(n).cuda()
 def loss(model):
  z,_=model(x,model.initial_state(128,'cuda'));z=z[:,0,:]
  return (torch.nn.functional.softplus(-z)*p+20*torch.nn.functional.softplus(z)*n).sum()/128
 ref=copy.deepcopy(m);first=loss(ref);first.backward()
 flat=torch.cat([v.grad.flatten().double() for v in ref.parameters()]);norm=flat.norm();deltas=[]
 with torch.no_grad():
  for v in ref.parameters():
   delta=(v.grad.double()*(.05/(norm+1e-12))).float();v.add_(delta);deltas.append(delta)
 # float64 norm yields small rounding differences; independent tolerance fixed here, not dev tuned.
 ref.zero_grad(set_to_none=True);second=loss(ref);second.backward()
 a,b,c=sam_gradients(m,o,lambda:loss(m),.05)
 assert torch.allclose(a,first,rtol=1e-5,atol=1e-6)
 assert abs(b-float(second.detach()))<=1e-4
 errors=[]
 for u,v in zip(m.parameters(),ref.parameters(),strict=True):
  assert torch.allclose(u.grad,v.grad,rtol=1e-3,atol=1e-5)
  errors.append(float((u.grad-v.grad).abs().max()))
 assert all(torch.equal(v,original[k]) for k,v in m.state_dict().items()) and len(o.state)==0
 records.append(dict(batch=step+1,first_loss=float(a),second_loss=b,norm=c,max_gradient_error=max(errors),feature_sha256=digest(x.cpu().permute(0,2,1).numpy()),exact_restoration=True))
result=dict(schema='binary40-sam-preflight-v1',all_passed=True,optimizer_updates=0,gpu=torch.cuda.get_device_name(),records=records,source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path(__file__).with_name('sam_step_v1.py')]})
with Path('research/wakeword_p4/results/4090-binary40-sam-preflight-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
