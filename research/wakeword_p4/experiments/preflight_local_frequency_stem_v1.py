import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
import torch
from causal_binary40_local_stem_v1 import Binary40LocalStem
from causal_binary40_span_v1 import Binary40Span,CONTROL
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
m=Binary40LocalStem().cuda();x=torch.randn(2,40,300,device='cuda');initial={k:v.detach().clone() for k,v in m.state_dict().items()}
y,fs=m(x,m.initial_state(2,'cuda'));state=m.initial_state(2,'cuda');parts=[]
for i in range(0,300,3):
 z,state=m(x[:,:,i:i+3],state);parts.append(z)
error=float((y-torch.cat(parts,2)).abs().max());state_error=max(float((a-b).abs().max()) for a,b in zip(fs,state));assert error<1e-4 and state_error<1e-4
loss=y.square().mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());assert all(torch.equal(v,initial[k]) for k,v in m.state_dict().items())
a=m.geometry();b=Binary40Span(CONTROL).geometry();assert a['state_elements']==b['state_elements'];p=Path('research/wakeword_p4/training/causal_binary40_local_stem_v1.py')
out=dict(gpu=torch.cuda.get_device_name(),seed=726,optimizer_updates=0,full_chunk_max_error=error,state_max_error=state_error,all_gradients_finite=True,weights_unchanged=True,prototype=a,original=b,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),limits='Random input CUDA mechanical preflight only, not training, speech recognition, INT8 export or P4 resource evidence. Ordered frontend channels must be validated against the pinned frontend before training; locality is a hypothesis.')
with open('research/wakeword_p4/results/4090-local-frequency-stem-preflight-v1.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
