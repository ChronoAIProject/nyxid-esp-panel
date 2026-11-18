"""Two-batch independent source/mask reconstruction and CUDA backward, no updates."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from causal_cross_engine_carrier_batch_v1 import Sampler
from causal_source_balanced_batch_v1 import Sampler as Original,digest
from causal_sequence_supervision_v1 import masks
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_binary40_span_v1 import Binary40Span,CONTROL
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
m=Binary40Span(CONTROL).cuda();parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
assert all(torch.equal(v.cpu(),parent[k]) for k,v in m.state_dict().items())
s=Sampler(726);o=Original(726);rng=np.random.default_rng(242726);records=[]
for step in range(2):
 (x,p,n),d=s.batch();base,bd=o.batch()
 assert len(d)==136 and d[:128]==bd and all(np.array_equal(a[:128],b) for a,b in zip((x,p,n),base,strict=True))
 selected=[]
 for pair in range(4):
  engine=['piper','kokoro'][int(rng.integers(2))];voices=s.voices[engine];voice=voices[int(rng.integers(len(voices)))];pool=s.pools[engine][voice];positive=pool[int(rng.integers(27))];k=positive['source_pair_key']
  negative=s.negatives[engine][voice,float(k['scale']),k['carrier'],positive['variant']]
  for j,(row,label) in enumerate([(positive,1),(negative,0)]):
   slot=128+pair*2+j;desc=d[slot];frames=row['frames']
   assert desc['engine']==engine and desc['conditioning_id']==voice and desc['index']==row['offset'] and desc['label']==label and desc['pair']==pair
   assert desc['carrier']==k['carrier'] and desc['speed']==float(k['scale']) and desc['variant']==positive['variant']
   raw=s.arrays['positive' if label else engine][row['offset']:row['offset']+frames]
   xx=np.zeros((846,40),np.float32);xx[:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
   _,pw,nw=masks(frames,label,(row['lead_frame']+2)*160 if label else None,row['generated_prefix_end_sample'] if label else None)
   pp=np.zeros(282,np.float32);nn=pp.copy();pp[:len(pw)]=pw;nn[:len(nw)]=nw
   assert np.array_equal(xx,x[slot]) and np.array_equal(pp,p[slot]) and np.array_equal(nn,n[slot])
  selected.append(dict(engine=engine,voice=voice,positive_offset=positive['offset'],negative_offset=negative['offset']))
 assert np.allclose(p.sum(1),[a['label'] for a in d]) and np.allclose(n.sum(1),1)
 m.zero_grad(set_to_none=True);z,_=m(torch.from_numpy(x).permute(0,2,1).cuda(),m.initial_state(136,'cuda'));z=z[:,0,:]
 loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(p).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(n).cuda()).sum()/128
 assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
 assert all(torch.equal(v.cpu(),parent[k]) for k,v in m.state_dict().items())
 records.append(dict(batch=step+1,features_sha256=digest(x),positive_sha256=digest(p),negative_sha256=digest(n),selected=selected,loss=float(loss.detach())))
out=dict(all_passed=True,batches=2,original_rows_verified=256,appended_rows_independently_reconstructed=16,actual_backward_batch_size=136,optimizer_updates=0,weights_unchanged=True,gpu=torch.cuda.get_device_name(),records=records,source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path(__file__).with_name('causal_cross_engine_carrier_batch_v1.py')]})
with Path('research/wakeword_p4/results/4090-cross-engine-carrier-preflight-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out),flush=True)
