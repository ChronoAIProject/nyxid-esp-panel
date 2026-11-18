"""Frozen channel-BN mechanical checks, original training batches, zero optimizer updates."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-broadcast-channel-bn-preflight-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 files=dict(read(R/'4090-broadcast-residual-training-plan-v1.json')['files'])
 for p in [Path(__file__),Path('research/wakeword_p4/training/causal_broadcast_channel_bn_v1.py'),R/'4090-streaming-normalization-reference-review-v1.json']:
  files[str(p)]=sha(p)
 with LOCK.open('x') as f:json.dump(dict(files=files,seed=726,training_moment_batches=2,probe_shape=[2,40,300],chunk_frames=3,atol=.0002,rtol=.0002,policy='Untrained fresh model; ordinary channel BN after stem/spectral/temporal convolutions before existing ReLU. Not reference SSN/Swish or a reproduction. First2full original128row training batches update moving moments only, no optimizer. Check original initialization, finite training backward, eval full/chunk/future causality and frozen folding. Training-mode BN is explicitly noncausal across training frames and not chunk invariant. No heldout/evaluation audio, tuning, INT8/P4 or recognition claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=read(LOCK)
 for path,h in p['files'].items():assert sha(path)==h,path
 import torch
 import numpy as np
 from causal_source_balanced_batch_v1 import Sampler,digest
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_broadcast_residual_v1 import CausalBroadcastResidual
 from causal_broadcast_channel_bn_v1 import CausalBroadcastChannelBN
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 torch.manual_seed(726);base=CausalBroadcastResidual().cuda()
 torch.manual_seed(726);model=CausalBroadcastChannelBN().cuda().train()
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
 weights={k:v.detach().clone() for k,v in model.named_parameters()};sampler=Sampler(726);batches=[]
 for i in range(2):
  (x,pw,nw),desc=sampler.batch();model.zero_grad(set_to_none=True)
  z,_=model(torch.from_numpy(x).permute(0,2,1).cuda(),model.initial_state(128,'cuda'));z=z[:,0]
  loss=(torch.nn.functional.softplus(-z)*torch.from_numpy(pw).cuda()+20*torch.nn.functional.softplus(z)*torch.from_numpy(nw).cuda()).sum(1).mean();loss.backward()
  assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
  batches.append(dict(index=i,descriptors=desc,features_sha256=digest(x),positive_sha256=digest(pw),negative_sha256=digest(nw),loss=float(loss.detach())))
 assert all(torch.equal(v,weights[k]) for k,v in model.named_parameters())
 bn=[m for m in model.modules() if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))]
 assert len(bn)==11 and all(int(m.num_batches_tracked)==2 for m in bn)
 model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
 x=torch.tensor(np.random.default_rng(726).normal(size=p['probe_shape']).astype('float32'),device='cuda')
 def compare(a,b):
  assert torch.allclose(a,b,atol=p['atol'],rtol=p['rtol']);return float((a-b).abs().max())
 with torch.no_grad():
  full,fs=model(x,model.initial_state(2,'cuda'));states=model.initial_state(2,'cuda');parts=[]
  for i in range(0,300,3):
   y,states=model(x[:,:,i:i+3],states);parts.append(y)
  stream_error=compare(full,torch.cat(parts,dim=2));cache_error=max(compare(a,b) for a,b in zip(fs,states))
  altered=x.clone();altered[:,:,150:]+=10
  other,_=model(altered,model.initial_state(2,'cuda'));future_error=compare(full[:,:,:50],other[:,:,:50]);assert future_error==0
  folded=model.folded_eval();ff,fc=folded(x,folded.initial_state(2,'cuda'));fold_error=compare(full,ff);fold_cache=max(compare(a,b) for a,b in zip(fs,fc))
 probe=x.detach().clone().requires_grad_(True);z,_=model(probe,model.initial_state(2,'cuda'));z[:,:,:50].sum().backward();future_gradient=float(probe.grad[:,:,150:].abs().max());assert future_gradient==0
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
 result=dict(all_passed=True,plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name(),optimizer_updates=0,initial_original_convolutions_exact=True,trainable_weights_unchanged=True,eval_statistics_unchanged=True,bn_layers=11,moment_batches=2,parameters=sum(v.numel() for v in model.parameters()),folded_parameters=sum(v.numel() for v in folded.parameters()),batches=batches,eval_full_chunk_max_error=stream_error,eval_cache_max_error=cache_error,future_prefix_max_error=future_error,future_gradient_max=future_gradient,folded_output_max_error=fold_error,folded_cache_max_error=fold_cache,limits=p['policy'])
 with (R/'4090-broadcast-channel-bn-preflight-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='batches'}))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
