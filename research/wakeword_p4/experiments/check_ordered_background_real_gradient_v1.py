"""Training-only full-wave background gradient admission; no optimizer updates."""
import gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC,FrameClock
from ordered_generator_matcher_v1 import PATTERN,BOUNDARIES
from ordered_unit_background_loss_v1 import acceptance_probability,background_rejection_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.manual_seed(726);assert torch.cuda.is_available()
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 index=Path('research/wakeword_p4/results/4090-causal-unit-training-index-v1.json.gz')
 data=json.load(gzip.open(index));mapping=data['dense_to_original_label']
 # Original matcher IDs are token IDs; supervision labels are token ID + 1.
 pattern=tuple(mapping.index(t+1) for t in PATTERN);boundaries=tuple(mapping.index(t+1) for t in BOUNDARIES)
 source=Path('.local/wakeword/matchbox-long-material-v1/records.json')
 records=sorted((r for r in json.load(source.open()) if r['kind']=='libri-training-background'),key=lambda r:(r['samples'],r['path']))[:3]
 statepath=Path('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt')
 front=ContinuousPastMFCC(torch.load(statepath,weights_only=True,map_location='cpu')).cuda().eval()
 model=CausalGeneratorUnitModel(len(mapping)).cuda();results=[]
 for r in records:
  assert r['role']=='training' and sha(r['path'])==r['sha256']
  pcm,sr=sf.read(r['path'],dtype='int16');assert sr==16000 and pcm.ndim==1 and len(pcm)==r['samples']
  clock=FrameClock();frames,ends=clock.push(pcm)
  with torch.no_grad():x=front(torch.tensor(frames.astype(np.float32)/32768,device='cuda')).T[None]
  model.zero_grad(set_to_none=True);logits,_=model(x,model.encoder.initial_state(device='cuda'));logits.retain_grad()
  p=logits.double().softmax(1);q=acceptance_probability(p,pattern,boundaries);loss=background_rejection_loss(p,pattern,boundaries);loss.backward()
  grads=torch.cat([v.grad.flatten() for v in model.parameters()]);assert torch.isfinite(grads).all()
  results.append(dict(path=r['path'],sha256=r['sha256'],pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),samples=len(pcm),frames=len(frames),clock=clock.finish(),acceptance_probability=float(q.detach()),loss=float(loss.detach()),parameter_gradient_l2=float(grads.norm()),parameter_gradient_abs_max=float(grads.abs().max()),logit_gradient_l2=float(logits.grad.norm()),nonzero_parameter_gradients=int(torch.count_nonzero(grads))))
 out=dict(schema='ordered-background-real-gradient-v1',seed=726,gpu=torch.cuda.get_device_name(),dense_pattern=pattern,dense_boundaries=boundaries,selection='Three shortest admitted training-only Libri backgrounds, deterministic samples/path order; full PCM, no crop, no transcript used.',files={str(p):sha(p) for p in [Path(__file__),index,source,statepath,Path('research/wakeword_p4/training/ordered_unit_background_loss_v1.py')]},records=results,optimizer_updates=0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),limitations='Untrained model only; sequence-probability surrogate differs from deployed greedy event decoder. Nonzero gradients do not establish useful background rejection. No heldout data consumed. No model qualification.')
 dest=Path('research/wakeword_p4/results/4090-ordered-background-real-gradient-v1.json')
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
