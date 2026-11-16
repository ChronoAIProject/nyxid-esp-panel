"""Analytic cache/index oracle and CUDA interface check, no recognition/equality gate."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_cached_encoder_v1 import CausalCachedEncoder,geometry
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 g=geometry();cases=0
 for history in g['history_frames']:
  for chunk in [1,3,7,32]:
   cache=np.arange(-history,0);end=0
   for _ in range(12):
    incoming=np.arange(end,end+chunk);joined=np.concatenate((cache,incoming));end+=chunk;cache=joined[-history:]
    assert np.array_equal(cache,np.arange(end-history,end))
    # Each dilated kernel uses current and four past positions, never future.
    dilation=history//4
    for i in range(chunk):assert np.array_equal(joined[i:i+history+1:dilation],np.arange(end-chunk+i-history,end-chunk+i+1,dilation))
   cases+=1
 assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(726)
 model=CausalCachedEncoder().cuda().eval();assert sum(p.numel() for p in model.parameters())==g['parameters'];states=model.initial_state(device='cuda');count=0
 with torch.no_grad():
  for _ in range(16):
   x=torch.zeros(1,64,3,device='cuda');y,states=model(x,states);assert y.shape==(1,64,3) and torch.isfinite(y).all()
   assert all(s.shape==(1,64,h) and torch.isfinite(s).all() for s,h in zip(states,g['history_frames'],strict=True));count+=1
 out=dict(schema='causal-cache-clock-function-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/causal_cached_encoder_v1.py')]},geometry=g,analytic_cache_index_cases=cases,all_index_cases_passed=True,cuda_interface_calls=count,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,state_bytes_float32=g['state_elements']*4,state_bytes_if_int8=g['state_elements'],conv_mac_per_second_at100fps=g['conv_mac_per_frame']*100,policy='Untrained architecture/interface feasibility only. Analytic cache/frame index oracle and finiteCUDAshape calls onzeroanalyticinput, no audio, recognition, CPU/GPU or full/chunk neural equality gate. Rightcontext0 byconstruction. Receptivefield253frames spans2520msbetweenframecenters, not completephrase duration guarantee or triggerlatency. No lexicalhead/objective chosen, no qualification. INT8statebytes conditional; currentprototypefloatstates. Quantization acrossstateABI, calibration, exporter, frontendP4port/arena/resourceperformance unproven. Caches/activations/weights/temporarybuffers additional; no pretrainedbehavior inherited.')
 with (R/'4090-causal-cache-clock-function-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
