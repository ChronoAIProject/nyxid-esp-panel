"""Training-wave normalization parity, finite gradients and causality admission."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_frame_normalized_encoder_v1 import CausalFrameNormalizedEncoder
from causal_prospective_sampler_v1 import Sampler,INDEX
from causal_unit_full_batch_v1 import batch
from continuous_past_mfcc_v1 import ContinuousPastMFCC

def main():
 torch.set_num_threads(1);torch.manual_seed(726);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();s=Sampler();x,_,mask,d=batch(s,front);gpu=CausalFrameNormalizedEncoder().cuda();cpu=CausalFrameNormalizedEncoder();cpu.load_state_dict(gpu.state_dict());x=x.detach().requires_grad_(True);a=gpu.normalize(x);b=cpu.normalize(x.detach().cpu());err=float((a.detach().cpu()-b).abs().max());assert torch.allclose(a.detach().cpu(),b,atol=1e-5,rtol=1e-5)
 y,_=gpu(x,gpu.encoder.initial_state(24,'cuda'));loss=y.square().mean();loss.backward();assert torch.isfinite(x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in gpu.parameters());changed=x.detach().clone();cut=100;changed[:,:,cut:]+=123
 with torch.no_grad():z,_=gpu(changed,gpu.encoder.initial_state(24,'cuda'))
 future_error=float((z[:,:,:cut]-y.detach()[:,:,:cut]).abs().max());assert future_error==0
 out=dict(schema='frame-normalization-real-admission-v1',gpu=torch.cuda.get_device_name(),batch=list(x.shape),valid_frames=int(mask.sum()),cpu_gpu_normalization_max_abs=err,future_perturbation_prefix_max_abs=future_error,finite_input_and_parameter_gradients=True,optimizer_updates=0,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),INDEX,Path('research/wakeword_p4/training/causal_frame_normalized_encoder_v1.py')]},limitations='First24training-only complete waves from16conditioning pool. Numerical/causality function check only; no optimization/recognition/INT8 qualification. Development remains reused, not globally fresh.')
 with Path('research/wakeword_p4/results/4090-frame-normalization-real-admission-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
