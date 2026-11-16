"""One full matched24record CUDA batch, loss/padding checks, zero updates."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_full_batch_v1 import batch,recording_normalized_loss
from causal_unit_sampler_v1 import Sampler,INDEX
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(726);torch.cuda.manual_seed_all(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 statepath=B/'matchbox-grid-aligned-v1/frontend.pt';front=ContinuousPastMFCC(torch.load(statepath,map_location='cpu',weights_only=True)).cuda().eval();sampler=Sampler();x,p,m,d=batch(sampler,front);model=CausalGeneratorUnitModel(len(sampler.data['dense_to_original_label'])).cuda();logits,states=model(x,model.encoder.initial_state(batch=24,device='cuda'));logits.retain_grad();loss=recording_normalized_loss(logits,p,m);loss.backward();assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters());assert bool((logits.grad.permute(0,2,1)[~m]==0).all())
 # Arithmetic oracle: equal recording mass despite differing valid lengths.
 a=torch.zeros(2,3,7,device='cuda',requires_grad=True);q=torch.zeros_like(a);q[:,1,:]=1;mask=torch.zeros(2,7,dtype=torch.bool,device='cuda');mask[0,:2]=True;mask[1,:]=True;l=recording_normalized_loss(a,q,mask);l.backward();mass=a.grad.abs().sum((1,2));assert torch.isclose(mass[0],mass[1]);assert bool((a.grad[0,:,2:]==0).all())
 files=[Path(__file__),INDEX,statepath,*[Path('research/wakeword_p4/training')/n for n in ['causal_unit_full_batch_v1.py','causal_unit_sampler_v1.py','causal_generator_unit_model_v1.py','causal_cached_encoder_v1.py','continuous_past_mfcc_v1.py','generator_overlap_targets_v1.py']]]
 out=dict(schema='causal-unit-full-batch-function-v1',files={str(f):sha(f) for f in files},batch_shape=list(x.shape),target_shape=list(p.shape),valid_frames=int(m.sum()),padding_frames=int((~m).sum()),loss=float(loss.detach()),all_parameter_gradients_finite=True,padding_logit_gradients_exact_zero=True,recording_mass_arithmetic_oracle_passed=True,descriptors=d,optimizer_updates=0,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),policy='Actual24existingfulltrainingrenders, balanced4carrier*6class; noPCMcut/rewrite, allcompletefrontendframesretained, finishresidualreported/nozeropadframe. Featurepaddingonlyafterlastvalidframe, nofuturecontextbyarchitecture, separatezeroinitialcacheperrecord. Finalstatesforpaddedrecordsnotreusableasrealendstates. Lossmeanperrecord/validlengththenmean24, noimplicitlongcommandweight. Zerooptimization, noCPU/GPUagreement orrecognitionclaim. Realunknownbackgroundnotincludedandrequiresseparatenonwakeobjectivewithoutpseudotranscript. Model/export/eventqualificationnotestablished.')
 with (R/'4090-causal-unit-full-batch-function-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:out[k] for k in ['batch_shape','valid_frames','padding_frames','loss','peak_cuda_allocated_bytes','optimizer_updates']}))
if __name__=='__main__':main()
