"""One complete existing training wave, finite loss/gradients only; no update."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel,soft_target_loss
from generator_overlap_targets_v1 import targets
from continuous_past_mfcc_v1 import FrameClock,ContinuousPastMFCC
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 admission=R/'4090-joined-command-pcm-admission-v1.json';a=json.load(admission.open());duration=R/'4090-joined-generator-token-intervals-v1.json.gz'
 with gzip.open(duration,'rt') as f:rows=json.load(f)['records']
 vocabulary=sorted({t['token_id']+1 for r in rows for t in r['intervals']});labels=[0,*vocabulary];row=rows[0];pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and len(pcm)==row['samples'];assert sha(row['path'])==a['native_files'][row['path']]
 # Native complete waveform only, no context padding/reset at wakeprefix.
 y,ends,mass=targets(len(pcm),0,row['intervals']);frames,times=FrameClock().push(pcm);assert np.array_equal(ends,times) and (mass>0).all();dense=np.zeros((len(y),len(labels)),np.float32)
 for i,label in enumerate(labels):
  if label<y.shape[1]:dense[:,i]=y[:,label]
 assert np.allclose(dense.sum(1),1,atol=1e-6)
 assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(726);torch.cuda.manual_seed_all(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 statepath=B/'matchbox-grid-aligned-v1/frontend.pt';state=torch.load(statepath,map_location='cpu',weights_only=True);front=ContinuousPastMFCC(state).cuda().eval();model=CausalGeneratorUnitModel(len(labels)).cuda()
 with torch.no_grad():features=front(torch.from_numpy(frames.astype(np.float32)/32768).cuda()).T.unsqueeze(0)
 logits,states=model(features,model.encoder.initial_state(device='cuda'));prob=torch.from_numpy(dense.T[None]).cuda();loss=soft_target_loss(logits,prob);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 out=dict(schema='causal-generator-unit-loss-function-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/causal_generator_unit_model_v1.py'),Path('research/wakeword_p4/training/causal_cached_encoder_v1.py'),Path('research/wakeword_p4/training/generator_overlap_targets_v1.py'),Path('research/wakeword_p4/training/continuous_past_mfcc_v1.py'),admission,duration,statepath,Path(row['path'])]},dense_to_original_label=labels,classes=len(labels),parameters=sum(p.numel() for p in model.parameters()),readout_parameters=sum(p.numel() for p in model.readout.parameters()),training_native_path=row['path'],complete_pcm_samples=len(pcm),frames=len(frames),logits_shape=list(logits.shape),loss=float(loss.detach()),all_parameter_gradients_finite=True,optimizer_updates=0,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,policy='One training-onlyfunctioncheck, nooptimization/inferencequalification. FullnativePCM, noRIR/noisemodification orcrop. Densevocabularyboundtoall1080trainingmetadata,originalunitIDs retainedincludingpadding/space/punctuation. LossunweightedframewisesoftCE, thischecknottrainingrecipe. NoCPU/GPUgradientagreementgate, noASR/CTCblankoracousticphoneclaim. Retainedreadoutoutputsorderedframeunitlogits; actualtraineddecoder/eventpolicy, INT8head/export/statecalibration, backgroundhandling andbare/multienginesupport remainunresolved. v30unchanged.')
 with (R/'4090-causal-generator-unit-loss-function-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:out[k] for k in ['classes','parameters','frames','loss','all_parameter_gradients_finite','optimizer_updates']}))
if __name__=='__main__':main()
