"""Full training index delay coverage and actual CUDA batch gradient check."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from delayed_generator_targets_v1 import DELAY_FRAMES,delay_frame_targets,delay_position_weights
from generator_position_balanced_loss_v1 import position_weights
from causal_unit_sampler_v1 import Sampler,INDEX
from causal_unit_full_batch_v1 import batch,recording_normalized_loss
from continuous_past_mfcc_v1 import ContinuousPastMFCC
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel

def main():
 torch.set_num_threads(1);torch.manual_seed(726);assert torch.cuda.is_available();data=json.load(gzip.open(INDEX));positions=0;minimum=None
 for r in data['records']:
  kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {};a,ends=position_weights(r['samples'],r['lead'],r['intervals'],**kwargs);shifted=delay_position_weights(a);assert np.array_equal(shifted[:,4:],a[:,:-4]);positions+=len(a);occupied=np.flatnonzero(a.sum(0)>0);margin=len(ends)-1-int(occupied[-1]);minimum=margin if minimum is None else min(minimum,margin)
 sampler=Sampler();front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();x,p,mask,d=batch(sampler,front);q=delay_frame_targets(p,mask);assert torch.equal(q[:,:,:4].sum(1),torch.ones_like(q[:,:,:4].sum(1)))
 for i,row in enumerate(d):
  n=row['frames'];assert torch.equal(q[i,:,4:n],p[i,:,:n-4]);assert bool((q[i,:,n:]==0).all())
 model=CausalGeneratorUnitModel(34).cuda();y,_=model(x,model.encoder.initial_state(batch=24,device='cuda'));loss=recording_normalized_loss(y,q,mask);loss.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters())
 out=dict(schema='delayed-generator-target-admission-v1',delay_frames=4,delay_samples=640,delay_ms=40,recordings=len(data['records']),positions=positions,minimum_trailing_unallocated_frames=minimum,position_mass_dropped=0,features_shifted=False,pcm_padding=False,first4frames_unallocated=True,padded_frames_zero_mass=True,frame_loss=float(loss.detach()),all_gradients_finite=True,optimizer_updates=0,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),INDEX,Path('research/wakeword_p4/training/delayed_generator_targets_v1.py')]},limitations='Training-only allocation coverage and one CUDA gradient batch. Fixed output label delay has40ms declared cost, no acoustic truth or recognition claim.')
 with Path('research/wakeword_p4/results/4090-delayed-generator-target-admission-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
