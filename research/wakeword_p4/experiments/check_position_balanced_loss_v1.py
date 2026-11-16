"""Actual complete training batch position-mass and gradient admission."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_full_batch_v1 import batch,recording_normalized_loss
from causal_unit_sampler_v1 import Sampler
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC
from generator_position_balanced_loss_v1 import position_weights,position_balanced_loss

def main():
 torch.set_num_threads(1);torch.manual_seed(726);assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sampler=Sampler();front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();x,q,mask,desc=batch(sampler,front);model=CausalGeneratorUnitModel(34).cuda();logits,_=model(x,model.encoder.initial_state(batch=24,device='cuda'));losses=[];positions=0;maximum=0.
 for i,d in enumerate(desc):
  r=sampler.rows[d['index']];kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {};a,ends=position_weights(r['samples'],r['lead'],r['intervals'],**kwargs);n=len(ends);assert n==d['frames'];error=float(np.abs(a.sum(1)-1/len(a)).max());maximum=max(maximum,error);assert error<1e-6;positions+=len(a);labels=[sampler.data['dense_to_original_label'].index(t['token_id']+1) for t in r['intervals']];losses.append(position_balanced_loss(logits[i,:,:n],torch.tensor(a,device='cuda'),labels))
 frame=recording_normalized_loss(logits,q,mask);position=torch.stack(losses).mean();params=list(model.parameters());gf=torch.cat([g.flatten() for g in torch.autograd.grad(frame,params,retain_graph=True)]);gp=torch.cat([g.flatten() for g in torch.autograd.grad(position,params,retain_graph=True)]);combined=.5*(frame+position);combined.backward();assert all(torch.isfinite(p.grad).all() for p in params)
 out=dict(schema='position-balanced-loss-admission-v1',gpu=torch.cuda.get_device_name(),recordings=24,positions=positions,max_position_mass_error=maximum,frame_loss=float(frame.detach()),position_loss=float(position.detach()),frame_parameter_gradient_l2=float(gf.norm()),position_parameter_gradient_l2=float(gp.norm()),gradient_cosine=float(torch.nn.functional.cosine_similarity(gf[None],gp[None])),mixture='0.5 frame-normalized + 0.5 position-normalized',all_combined_gradients_finite=True,optimizer_updates=0,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/wakeword_p4/training/generator_position_balanced_loss_v1.py')]},limitations='One fixed training batch, untrained seed726 model. Equal generator position mass is an objective choice, not acoustic truth or evidence of recognition improvement. No background objective or heldout use.')
 with Path('research/wakeword_p4/results/4090-position-balanced-loss-admission-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
