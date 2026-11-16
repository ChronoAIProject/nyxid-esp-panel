import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_sampler_v1 import Sampler,INDEX
from causal_unit_full_batch_v1 import batch
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
from continuous_past_mfcc_v1 import ContinuousPastMFCC
from generator_position_balanced_loss_v1 import position_weights
from contextual_generator_contrast_v1 import contextual_contrast_loss,selected_position_labels
R=Path('research/wakeword_p4/results')
def main():
 torch.set_num_threads(1);torch.manual_seed(726);assert torch.cuda.is_available();s=Sampler();front=ContinuousPastMFCC(torch.load('.local/wakeword/matchbox-grid-aligned-v1/frontend.pt',weights_only=True,map_location='cpu')).cuda().eval();x,p,m,d=batch(s,front);model=CausalGeneratorUnitModel(34).cuda();logits,_=model(x,model.encoder.initial_state(batch=24,device='cuda'));rows=[];losses=[]
 for i,desc in enumerate(d):
  r=s.rows[desc['index']];sel=selected_position_labels(r['intervals'],s.data['dense_to_original_label'].index);rows.append(len(sel));n=desc['frames'];
  for pos,kind,label in sel:
   a,_=position_weights(r['samples'],r['lead'],r['intervals'],**(dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}));frames=np.flatnonzero(a[pos]>0);frames=frames[(frames<n)];
   if len(frames):
    neg=s.data['dense_to_original_label'].index(53+1 if kind=='terminal61' else 56+1)
    losses.append(contextual_contrast_loss(logits[i,:,frames],torch.full((len(frames),),label,device='cuda'),torch.full((len(frames),),neg,device='cuda')))
 loss=torch.stack(losses).mean();loss.backward();g=torch.cat([v.grad.flatten() for v in model.parameters()]);assert torch.isfinite(loss) and torch.isfinite(g).all() and (g!=0).any();out=dict(schema='contextual-generator-contrast-admission-v1',selected_positions=sum(rows),selected_batches=len(losses),loss=float(loss),gradient_l2=float(g.norm()),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('research/wakeword_p4/training/contextual_generator_contrast_v1.py'),INDEX]},optimizer_updates=0,policy='Only declared terminal53->61 positions and class5 55 positions; no blanket token61 penalty. Negative labels are local alternatives, not acoustic truth. One untrained batch, no heldout/threshold/model claim.')
 (R/'4090-contextual-generator-contrast-admission-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
