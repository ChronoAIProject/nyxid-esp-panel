"""Training-only soft-allocation support versus hard contextual margin targets."""
import gzip,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_overlap_targets_v1 import targets
from generator_position_balanced_loss_v1 import position_weights
from contextual_generator_contrast_v1 import selected_position_labels
from causal_prospective_sampler_v1 import INDEX

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with gzip.open(INDEX,'rt') as f:data=json.load(f)
 mapping=data['dense_to_original_label'];groups=defaultdict(lambda:dict(positions=0,selected_frames=0,other_argmax_frames=0,alternative_greater_frames=0,target_probability_sum=0.,allocation_weighted_target_probability_sum=0.))
 for r in data['records']:
  assert r['role']=='training';kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {}
  y,ends,_=targets(r['samples'],r['lead'],r['intervals'],**kwargs);a,_=position_weights(r['samples'],r['lead'],r['intervals'],**kwargs)
  for pos,kind,dense in selected_position_labels(r['intervals'],mapping.index):
   token=mapping[dense];neg=54 if kind=='terminal61' else 57;mask=a[pos]>0;v=y[mask,token];alternative=y[mask,neg] if neg<y.shape[1] else np.zeros(mask.sum());g=groups[f"class{r['class_id']}:{kind}:rir{bool(r['rir_tail'])}"]
   g['positions']+=1;g['selected_frames']+=int(mask.sum());g['other_argmax_frames']+=int((y[mask].argmax(1)!=token).sum());g['alternative_greater_frames']+=int((alternative>v).sum());g['target_probability_sum']+=float(v.sum());g['allocation_weighted_target_probability_sum']+=float(np.average(v,weights=a[pos,mask]))
 for g in groups.values():
  g['mean_target_allocation_on_uniform_margin_frames']=g['target_probability_sum']/g['selected_frames'];g['mean_position_weighted_target_allocation']=g['allocation_weighted_target_probability_sum']/g['positions']
 paths=[INDEX,Path(__file__),*[Path('research/wakeword_p4/training')/n for n in ['generator_overlap_targets_v1.py','generator_position_balanced_loss_v1.py','contextual_generator_contrast_v1.py']]]
 out=dict(schema='contextual-contrast-support-audit-v1',inputs={str(p):sha(p) for p in paths},groups=dict(groups),limits='Metadata support compatibility only; allocation is synthetic reference, not acoustic posterior. Hard margin may intentionally override mixed CE; counts do not prove training failure causality. No PCM/model/dev/final calls.')
 p=Path('research/wakeword_p4/results/4090-contextual-contrast-support-audit-v1.json')
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out['groups'],indent=2))
if __name__=='__main__':main()
