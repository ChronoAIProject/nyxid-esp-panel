"""Independent nominal-saturation contribution reduction from saved actual logits."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-final-raw-consistency-saturation-result-v1.json';d=json.loads(p.read_text());plan=R/'4090-final-raw-consistency-saturation-plan-v1.json';assert d['plan_sha256']==sha(plan) and len(d['cases'])==16;seen=set();rows=[];aggregate={}
 for c in d['cases']:
  key=(c['target'],c['condition'],c['batch']);assert key not in seen;seen.add(key);assert sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a:
   for name in ('float_logits','quant_logits','losses'):assert np.array_equal(a[name],a['repeat_'+name]) and np.isfinite(a[name]).all()
   f=a['float_logits'].astype(float).reshape(-1);q=a['quant_logits'].astype(float).reshape(-1);y=a['y'].reshape(-1);assert len(f)==len(q)==len(y)==128;assert set(np.unique(y))<={0,1}
   gap=(f-q)/8;mag=np.abs(gap);terms=np.where(mag<=1,.5*gap*gap,mag-.5);prob=1/(1+np.exp(-np.clip(f,-8,8)));other=1/(1+np.exp(-np.clip(q,-8,8)));diff=np.abs(prob-other)
   positive=(f>=8)&(q>=8);negative=(f<=-8)&(q<=-8);opposed=((f>=8)&(q<=-8))|((f<=-8)&(q>=8));rest=~(positive|negative|opposed);masks=dict(same_positive=positive,same_negative=negative,opposed_saturation=opposed,other=rest);groups={}
   for name,mask in masks.items():
    total=float(terms[mask].sum());groups[name]=dict(count=int(mask.sum()),raw_c_sum=total,fraction_of_c=total/float(terms.sum()) if terms.sum() else 0,nominal_probability_gap_max=float(diff[mask].max()) if mask.any() else None);g=aggregate.setdefault(c['target']+'/'+c['condition'],{}).setdefault(name,dict(count=0,raw_c_sum=0.,nominal_probability_gap_max=0.));g['count']+=int(mask.sum());g['raw_c_sum']+=total;g['nominal_probability_gap_max']=max(g['nominal_probability_gap_max'],float(diff[mask].max()) if mask.any() else 0.)
   assert sum(v['count'] for v in groups.values())==128;assert np.all(diff[positive|negative]==0)
   rows.append(dict(target=c['target'],condition=c['condition'],batch=c['batch'],reported_tf32_s=float(a['losses'][0]),nominal_probability_gap_mean=float(diff.mean()),nominal_probability_gap_max=float(diff.max()),groups=groups,same_side_nominal_saturated_wrong_labels=int(np.sum((positive&(y==0))|(negative&(y==1)))),recomputed_float64_c=float(terms.mean()),original_tf32_c=float(a['losses'][1]),c_reduction_difference=float(terms.mean())-float(a['losses'][1]),initialization_saved_losses_exact=c['initialization_saved_losses_exact']))
 for target,groups in aggregate.items():
  total=sum(g['raw_c_sum'] for g in groups.values())
  for g in groups.values():g['fraction_of_c']=g['raw_c_sum']/total if total else 0
 with (R/'4090-final-raw-consistency-saturation-independent-reduction-v1.json').open('x') as f:json.dump(dict(files={str(v):sha(v) for v in (Path(__file__).resolve().relative_to(Path.cwd()),p,plan)},cases=rows,aggregate=aggregate,optimizer_updates=0,limits='All16 finalmodel probe batches2048samples, newly observed saved logits. Independent float64 Huber/nominalclip-sigmoid reduction. Same-side nominal probabilities equal by definition; does not prove actual full quantized head equality, fullinput finalmodel saturation, downstream performance or sole cause of35failures. No lambda/threshold/checkpoint/model selection, freshdev or acceptance.'),f,indent=2);f.write('\n')
 print(json.dumps(aggregate))
if __name__=='__main__':main()
