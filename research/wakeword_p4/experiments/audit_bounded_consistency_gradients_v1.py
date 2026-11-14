"""Saved-gradient finite/repeat/canonical-merge and independent nominal loss audit."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-bounded-consistency-gradients-result-v1.json';result=json.loads(p.read_text());plan=R/'4090-bounded-consistency-gradients-plan-v1.json';assert result['plan_sha256']==sha(plan) and len(result['cases'])==8
 lock=json.loads(plan.read_text())
 for path,h in lock['files'].items():assert sha(path)==h,path
 seen=set();reports=[]
 for c in result['cases']:
  key=(c['target'],c['batch']);assert key not in seen;seen.add(key);assert sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a:
   keys=[k for k in a.files if 'repeat_'+k in a.files];assert all(np.array_equal(a[k],a['repeat_'+k]) and np.isfinite(a[k]).all() for k in keys)
   for n in range(15):
    s=a[f'supervised_{n}'];d=a[f'consistency_{n}'];assert np.array_equal(a[f'merged0_{n}'],s) and np.array_equal(a[f'merged1_{n}'],s+d)
   sn=float(np.sqrt(sum(np.sum(a[f'supervised_{n}'].astype(float)**2) for n in range(15))));cn=float(np.sqrt(sum(np.sum(a[f'consistency_{n}'].astype(float)**2) for n in range(15))));assert sn==c['supervised_gradient_l2'] and cn==c['consistency_gradient_l2']
   f=a['float_logits'].astype(float);q=a['quant_logits'].astype(float);bf=a['float_bounded'].astype(float);bq=a['quant_bounded'].astype(float);assert np.array_equal(bf,np.clip(f,-8,8))
   # Reproduce explicitly declared float32 nominal FC coordinate transform.
   actual_bq=np.clip((a['observed_scalar_fc_exact'].astype(np.float32)-np.float32(3))*np.float32(8/3),-8,8);assert np.array_equal(bq,actual_bq)
   gap=(bf-bq)/8;mag=np.abs(gap);terms=np.where(mag<=1,.5*gap*gap,mag-.5);independent_c=float(terms.mean());y=a['y'].astype(float);w=a['w'].astype(float).reshape(-1,1);independent_s=float(np.mean(w*(np.logaddexp(0,f)-y*f+np.logaddexp(0,q)-y*q)/2));saturated=((bf==8)&(bq==8))|((bf==-8)&(bq==-8));assert np.all(terms[saturated]==0)
   reports.append(dict(target=c['target'],batch=c['batch'],samples=len(f),repeat_arrays=len(keys),gradient_tensors=15,supervised_gradient_l2=sn,consistency_gradient_l2=cn,zero_consistency_gradient_tensors=sum(not np.any(a[f'consistency_{n}']) for n in range(15)),same_side_nominal_saturated=int(saturated.sum()),float64_s=independent_s,float64_c=independent_c,reported_tf32_s=float(a['losses'][0]),reported_tf32_c=float(a['losses'][1]),s_reduction_difference=independent_s-float(a['losses'][0]),c_reduction_difference=independent_c-float(a['losses'][1]),canonical_merge_both_coefficients_exact=True))
 assert seen=={(target,batch) for target in ('hey_willow','hey_nyx') for batch in range(1,5)}
 with (R/'4090-bounded-consistency-gradients-independent-audit-v1.json').open('x') as f:json.dump(dict(result_sha256=sha(p),cases=reports,optimizer_updates=0,limits='Independent saved hash/finite/repeat/componentnorms/explicitmerge0and1/nominalcoordinates and float64loss decomposition. No independent network autodiff or variable-unchanged proof, fullquantized posthead, usefulness/training/recognition/admission claim. Reduction differences descriptive, no new candidate tolerance or coefficient tuning.'),f,indent=2);f.write('\n')
 print([(c['target'],c['batch'],c['reported_tf32_c'],c['consistency_gradient_l2'],c['same_side_nominal_saturated']) for c in reports])
if __name__=='__main__':main()
