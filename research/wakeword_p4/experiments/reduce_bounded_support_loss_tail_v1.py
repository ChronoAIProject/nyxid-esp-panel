"""Read-only original training-support C mean versus nominal probability tails."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sigmoid(x):
 a=np.exp(-np.abs(x));return np.where(x>=0,1/(1+a),a/(1+a))
def main():
 p=R/'4090-bounded-training-support-forward-result-v1.json';d=json.loads(p.read_text());rows=[]
 for c in d['cases']:
  assert sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as z:
   for mode in ['nominal','phase']:
    f=np.clip(z[mode+'_r0_float_logit'],-8,8);q=np.clip((z[mode+'_r0_exact_fc'].astype(np.float32)-3)*np.float32(8/3),-8,8)
    gap=(f-q)/np.float32(8);m=np.abs(gap);loss=np.where(m<=1,np.float32(.5)*gap*gap,m-np.float32(.5)).reshape(10,128)
    probgap=np.abs(sigmoid(f.astype(float))-sigmoid(q.astype(float))).reshape(10,128)
    rows.append(dict(target=c['target'],condition=c['condition'],mode=mode,batches=[dict(batch_index=i,original_uniform_C_mean=float(loss[i].mean()),largest_per_example_C=float(loss[i].max()),nominal_probability_gap_max=float(probgap[i].max()),examples_nominal_gap_above_point1=int(np.sum(probgap[i]>.1))) for i in range(10)]))
 out=dict(source_sha256=sha(p),cases=rows,limits='Old final bounded-family training-support saved arrays only, not new oneway model. Float32 reconstruction of C arithmetic, no autodiff or original reduction-bit identity claim. Nominal sigmoid is not deployed posthead UINT8. Static independent contexts are not contiguous five-step windows. Descriptive association cannot establish dilution as cause; no tail mining, relabeling, reweighting, new model invocation or admission.')
 with (R/'4090-bounded-support-loss-tail-reduction-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 for c in rows:
  if c['condition']=='bounded_logit_consistency':
   b=c['batches'];print(c['target'],c['mode'],'tail_batches',sum(x['examples_nominal_gap_above_point1']>0 for x in b),'largest_Cmean',max(x['original_uniform_C_mean'] for x in b),'max_nominalgap',max(x['nominal_probability_gap_max'] for x in b))
if __name__=='__main__':main()
