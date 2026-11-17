"""Training-only saved-batch temporal mask audit, without inference/optimization."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=R/'4090-binary40-training-fit-original-v1.json';r=json.loads(report.read_text());trace=Path(r['trace']['path']);assert sha(trace)==r['trace']['sha256']
files={str(report):sha(report),str(trace):sha(trace),__file__:sha(__file__)}
for name in ['causal_sequence_supervision_v1.py','causal_source_balanced_batch_v1.py','causal_aligned_prefix_batch_v1.py','causal_fullsequence_graph_v1.py']:
 p=Path(__file__).with_name(name);files[str(p)]=sha(p)
rows=[]
with np.load(trace) as z:
 p=z['positive_weights'];n=z['negative_weights'];f=z['float_probability'];codes=z['uint8'];assert len(p)==len(r['records'])==128
 for i,rec in enumerate(r['records']):
  pos=np.flatnonzero(p[i]>0)
  if not len(pos):continue
  runs=int(np.count_nonzero(np.diff(np.r_[False,p[i]>0,False].astype(int))==1))
  after=np.arange(len(p[i]))>pos[-1];unweighted=after&(p[i]==0)&(n[i]==0)
  assert runs==1 and not np.any(n[i,after])
  rows.append(dict(index=i,source=rec['source'],positive_runs=runs,positive_calls=len(pos),first_positive_call=int(pos[0]+1),last_positive_call=int(pos[-1]+1),post_positive_tensor_calls=int(after.sum()),post_positive_unweighted_calls=int(unweighted.sum()),post_positive_float_above_threshold=int(np.count_nonzero(f[i,after]>=.91796875)),post_positive_int8_above_threshold=int(np.count_nonzero(codes[i,after]>=235)),limits='Post-mask tensor includes padding; not an acoustic-tail or decoder-rearm measurement.'))
out=dict(schema='binary40-temporal-mask-audit-v1',files=files,rows=rows,summary=dict(batch_rows=128,positive_rows=len(rows),single_positive_region_rows=sum(x['positive_runs']==1 for x in rows),rows_with_no_post_positive_negative_supervision=len(rows),positive_calls=sorted(set(x['positive_calls'] for x in rows))),limits='Original Sampler726 first saved training batch only, 16 rows overlap export calibration. No inference, optimization, evaluation access, phonetic endpoint truth or causal proof. Supervision after positive interval is absent by implementation, but this does not establish that persistent scores cause continuous misses; decoder state needs separate evidence. Prior prefix alignment and temporal-risk experiments already exist and are not new interventions.')
(R/'4090-binary40-temporal-mask-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']))
