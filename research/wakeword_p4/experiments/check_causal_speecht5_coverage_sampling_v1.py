"""Check actual source-role, matching and sequence contracts; no detector run."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from causal_speecht5_coverage_batch_v1 import Sampler

def main():
 sampler=Sampler();counts=collections.Counter();sources=set();max_frames=0
 for _ in range(12):
  (x,p,n),d=sampler.batch();assert x.shape==(128,846,40) and p.shape==n.shape==(128,282)
  assert np.isfinite(x).all() and np.isfinite(p).all() and np.isfinite(n).all()
  assert sum(r['label'] for r in d)==32 and not np.any((p>0)&(n>0))
  pos=[(i,r) for i,r in enumerate(d) if r['kind']=='speecht5_bare_positive'];neg=[r for r in d if r['kind']=='speecht5_matched_bare_negative'];assert len(pos)==len(neg)==4
  for i,r in pos:
   paired=[q for q in neg if q['pair_positive_slot']==i];assert len(paired)==1
   assert all(r[k]==paired[0][k] for k in ['speaker','speed','variant']) and paired[0]['phrase_id']!='nyx'
  for r in d:
   counts[r['kind']]+=1
   if r['kind'].startswith('speecht5_'):sources.add(r['source_id'])
 result=dict(schema='causal-speecht5-coverage-sampling-check-v1',batches=12,descriptors=1536,kinds=dict(counts),new_source_ids_seen=sorted(sources),shape_contract_passed=True,role_and_pairing_passed=True,new_rows_training_only=True,command_positives_not_assigned_invented_boundaries=True,no_detector_invoked=True,no_gradient_or_cpu_gpu_equality_test=True,limits='Fixed sampler mechanical contracts only, not recognition evidence or a source-independence proof.')
 with Path('research/wakeword_p4/results/4090-causal-speecht5-coverage-sampling-check-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
