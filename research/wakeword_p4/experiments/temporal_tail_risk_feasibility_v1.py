"""Enumerate all training ID-pair/speed legal intervals without score access."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
path=Path('.local/wakeword/factorized-training-features-v1/report.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(path)=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
rows=json.loads(path.read_text())['records'];groups=collections.defaultdict(list)
assert len(rows)==40320
for row in rows:
 assert row['split']=='training' and row['mechanical_pass'] and row['entire_signal_fits']
 groups[int(row['voice']),row['length_multiplier']].append(row)
voices=sorted({key[0] for key in groups});assert len(voices)==672
reports=[]
for speed in (.9,1.,1.1):
 bounds=[]
 for voice in voices:
  records=groups[voice,speed];assert len(records)==20
  assert {(r['target'],r['arm']) for r in records}=={(p,a) for p in ('hey_nyx','hey_nick','hey_next','hey_mix') for a in ('quiet','noise25','combined','noise15','rir')}
  low=max(r['feature_crop_end'] for r in records)+12
  high=min(350,*(r['raw_shape'][0]-1 for r in records))
  for r in records:
   assert (low-12+1)*160>=32000+r['signal_samples']
   assert (high-150)*160<=32000
  bounds.append((low,high))
 b=np.asarray(bounds);lower=np.maximum(b[:,0,None],b[None,:,0]);upper=np.minimum(b[:,1,None],b[None,:,1])
 counts=np.maximum(0,upper//3-(lower+2)//3+1)
 i,j=np.triu_indices(len(voices),1);n=counts[i,j]
 reports.append({'speed':speed,'distinct_id_pairs':len(i),'infeasible_pairs':int(np.count_nonzero(n==0)),
  'min_legal_endpoints':int(n.min()),'max_legal_endpoints':int(n.max()),'max_lower_bound':int(b[:,0].max())})
out={'report_sha256':sha(path),'records':len(rows),'pair_speed_cases':sum(r['distinct_id_pairs'] for r in reports),
 'reports':reports,'all_feasible':all(r['infeasible_pairs']==0 for r in reports),'script_sha256':sha(Path(__file__)),
 'limits':'Full metadata interval/support feasibility only; raw byte/window authentication, independent RNG replay and actual-network gradients still required. No score access or source exclusions.'}
p=Path('research/wakeword_p4/results/4090-temporal-tail-risk-feasibility-v1.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
