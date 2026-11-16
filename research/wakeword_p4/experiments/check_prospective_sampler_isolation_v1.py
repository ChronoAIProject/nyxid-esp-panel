"""Full-budget metadata isolation/pairing/restoration oracle; no audio reads."""
import copy,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_prospective_sampler_v1 import Sampler,INDEX
RESERVED={'af_alloy','af_kore','am_echo','am_santa'}
def main():
 sampler=Sampler();assert len(sampler.rows)==3456
 assert not RESERVED & {r['conditioning'] for r in sampler.rows}
 assert len({r['conditioning'] for r in sampler.rows})==16
 seen=set();classes=Counter();carriers=Counter()
 for step in range(5000):
  before=copy.deepcopy(sampler.state());batch=sampler.descriptors();after=copy.deepcopy(sampler.state())
  if step in [0,249,4999]:
   sampler.restore(before);assert sampler.descriptors()==batch;assert sampler.state()==after
  for carrier in ['bare','light','time','timer']:
   group=[d for d in batch if d['carrier']==carrier];assert sorted(d['class_id'] for d in group)==list(range(1,7))
   rows=[sampler.rows[d['index']] for d in group];assert len({(r['conditioning'],r['speed'],r['variant']) for r in rows})==1
  for d in batch:
   r=sampler.rows[d['index']];assert r['conditioning'] not in RESERVED and r['role']=='training';seen.add(d['index']);classes[str(d['class_id'])]+=1;carriers[d['carrier']]+=1
 files=[Path(__file__),INDEX,Path('research/wakeword_p4/training/causal_prospective_sampler_v1.py')]
 out=dict(schema='prospective-sampler-isolation-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},metadata_batches=5000,draws=120000,unique_records=len(seen),reserved_conditioning=sorted(RESERVED),reserved_draws=0,classes=dict(classes),carriers=dict(carriers),matched_six_class_cells=True,resume_oracles_steps=[0,249,4999],optimizer_updates=0,limitations='Metadata sampler exercise only; actual optimization descriptors require separate comparison. Partition is excluded from current optimization, but previously consumed by earlier experiments; never globally fresh acceptance evidence.')
 with Path('research/wakeword_p4/results/4090-prospective-sampler-isolation-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
