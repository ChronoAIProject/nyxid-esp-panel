"""Full planned paired schedule and resume witness, metadata only."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_conditioning_balanced_sampler_v1 import Sampler
from causal_prospective_sampler_v1 import INDEX

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=Sampler();b=Sampler();counts=Counter();cycles={};h=hashlib.sha256()
 for step in range(1,5001):
  x=a.descriptors();assert x==b.descriptors();group=x[0]['conditioning'];assert all(d['conditioning']==group for d in x)
  assert Counter((d['carrier'],d['class_id']) for d in x)==Counter({(c,k):1 for c in ['bare','light','time','timer'] for k in range(1,7)})
  cycles.setdefault(x[0]['cycle'],[]).append(group);counts[group]+=1;h.update((json.dumps(x,sort_keys=True)+'\n').encode())
  if step in [1,15,16,17,249,250,4999]:
   restored=Sampler(999);restored.restore(a.state());probe=Sampler();probe.restore(a.state())
   for _ in range(33):assert restored.descriptors()==probe.descriptors()
 complete=[g for g in cycles.values() if len(g)==16];assert len(complete)==312 and all(len(set(g))==16 for g in complete)
 assert max(counts.values())-min(counts.values())==1
 p=Path('research/wakeword_p4/results/4090-conditioning-balanced-sampler-check-v1.json');out=dict(schema='conditioning-balanced-sampler-check-v1',steps=5000,paired_schedule_sha256=h.hexdigest(),full_cycles=312,partial_final_cycle=8,group_batch_counts=dict(counts),inputs={str(INDEX):sha(INDEX),'sampler':sha('research/wakeword_p4/training/causal_conditioning_balanced_sampler_v1.py'),'script':sha(__file__)},all_passed=True,limits='Metadata-only paired schedule/resume audit. No model, PCM, optimizer, development or acceptance calls. Uneven last8groups retained identically in both arms.')
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
