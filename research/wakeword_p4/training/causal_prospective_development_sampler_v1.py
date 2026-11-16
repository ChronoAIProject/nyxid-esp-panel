"""Full-recording cell sampler, four carriers/six classes per batch24."""
import gzip,json
import numpy as np
from pathlib import Path
INDEX=Path('research/wakeword_p4/results/4090-causal-prospective-development-index-v1.json.gz')
CARRIERS=('bare','light','time','timer')
class Sampler:
 def __init__(self,seed=726):
  with gzip.open(INDEX,'rt') as f:self.data=json.load(f)
  self.rows=self.data['records'];self.cells={c:[g for g in self.data['cells'] if g['carrier']==c] for c in CARRIERS};assert all(len(v)==36 for v in self.cells.values())
  self.rngs={c:np.random.default_rng(seed+100+i) for i,c in enumerate(CARRIERS)};self.permutation=np.random.default_rng(seed+200)
 def descriptors(self):
  result=[]
  for carrier in CARRIERS:
   group=int(self.rngs[carrier].integers(36));cell=self.cells[carrier][group]
   for label,index in enumerate(cell['indices'],1):
    row=self.rows[index];assert row['role']=='training' and row['class_id']==label and row['carrier']==carrier
    result.append(dict(index=index,class_id=label,carrier=carrier,cell=group,samples=row['samples']))
  return [result[int(i)] for i in self.permutation.permutation(24)]
 def state(self):return dict(sources={c:r.bit_generator.state for c,r in self.rngs.items()},permutation=self.permutation.bit_generator.state)
 def restore(self,s):
  for c,r in self.rngs.items():r.bit_generator.state=s['sources'][c]
  self.permutation.bit_generator.state=s['permutation']
