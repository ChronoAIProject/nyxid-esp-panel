"""One conditioning/four carriers/six classes, all16groups per16batch cycle."""
import copy,gzip,json
import numpy as np
from causal_prospective_sampler_v1 import INDEX,CARRIERS
class Sampler:
 def __init__(self,seed=726):
  with gzip.open(INDEX,'rt') as f:self.data=json.load(f)
  self.rows=self.data['records'];self.names=sorted({r['conditioning'] for r in self.rows});assert len(self.names)==16
  self.cells={(name,c):[g for g in self.data['cells'] if g['conditioning']==name and g['carrier']==c] for name in self.names for c in CARRIERS}
  assert all(len(v)==9 for v in self.cells.values())
  assert all(r['role']=='training' for r in self.rows)
  self.rng=np.random.default_rng(seed+300);self.order=[];self.cursor=0;self.cycle=-1
 def descriptors(self):
  if self.cursor==len(self.order):self.order=self.rng.permutation(16).tolist();self.cursor=0;self.cycle+=1
  group=self.order[self.cursor];self.cursor+=1;name=self.names[group];result=[]
  for carrier in CARRIERS:
   cell=self.cells[name,carrier][int(self.rng.integers(9))]
   for label,index in enumerate(cell['indices'],1):
    r=self.rows[index];assert r['conditioning']==name and r['class_id']==label and r['carrier']==carrier
    result.append(dict(index=index,class_id=label,carrier=carrier,conditioning=name,group=group,cycle=self.cycle,samples=r['samples']))
  return [result[int(i)] for i in self.rng.permutation(24)]
 def state(self):return copy.deepcopy(dict(rng=self.rng.bit_generator.state,order=self.order,cursor=self.cursor,cycle=self.cycle))
 def restore(self,state):
  s=copy.deepcopy(state);self.rng.bit_generator.state=s['rng'];self.order=s['order'];self.cursor=s['cursor'];self.cycle=s['cycle']
