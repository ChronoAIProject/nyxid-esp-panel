"""Add four new-source bare positive/near-word pairs, retain parent supervision."""
import collections
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent,digest
from causal_sequence_supervision_v1 import masks
ROOT=Path('.local/wakeword/speecht5-training-features-v1')
PHRASES=('official','nick','next','mix','muse')
class Sampler(Parent):
 def __init__(self,seed=726):
  super().__init__(seed);self.new_rng=np.random.default_rng(seed+3)
  all_rows=json.loads((ROOT/'features.json').read_text());assert len(all_rows)==1512 and all(r['role']=='training' and r['passed'] for r in all_rows)
  rows=[r for r in all_rows if r['eligible_first_experiment']];assert len(rows)==378 and all(r['carrier']=='bare' for r in rows)
  self.new_rows={(r['speaker'],r['speed'],r['variant'],r['phrase_id']):r for r in rows};assert len(self.new_rows)==378
  self.new_speakers=sorted({r['speaker'] for r in rows});assert len(self.new_speakers)==7
  self.new_values=np.memmap(ROOT/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
 def batch(self):
  (x,p,n),descriptors=super().batch()
  positives=[i for i,d in enumerate(descriptors) if d['kind']=='existing_broad_bare_positive'];assert len(positives)==8
  negative_slots=collections.defaultdict(collections.deque)
  for i,d in enumerate(descriptors):
   if d['kind']=='source_balanced_bare_negative':negative_slots[d['positive_offset']].append(i)
  pairs=[(i,negative_slots[descriptors[i]['index']].popleft()) for i in positives]
  for i,j in pairs[:4]:
   speaker=self.new_speakers[int(self.new_rng.integers(7))];speed=['slow','native','fast'][int(self.new_rng.integers(3))];variant=int(self.new_rng.integers(3));phrase=PHRASES[int(self.new_rng.integers(5))]
   for slot,phrase_id,label in [(i,'nyx',1),(j,phrase,0)]:
    r=self.new_rows[(speaker,speed,variant,phrase_id)];frames=r['frames'];assert frames<=846 and r['label']==label
    raw=self.new_values[r['offset']:r['offset']+frames]
    _,pw,nw=masks(frames,label,int(r['lead_samples']) if label else None,int(r['generated_container_end_sample']) if label else None)
    x[slot].fill(0);p[slot].fill(0);n[slot].fill(0);x[slot,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);p[slot,:len(pw)]=pw;n[slot,:len(nw)]=nw
    descriptors[slot]=dict(kind='speecht5_bare_positive' if label else 'speecht5_matched_bare_negative',index=r['offset'],source_id=r['source_request_id'],speaker=speaker,speed=speed,variant=variant,phrase_id=phrase_id,label=label,pair_positive_slot=i,lead_samples=r['lead_samples'],container_end_samples=r['generated_container_end_sample'])
  return (x,p,n),descriptors
