"""Fixed four positive/four Nick carrier replacements; no new source generation."""
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent,digest
from causal_sequence_supervision_v1 import masks
class Sampler(Parent):
 def __init__(self,seed=726):
  super().__init__(seed);self.carrier_rng=np.random.default_rng(seed+1701)
  rows=json.loads(Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json').read_text());self.carrier_positives=[r for r in rows if r['label']==1 and r['engine']=='kokoro' and r['source_pair_key']['carrier']!='bare'];assert len(self.carrier_positives)==540
  negative=json.loads(Path('.local/wakeword/hey-common-carrier-features-v1/features.json').read_text());self.carrier_negatives={(r['conditioning_id'],float(r['speed']),r['carrier'],r['variant']):r for r in negative if r['text'].startswith('Hey Nick,')};assert len(self.carrier_negatives)==540
  self.carrier_voices=sorted({r['conditioning_id'] for r in self.carrier_positives});assert len(self.carrier_voices)==20
  self.carrier_pools={v:[r for r in self.carrier_positives if r['conditioning_id']==v] for v in self.carrier_voices};assert all(len(v)==27 for v in self.carrier_pools.values())
 def batch(self):
  (x,p,n),d=super().batch();ps=[i for i,v in enumerate(d) if v['kind']=='existing_broad_bare_positive'][:4];ns=[i for i,v in enumerate(d) if v['kind']=='source_balanced_bare_negative'][:4];assert len(ps)==len(ns)==4
  for i,j in zip(ps,ns,strict=True):
   voice=self.carrier_voices[int(self.carrier_rng.integers(20))];pool=self.carrier_pools[voice];a=pool[int(self.carrier_rng.integers(27))];k=a['source_pair_key'];b=self.carrier_negatives[voice,float(k['scale']),k['carrier'],a['variant']]
   for slot,row,label in [(i,a,1),(j,b,0)]:
    frames=row['frames'];raw=(self.extra_positive_values if label else self.extra_values)[row['offset']:row['offset']+frames];assert frames<=846
    lead=(row['lead_frame']+2)*160 if label else None;end=row['generated_prefix_end_sample'] if label else None
    _,pw,nw=masks(int(frames),label,int(lead) if label else None,int(end) if label else None)
    x[slot].fill(0);p[slot].fill(0);n[slot].fill(0);x[slot,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);p[slot,:len(pw)]=pw;n[slot,:len(nw)]=nw
    d[slot]=dict(kind='paired_carrier_positive' if label else 'paired_carrier_nick_negative',index=row['offset'],source_id=row.get('source_id',row.get('anchor_id')),conditioning_id=voice,carrier=k['carrier'],speed=float(k['scale']),variant=a['variant'],positive_offset=a['offset'],negative_offset=b['offset'],label=label)
  return (x,p,n),d
