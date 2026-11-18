"""Original128 unchanged, append four engine-balanced existing training pairs."""
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Original
from causal_sequence_supervision_v1 import masks
B=Path('.local/wakeword')
def read(p):return json.loads(Path(p).read_text())
class Sampler:
 def __init__(self,seed=726):
  assert read('research/wakeword_p4/results/4090-piper-nick-command-feature-review-v1.json')['sampler846_compatible']
  self.original=Original(seed);self.rng=np.random.default_rng(seed+242000)
  rows=read(B/'hey-broad-training-features-v1/hey_nyx/features.json')
  self.positives={e:[r for r in rows if r['label']==1 and r['engine']==e and r['source_pair_key']['carrier']!='bare'] for e in ['piper','kokoro']}
  assert len(self.positives['piper'])==18144 and len(self.positives['kokoro'])==540
  self.voices={e:sorted({r['conditioning_id'] for r in rs}) for e,rs in self.positives.items()}
  self.pools={e:{v:[r for r in rs if r['conditioning_id']==v] for v in self.voices[e]} for e,rs in self.positives.items()}
  assert all(len(pool)==27 for pools in self.pools.values() for pool in pools.values())
  self.negatives={e:{} for e in self.positives}
  for e,path in [('piper',B/'piper-nick-command-features-v1/features.json'),('kokoro',B/'hey-common-carrier-features-v1/features.json')]:
   for r in read(path):
    if not r['text'].startswith('Hey Nick,'):continue
    k=r['source_pair_key'] if e=='piper' else dict(carrier=r['carrier'],scale=r['speed'])
    key=(r['conditioning_id'],float(k['scale']),k['carrier'],r['variant']);assert key not in self.negatives[e];self.negatives[e][key]=r
  self.arrays={e:np.memmap(path,dtype='<u2',mode='r').reshape(-1,40) for e,path in [('positive',B/'hey-broad-training-features-v1/hey_nyx/features.bin'),('piper',B/'piper-nick-command-features-v1/features.bin'),('kokoro',B/'hey-common-carrier-features-v1/features.bin')]}
  for e,rs in self.positives.items():
   for r in rs:
    k=r['source_pair_key'];assert (r['conditioning_id'],float(k['scale']),k['carrier'],r['variant']) in self.negatives[e]
 def batch(self):
  arrays,d=self.original.batch();xs=[];ps=[];ns=[];extra=[]
  for pair in range(4):
   e=['piper','kokoro'][int(self.rng.integers(2))];voices=self.voices[e];v=voices[int(self.rng.integers(len(voices)))];pool=self.pools[e][v];a=pool[int(self.rng.integers(27))];k=a['source_pair_key'];b=self.negatives[e][v,float(k['scale']),k['carrier'],a['variant']]
   for row,label in [(a,1),(b,0)]:
    frames=row['frames'];assert 225<=frames<=846
    raw=self.arrays['positive' if label else e][row['offset']:row['offset']+frames]
    _,p,n=masks(frames,label,(row['lead_frame']+2)*160 if label else None,row['generated_prefix_end_sample'] if label else None)
    x=np.zeros((846,40),np.float32);pw=np.zeros(282,np.float32);nw=pw.copy();x[:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);pw[:len(p)]=p;nw[:len(n)]=n
    xs.append(x);ps.append(pw);ns.append(nw);extra.append(dict(kind='cross_engine_carrier_positive' if label else 'cross_engine_carrier_nick_negative',engine=e,conditioning_id=v,source_id=row.get('source_id',row.get('anchor_id')),index=row['offset'],carrier=k['carrier'],speed=float(k['scale']),variant=a['variant'],label=label,pair=pair))
  return tuple(np.concatenate([old,np.stack(new)],axis=0) for old,new in zip(arrays,[xs,ps,ns],strict=True)),d+extra
