"""Add existing training-only complete bare positives; keep timely commands and negatives."""
import json
from pathlib import Path
import numpy as np
from causal_multivoice_nearword_batch_v1 import Sampler as Parent,digest,R
from causal_sequence_supervision_v1 import masks
ROOT=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx')
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        rows=json.loads((ROOT/'features.json').read_text());self.extra_positive=[r for r in rows if r['label']==1 and r['source_pair_key']['carrier']=='bare'];assert len(self.extra_positive)==6444 and all(r['split']=='training' and r['target_id']=='hey_nyx' for r in self.extra_positive)
        self.extra_positive_values=np.memmap(ROOT/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.positive_pools=[]
        for engine in ['piper','kokoro','qwen']:
            selected=[r for r in self.extra_positive if r['engine']==engine]
            pools=[[r for r in selected if r['conditioning_id']==voice] for voice in sorted({r['conditioning_id'] for r in selected})]
            self.positive_pools.append(pools)
        assert [len(p) for p in self.positive_pools]==[672,20,24]
    def batch(self):
        (x,p,n),d=super().batch();slots=[i for i,r in enumerate(d) if r['label']==1][:8];assert len(slots)==8
        for i in slots:
            engine=int(self.rng.integers(3));voices=self.positive_pools[engine];pool=voices[int(self.rng.integers(len(voices)))];row=pool[int(self.rng.integers(len(pool)))];frames=row['frames'];raw=self.extra_positive_values[row['offset']:row['offset']+frames]
            lead=(row['lead_frame']+2)*160;end=row['generated_container_end_sample'];assert row['text']=='Hey Nicks.' and frames<=846 and lead<end<=row['samples']
            _,pw,nw=masks(frames,1,int(lead),int(end));x[i].fill(0);p[i].fill(0);n[i].fill(0);x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);p[i,:len(pw)]=pw;n[i,:len(nw)]=nw
            d[i]=dict(kind='existing_broad_bare_positive',index=int(row['offset']),source_id=row['source_id'],variant=row['variant'],label=1)
        return (x,p,n),d
