"""Reuse all existing training-only Kokoro carriers with timely causal positives."""
import json
from pathlib import Path
import numpy as np
from causal_existing_nearword_batch_v1 import Sampler as Parent,digest,R
from causal_sequence_supervision_v1 import masks
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        root=Path('.local/wakeword/hey-common-carrier-features-v1')
        self.extra=json.loads((root/'features.json').read_text());assert len(self.extra)==4860 and all(r['split']=='training' and r['label']==0 for r in self.extra)
        self.extra_values=np.memmap(root/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.extra_pools=[[r for r in self.extra if r['conditioning_id']==v] for v in sorted({r['conditioning_id'] for r in self.extra})];assert len(self.extra_pools)==20 and all(len(p)==243 for p in self.extra_pools)
        assert max(r['frames'] for r in self.extra)==844
    def batch(self):
        (x,p,n),desc=super().batch()
        x=np.pad(x,((0,0),(0,96),(0,0)));p=np.pad(p,((0,0),(0,32)));n=np.pad(n,((0,0),(0,32)))
        slots=[i for i,d in enumerate(desc) if d['kind']=='existing_nearword'][:12];assert len(slots)==12
        for i in slots:
            pool=self.extra_pools[int(self.rng.integers(20))];row=pool[int(self.rng.integers(len(pool)))];frames=row['frames'];raw=self.extra_values[row['offset']:row['offset']+frames]
            _,_,nw=masks(frames,0,None,None);x[i].fill(0);p[i].fill(0);n[i].fill(0);x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);n[i,:len(nw)]=nw
            desc[i]=dict(kind='existing_kokoro_carrier',index=row['order'],label=0)
        return (x,p,n),desc
