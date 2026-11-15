"""Keep aligned-prefix positives; replace half of synthetic negatives with existing nearwords."""
import json
from pathlib import Path
import numpy as np
from causal_aligned_prefix_batch_v1 import Sampler as PrefixSampler,digest,R
from causal_sequence_supervision_v1 import masks
class Sampler(PrefixSampler):
    def __init__(self,seed=726):
        super().__init__(seed)
        report=json.loads((R/'4090-causal-existing-nearword-features-original-v1.json').read_text())
        self.nearwords=json.loads(Path(report['records']['path']).read_text());assert len(self.nearwords)==288 and all(r['role']=='training' and r['label']==0 for r in self.nearwords)
        self.nearword_values=np.memmap(report['features']['path'],dtype='<u2',mode='r').reshape(-1,40)
        self.nearword_pools=[[r for r in self.nearwords if r['voice']==voice] for voice in sorted({r['voice'] for r in self.nearwords})];assert len(self.nearword_pools)==2 and all(len(p)==144 for p in self.nearword_pools)
    def batch(self):
        (x,p,n),descriptors=super().batch()
        slots=[i for i,d in enumerate(descriptors) if d['kind']=='synthetic' and d['label']==0][:24];assert len(slots)==24
        for i in slots:
            pool=self.nearword_pools[int(self.rng.integers(2))];row=pool[int(self.rng.integers(len(pool)))];frames=row['frames'];offset=row['offset'];raw=self.nearword_values[offset:offset+frames]
            _,pw,nw=masks(int(frames),0,None,None);x[i].fill(0);p[i].fill(0);n[i].fill(0);x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);n[i,:len(nw)]=nw
            descriptors[i]=dict(kind='existing_nearword',index=row['index'],label=0)
        return (x,p,n),descriptors
