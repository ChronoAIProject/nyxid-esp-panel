"""Paired context intervention retaining bc01 target-source and parent schedules."""
import json
from pathlib import Path
import numpy as np
from causal_chatterbox_batch_v1 import Sampler as Parent
from causal_sequence_supervision_v1 import masks
CACHE=Path('.local/wakeword/chatterbox-context-cache-v1')
class Sampler(Parent):
    def __init__(self,seed=726,condition='speech'):
        super().__init__(seed)
        assert condition in ['speech','silence'];self.condition=condition
        self.context_rows=json.loads((CACHE/'records.json').read_text());assert len(self.context_rows)==168
        self.context_values=np.memmap(CACHE/f'{condition}-features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.context_pools={i:[r for r in self.context_rows if r['source']['index']==i] for i in {r['source']['index'] for r in self.context_rows}}
        assert all(len(pool)==2 for pool in self.context_pools.values())
        self.context_rng=np.random.default_rng(seed+4)
    def batch(self):
        (x,p,n),descriptors=super().batch()
        for i,d in enumerate(descriptors):
            if d['kind']!='chatterbox_training':continue
            pool=self.context_pools[d['index']];row=pool[int(self.context_rng.integers(2))]
            raw=self.context_values[row['offset']:row['offset']+row['frames']]
            _,pw,nw=masks(row['frames'],row['label'],row['lead'],row['endpoint'])
            x[i].fill(0);p[i].fill(0);n[i].fill(0)
            x[i,:row['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            p[i,:len(pw)]=pw;n[i,:len(nw)]=nw
            descriptors[i]=dict(kind='chatterbox_context_training',condition=self.condition,pair=row['pair'],source_index=d['index'],voice=d['voice'],label=row['label'],replaced=d)
        return (x,p,n),descriptors
