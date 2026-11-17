"""Two bare-support slots, retain all four original contextual positives."""
import json
from pathlib import Path
import numpy as np
from causal_chatterbox_daily_context_batch_v1 import Sampler as Parent
from causal_sequence_supervision_v1 import masks
CACHE=Path('.local/wakeword/chatterbox-bare-context-cache-v1')
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.bare_rows=json.loads((CACHE/'records.json').read_text());assert len(self.bare_rows)==46
        self.bare_values={c:np.memmap(CACHE/f'{c}-features.bin',dtype='<u2',mode='r').reshape(-1,40) for c in ['speech','silence']}
        self.bare_pools={v:[r for r in self.bare_rows if r['voice']==v] for v in self.chatter_voices};assert all(self.bare_pools.values())
        self.bare_rng=np.random.default_rng(seed+7);self.bare_step=0
    def batch(self):
        (x,p,n),ds=super().batch()
        slots=[i for i,d in enumerate(ds) if d['kind']=='existing_broad_bare_positive'];assert len(slots)==4
        selected=self.bare_rng.choice(slots,size=2,replace=False)
        # Each two successive batches cover all4voices; each4batch cycle covers both histories pervoice.
        for j,i in enumerate(selected):
            v=self.chatter_voices[(2*self.bare_step+j)%4]
            condition='speech' if (j+self.bare_step//2)%2 else 'silence'
            pool=self.bare_pools[v];row=pool[int(self.bare_rng.integers(len(pool)))]
            raw=self.bare_values[condition][row['offset']:row['offset']+row['frames']]
            _,pw,nw=masks(row['frames'],1,row['lead'],row['endpoint'])
            previous=ds[i];x[i].fill(0);p[i].fill(0);n[i].fill(0)
            x[i,:row['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);p[i,:len(pw)]=pw;n[i,:len(nw)]=nw
            ds[i]=dict(kind='chatterbox_bare_support_training',voice=v,label=1,condition=condition,pair=row['pair'],source_index=row['source']['index'],gap_samples=row['gap_samples'],replaced=previous)
        self.bare_step+=1
        return (x,p,n),ds
