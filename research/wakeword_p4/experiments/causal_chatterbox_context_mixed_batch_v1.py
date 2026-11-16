"""Equal per-label silence/speech history budget, same parent and pair draws."""
import numpy as np
from causal_chatterbox_context_batch_v1 import Sampler as Parent,CACHE
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed,'speech')
        self.silent_values=np.memmap(CACHE/'silence-features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.by_pair={r['pair']:r for r in self.context_rows}
        self.mix_rng=np.random.default_rng(seed+5)
    def batch(self):
        tensors,descriptors=super().batch()
        x,p,n=tensors
        for label in (0,1):
            slots=[i for i,d in enumerate(descriptors) if d['kind']=='chatterbox_context_training' and d['label']==label]
            assert len(slots)==4
            silent=set(self.mix_rng.permutation(slots)[:2].tolist())
            for i in slots:
                d=descriptors[i];condition='silence' if i in silent else 'speech'
                if condition=='silence':
                    row=self.by_pair[d['pair']]
                    raw=self.silent_values[row['offset']:row['offset']+row['frames']]
                    x[i].fill(0);x[i,:row['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
                descriptors[i]={**d,'condition':condition,'mixed_budget':True}
        return tensors,descriptors
