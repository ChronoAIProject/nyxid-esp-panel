"""Replace only four mixed-context negatives, same voice/gap/context budget."""
import json
from pathlib import Path
import numpy as np
from causal_chatterbox_context_mixed_batch_v1 import Sampler as Parent
from causal_sequence_supervision_v1 import masks
CACHE=Path('.local/wakeword/chatterbox-daily-context-cache-v1')
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.daily_rows=json.loads((CACHE/'records.json').read_text());assert len(self.daily_rows)==342
        self.daily_values={c:np.memmap(CACHE/f'{c}-features.bin',dtype='<u2',mode='r').reshape(-1,40) for c in ['speech','silence']}
        self.daily_groups=sorted({r['source']['request']['group'] for r in self.daily_rows});assert len(self.daily_groups)==6
        self.daily_pools={(v,g,gap):[r for r in self.daily_rows if r['voice']==v and r['source']['request']['group']==g and r['gap_samples']==gap] for v in self.chatter_voices for g in self.daily_groups for gap in [8000,32000]}
        assert all(self.daily_pools.values())
        self.daily_rng=np.random.default_rng(seed+6)
    def batch(self):
        (x,p,n),ds=super().batch()
        for i,d in enumerate(ds):
            if d['kind']!='chatterbox_context_training' or d['label']!=0:continue
            gap=self.by_pair[d['pair']]['gap_samples'];group=self.daily_groups[int(self.daily_rng.integers(6))]
            pool=self.daily_pools[d['voice'],group,gap];row=pool[int(self.daily_rng.integers(len(pool)))]
            raw=self.daily_values[d['condition']][row['offset']:row['offset']+row['frames']]
            _,pw,nw=masks(row['frames'],0,None,None)
            x[i].fill(0);p[i].fill(0);n[i].fill(0)
            x[i,:row['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            p[i,:len(pw)]=pw;n[i,:len(nw)]=nw
            ds[i]=dict(kind='chatterbox_daily_context_training',voice=d['voice'],label=0,condition=d['condition'],gap_samples=gap,group=group,pair=row['pair'],source_index=row['source']['index'],replaced=d)
        return (x,p,n),ds
