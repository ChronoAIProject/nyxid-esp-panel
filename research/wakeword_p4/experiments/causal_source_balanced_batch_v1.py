"""Keep positive coverage draws; add existing source-matched bare nearword negatives."""
import collections
import json
import numpy as np
from causal_positive_coverage_batch_v1 import Sampler as Parent,ROOT,digest
from causal_sequence_supervision_v1 import masks

PHRASES=('Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def exact_key(row):
    return (row['engine'],json.dumps(row['source_pair_key'],sort_keys=True),row['variant'])

class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.lexical_rng=np.random.default_rng(seed+1)
        rows=json.loads((ROOT/'features.json').read_text())
        self.positive_by_offset={r['offset']:r for r in self.extra_positive}
        self.exact=collections.defaultdict(list);self.conditioning=collections.defaultdict(list)
        for r in rows:
            if r['label']==0 and r['text'] in PHRASES and r['source_pair_key']['carrier']=='bare':
                assert r['split']=='training' and r['frames']<=846
                self.exact[(exact_key(r),r['text'])].append(r)
                self.conditioning[(r['engine'],r['conditioning_id'],r['text'])].append(r)
        for r in self.extra_positive:
            for phrase in PHRASES:assert self.conditioning[(r['engine'],r['conditioning_id'],phrase)]
    def batch(self):
        (x,p,n),d=super().batch()
        positives=[r for r in d if r['kind']=='existing_broad_bare_positive']
        slots=[i for i,r in enumerate(d) if r['kind']=='synthetic' and r['label']==0][:8]
        assert len(positives)==len(slots)==8
        for i,pos in zip(slots,positives,strict=True):
            source=self.positive_by_offset[pos['index']]
            phrase=PHRASES[int(self.lexical_rng.integers(4))]
            pool=self.exact.get((exact_key(source),phrase))
            match='exact_source_speed_draw_variant' if pool else 'same_engine_conditioning_fallback'
            if not pool:pool=self.conditioning[(source['engine'],source['conditioning_id'],phrase)]
            row=pool[int(self.lexical_rng.integers(len(pool)))];frames=row['frames']
            raw=self.extra_positive_values[row['offset']:row['offset']+frames]
            _,_,nw=masks(frames,0,None,None)
            x[i].fill(0);p[i].fill(0);n[i].fill(0)
            x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);n[i,:len(nw)]=nw
            d[i]=dict(kind='source_balanced_bare_negative',index=row['offset'],source_id=row['source_id'],engine=row['engine'],conditioning_id=row['conditioning_id'],phrase=phrase,positive_offset=source['offset'],match=match,label=0)
        return (x,p,n),d
