"""C161 source schedule unchanged; independent Bernoulli pitch feature substitution."""
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent
CACHE=Path('.local/wakeword/c161-pitch-cache-v2')

class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        result=json.loads((CACHE/'result.json').read_text())
        assert result['records']==31644
        records=[json.loads(x) for x in (CACHE/'records.jsonl').read_text().splitlines()]
        self.pitch_rows={r['original_offset']:r for r in records}
        assert len(self.pitch_rows)==31644
        self.pitch_features=np.memmap(CACHE/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.pitch_rng=np.random.default_rng(seed+2)
    def batch(self):
        (x,p,n),descriptors=super().batch()
        for i,row in enumerate(descriptors):
            if row['kind'] not in ('existing_broad_bare_positive','source_balanced_bare_negative'):
                continue
            original=row['index'];cached=self.pitch_rows[original]
            assert cached['label']==row['label']
            apply=bool(self.pitch_rng.integers(2))
            if apply:
                raw=self.pitch_features[cached['offset']:cached['offset']+cached['frames']]
                x[i].fill(0)
                x[i,:cached['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            descriptors[i]=dict(row,pitch_applied=apply,pitch_cents=cached['cents'] if apply else 0,pitch_original_offset=original)
        return (x,p,n),descriptors
