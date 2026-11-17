"""Original C161 schedule; independent50% fullwave envelope substitution."""
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent
CACHE=Path('.local/wakeword/c161-envelope-cache-v1')
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        result=json.loads((CACHE/'result.json').read_text());assert result['records']==31644
        rows=[json.loads(x) for x in (CACHE/'records.jsonl').read_text().splitlines()]
        self.envelope_rows={r['original_offset']:r for r in rows};assert len(self.envelope_rows)==31644
        self.envelope_features=np.memmap(CACHE/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        self.envelope_rng=np.random.default_rng(734)
    def batch(self):
        (x,p,n),descriptors=super().batch()
        for i,row in enumerate(descriptors):
            if row['kind'] not in ('existing_broad_bare_positive','source_balanced_bare_negative'):continue
            cached=self.envelope_rows[row['index']];assert cached['label']==row['label']
            apply=bool(self.envelope_rng.integers(2))
            if apply:
                raw=self.envelope_features[cached['offset']:cached['offset']+cached['frames']]
                x[i].fill(0);x[i,:cached['frames']]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            descriptors[i]=dict(row,envelope_applied=apply,envelope_alpha=cached['alpha'] if apply else 1.,envelope_original_offset=row['index'])
        return (x,p,n),descriptors
