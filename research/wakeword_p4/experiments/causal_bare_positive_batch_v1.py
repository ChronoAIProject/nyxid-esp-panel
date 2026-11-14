"""Minimal behavioral hypothesis: bare positives, unchanged negative pools."""
from causal_sequence_batch_v1 import Sampler as OriginalSampler,digest
class Sampler(OriginalSampler):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.pools[1]=[[r for r in pool if r['source']['text'] in ('Hey Nicks.','Hey Nicks!')] for pool in self.pools[1]]
        assert all(pool for pool in self.pools[1])
        assert all({r['source']['text'] for r in pool}=={'Hey Nicks.','Hey Nicks!'} for pool in self.pools[1])
