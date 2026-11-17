"""Preserve all128 original rows and append the fixed eight carrier rows."""
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Original
from causal_paired_carrier_batch_v1 import Sampler as Replacement


class Sampler:
    def __init__(self, seed=726):
        self.original = Original(seed)
        self.replacement = Replacement(seed)

    def batch(self):
        original, descriptors = self.original.batch()
        changed, other = self.replacement.batch()
        slots = [i for i,d in enumerate(other) if d['kind'] in
                 ['paired_carrier_positive', 'paired_carrier_nick_negative']]
        assert len(slots) == 8
        untouched = [i for i in range(128) if i not in slots]
        assert all(descriptors[i] == other[i] for i in untouched)
        assert all(np.array_equal(a[untouched], b[untouched])
                   for a,b in zip(original,changed,strict=True))
        result = tuple(np.concatenate([a,b[slots]],axis=0)
                       for a,b in zip(original,changed,strict=True))
        assert all(np.array_equal(a,b[:128]) for a,b in zip(original,result,strict=True))
        return result, descriptors+[other[i] for i in slots]
