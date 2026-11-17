"""Fixed component ablation; original128 plus exactly four existing rows."""
import numpy as np
from causal_additive_carrier_batch_v1 import Sampler as Joint

class Sampler:
    def __init__(self, seed=726, arm='positive_only'):
        assert arm in ['positive_only','negative_only']
        self.joint=Joint(seed)
        self.kind='paired_carrier_positive' if arm=='positive_only' else 'paired_carrier_nick_negative'
    def batch(self):
        arrays, descriptors=self.joint.batch()
        extra=[i for i in range(128,136) if descriptors[i]['kind']==self.kind]
        assert len(extra)==4
        indices=list(range(128))+extra
        return tuple(a[indices].copy() for a in arrays),[descriptors[i] for i in indices]
