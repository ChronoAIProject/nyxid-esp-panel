"""Original source-balanced inputs; mark eight lexical negatives for event risk."""
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent,digest
class Sampler(Parent):
    def batch(self):
        (x,p,n),descriptors=super().batch()
        eligible=np.zeros((128,278),bool);peak=np.zeros(128,bool)
        for i,row in enumerate(descriptors):
            if row['kind']!='source_balanced_bare_negative':continue
            assert row['label']==0 and not p[i].any()
            valid=n[i]>0
            eligible[i]=np.logical_and.reduce([valid[j:j+278] for j in range(5)])
            assert eligible[i].any();peak[i]=True
        assert peak.sum()==8
        return (x,p,n,eligible,peak),descriptors
