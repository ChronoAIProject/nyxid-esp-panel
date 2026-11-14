"""Mechanically valid masks for whole-wave causal startup training."""
import numpy as np

STRIDE=3
HOP=160

def masks(frames,label,lead_samples,container_end_samples):
    assert type(frames) is int and frames>0 and label in (0,1)
    calls=(frames+STRIDE-1)//STRIDE
    support=(STRIDE*np.arange(1,calls+1,dtype=np.int64)+2)*HOP
    if label:
        assert type(lead_samples) is int and type(container_end_samples) is int
        assert 0<lead_samples<container_end_samples
        negative=support<=lead_samples
        positive=(support>=container_end_samples+3*HOP)&(support<=container_end_samples+27*HOP)
        assert negative.any() and positive.any() and not np.any(negative&positive)
    else:
        negative=np.ones(calls,dtype=bool);positive=np.zeros(calls,dtype=bool)
    # Mean each valid presentation/region independently; no duration weighting.
    weights_negative=negative.astype(np.float32)/np.float32(negative.sum())
    weights_positive=positive.astype(np.float32)/np.float32(max(1,positive.sum()))
    return support,weights_positive,weights_negative
