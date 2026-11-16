"""Training-only deterministic channel masking; inference graph unchanged."""
import torch
def apply_channel_mask(x,rng,drop_probability=.125):
 if x.ndim!=3 or not 0<=drop_probability<1:raise ValueError('Expected B,C,T')
 mask=torch.rand((x.shape[0],x.shape[1],1),generator=rng,device=x.device)>=drop_probability
 return x*mask,mask
