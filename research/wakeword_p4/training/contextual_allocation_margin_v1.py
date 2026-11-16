"""Same contextual margin with normalized declared-position allocation weights."""
import torch

def allocation_margin_loss(logits,target,negative,allocation):
 if logits.ndim!=2 or target.shape!=negative.shape or target.shape!=(logits.shape[1],) or allocation.shape!=target.shape:raise ValueError('Expected C,T and matching T vectors')
 if target.dtype!=torch.long or negative.dtype!=torch.long:raise ValueError('Long labels required')
 if not bool(torch.isfinite(allocation).all()) or bool((allocation<0).any()) or not bool(allocation.sum()>0):raise ValueError('Finite nonnegative nonzero allocation required')
 j=torch.arange(logits.shape[1],device=logits.device)
 margin=torch.relu(.5-(logits[target,j]-logits[negative,j]))
 return (margin*(allocation/allocation.sum())).sum()
