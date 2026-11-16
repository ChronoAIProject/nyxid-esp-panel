"""Context-conditioned local contrast objective for declared generator positions."""
import torch

def contextual_contrast_loss(logits,target,negative,weight=1.):
 if logits.ndim!=2 or target.shape!=negative.shape or target.shape!=(logits.shape[1],):raise ValueError('Expected logits C,T and labels T')
 # target/negative are dense class indices at selected frames; no blanket token penalty.
 if target.dtype!=torch.long or negative.dtype!=torch.long:raise ValueError('Long labels')
 margin=weight*torch.relu(0.5-(logits[target,torch.arange(logits.shape[1],device=logits.device)]-logits[negative,torch.arange(logits.shape[1],device=logits.device)])).mean()
 return margin

def selected_position_labels(intervals, dense_index):
 # Only terminal 61 after preceding53 and Mix distinction 55; caller supplies class/context.
 out=[]
 for i,t in enumerate(intervals):
  if t['token_id']==61 and i and intervals[i-1]['token_id']==53:out.append((i,'terminal61'))
  if t['token_id']==55:out.append((i,'distinction55'))
 return [(i,k,dense_index(t['token_id']+1)) for i,k in out for t in [intervals[i]]]
