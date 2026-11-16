"""Equal synthetic allocation-position loss; not phone truth or event probability."""
import numpy as np
import torch

def position_weights(samples,lead,intervals,delays=(0,),amplitudes=(1.,)):
 """Each generator position has equal mass, retaining the declared RIR mixture."""
 ends=np.arange(400,samples+1,160,dtype=np.int64);starts=ends-400
 weights=np.abs(np.asarray(amplitudes,dtype=np.float64));weights/=weights.sum()
 allocation=np.zeros((len(intervals),len(ends)),np.float64)
 for delay,weight in zip(delays,weights,strict=True):
  for i,token in enumerate(intervals):
   lo=lead+token['start_sample']+delay;hi=lead+token['end_sample']+delay
   allocation[i]+=weight*np.maximum(0,np.minimum(ends,hi)-np.maximum(starts,lo))/400
 mass=allocation.sum(1);assert (mass>0).all()
 return (allocation/mass[:,None]/len(intervals)).astype(np.float32),ends

def position_balanced_loss(logits,allocation,labels):
 """One full recording, allocation[position,frame], labels[dense position]."""
 if logits.ndim!=2 or allocation.shape!=(len(labels),logits.shape[1]):raise ValueError('Expected C,T and positions,T')
 assert torch.allclose(allocation.sum(),allocation.new_tensor(1.),atol=1e-5)
 return -(torch.log_softmax(logits,dim=0)[labels]*allocation).sum()
