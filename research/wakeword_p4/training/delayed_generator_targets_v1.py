"""Fixed past-label output delay; no input lookahead or PCM padding."""
import numpy as np
import torch
DELAY_FRAMES=4

def delay_frame_targets(probabilities,mask,delay=DELAY_FRAMES):
 if delay<1 or probabilities.ndim!=3 or mask.shape!=(probabilities.shape[0],probabilities.shape[2]):raise ValueError('Invalid delay/geometry')
 result=torch.zeros_like(probabilities)
 for i in range(len(mask)):
  n=int(mask[i].sum());assert bool(mask[i,:n].all()) and not bool(mask[i,n:].any()) and n>delay
  result[i,0,:delay]=1;result[i,:,delay:n]=probabilities[i,:,:n-delay]
 assert torch.allclose(result.sum(1)[mask],torch.ones_like(result.sum(1)[mask]),atol=1e-5)
 return result

def delay_position_weights(allocation,delay=DELAY_FRAMES):
 if delay<1 or allocation.ndim!=2 or allocation.shape[1]<=delay:raise ValueError('Invalid delay/geometry')
 # Existing recordings have trailing unallocated time; fail if any position cut.
 assert np.all(allocation[:,-delay:]==0),'Delay would drop generator position mass'
 result=np.zeros_like(allocation);result[:,delay:]=allocation[:,:-delay]
 assert np.allclose(result.sum(1),allocation.sum(1),atol=1e-7)
 return result
