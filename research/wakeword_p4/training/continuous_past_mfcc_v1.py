"""Continuous past-only MFCC proposal; incompatible with frozen window frontend.

Frame k consumes PCM[160*k:160*k+400], timestamp160*k+400.
No reflection, leading pad, endpoint reset, normalization or finish padding.
Final incomplete frame is retained as residual, never silently scored.
"""
import torch
from torch import nn
class ContinuousPastMFCC(nn.Module):
 def __init__(self,state):
  super().__init__()
  for name,shape in [('window',(400,)),('mel',(257,64)),('dct',(64,64))]:
   value=state[name];assert tuple(value.shape)==shape and value.dtype==torch.float32
   self.register_buffer(name,value.detach().clone())
 def forward(self,frames):
  if frames.dtype!=torch.float32 or frames.ndim!=2 or frames.shape[1]!=400:raise ValueError('Expected [frames,400] float32 PCM normalized by32768')
  # Original400sample Hann, symmetric56sample zero extension to512 FFT.
  padded=torch.nn.functional.pad(frames*self.window,(56,56))
  power=torch.fft.rfft(padded,n=512).abs().square()
  return torch.log(power@self.mel+1e-6)@self.dct
class FrameClock:
 """Bounded hostPCM state; arbitrary input chunk boundaries, fixedglobalclock."""
 def __init__(self):
  import numpy as np
  self.pending=np.empty(0,dtype=np.int16);self.received=0;self.next_start=0
 def push(self,pcm):
  import numpy as np
  if pcm.dtype!=np.int16 or pcm.ndim!=1:raise ValueError('monoPCM16 required')
  self.received+=len(pcm);data=np.concatenate((self.pending,pcm));n=max(0,(len(data)-400)//160+1)
  frames=np.lib.stride_tricks.sliding_window_view(data,400)[::160][:n].copy() if n else np.empty((0,400),np.int16)
  ends=self.next_start+np.arange(n,dtype=np.int64)*160+400
  self.next_start+=n*160;self.pending=data[n*160:].copy()
  assert len(self.pending)<400 and self.received-self.next_start==len(self.pending)
  return frames,ends
 def finish(self):
  return dict(received_samples=self.received,next_frame_start_sample=self.next_start,residual_samples=len(self.pending),padding_applied=False)
