"""Untrained causal depthwise encoder prototype with explicit past caches.

No global mean, future context or normalization state. Chunk outputs preserve time.
This is a new architecture, never a functional conversion of frozen Matchbox.
"""
import torch
from torch import nn
CHANNELS=64
DILATIONS=(1,2,4,8,16,32)
class CausalBlock(nn.Module):
 def __init__(self,dilation):
  super().__init__();self.history=4*dilation
  self.dw=nn.Conv1d(64,64,5,dilation=dilation,groups=64,bias=True)
  self.pw=nn.Conv1d(64,64,1,bias=True)
 def forward(self,x,state):
  if state.shape!=(x.shape[0],64,self.history):raise ValueError('Invalid past cache')
  joined=torch.cat((state,x),dim=2)
  y=torch.relu(self.pw(torch.relu(self.dw(joined)))+x)
  return y,joined[:,:,-self.history:]
class CausalCachedEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.blocks=nn.ModuleList(CausalBlock(d) for d in DILATIONS)
 def initial_state(self,batch=1,device=None):
  return tuple(torch.zeros(batch,64,b.history,device=device) for b in self.blocks)
 def forward(self,x,states):
  if x.ndim!=3 or x.shape[1]!=64 or x.shape[2]<1 or len(states)!=len(self.blocks):raise ValueError('Expected [B,64,T] and six past caches')
  updated=[]
  for b,s in zip(self.blocks,states,strict=True):x,s=b(x,s);updated.append(s)
  return x,tuple(updated)
def geometry():
 histories=[4*d for d in DILATIONS]
 return dict(channels=64,kernel=5,dilations=list(DILATIONS),history_frames=histories,receptive_field_frames=1+sum(histories),right_context_frames=0,state_elements=sum(histories)*64,conv_mac_per_frame=len(histories)*(64*5+64*64),parameters=len(histories)*(64*5+64+64*64+64),chunk_frames=3,feature_hop_samples=160)
