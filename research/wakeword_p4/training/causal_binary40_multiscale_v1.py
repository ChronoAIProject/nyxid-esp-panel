"""Fixed single-logit multiscale Binary40; no trained weights."""
import torch
from torch import nn
from torch.nn import functional as F
from causal_binary40_span_v1 import Binary40Span,CONTROL
class Binary40Multiscale(Binary40Span):
 def __init__(self):
  super().__init__(CONTROL)
  self.head=nn.Conv1d(384,1,2)
 def initial_state(self,batch=1,device=None):
  states=super().initial_state(batch,device)
  return (*states[:-1],torch.zeros(batch,384,1,device=device))
 def forward(self,x,states):
  if x.ndim!=3 or x.shape[1]!=40 or x.shape[2]<3 or x.shape[2]%3:raise ValueError('Expected[B,40,3*N]')
  if len(states)!=7:raise ValueError('Expected7states')
  joined=torch.cat((states[0],x),2);y=F.relu(self.stem(joined));updated=[joined[:,:,-2:]];scales=[y]
  for i,(dw,pw,d) in enumerate(zip(self.dw,self.pw,self.dilations)):
   joined=torch.cat((states[i+1],y),2);updated.append(joined[:,:,-4*d:]);y=F.relu(pw(F.relu(dw(joined)))+y);scales.append(y)
  multiscale=torch.cat(scales,1);joined=torch.cat((states[-1],multiscale),2);updated.append(joined[:,:,-1:]);return self.head(joined),tuple(updated)
 def geometry(self):
  return dict(parameters=sum(p.numel() for p in self.parameters()),state_elements=6864,conv_mac_per_call=30528,receptive_field_frames=306,path_support_frames=[6,18,42,90,186,306],right_context_frames=0)
