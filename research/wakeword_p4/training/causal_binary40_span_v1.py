"""Matched-width shorter neural history hypothesis; no trained weights."""
import torch
from torch import nn
from torch.nn import functional as F
CONTROL=(1,2,4,8,10)
TREATMENT=(1,2,4,5,7)
class Binary40Span(nn.Module):
 def __init__(self,dilations=TREATMENT):
  super().__init__();self.dilations=tuple(dilations)
  if self.dilations not in [CONTROL,TREATMENT]:raise ValueError('Only the fixed matched control/treatment are supported')
  self.stem=nn.Conv1d(40,64,3,stride=3);self.dw=nn.ModuleList(nn.Conv1d(64,64,5,dilation=d,groups=64) for d in self.dilations);self.pw=nn.ModuleList(nn.Conv1d(64,64,1) for _ in self.dilations);self.head=nn.Conv1d(64,1,2)
 def initial_state(self,batch=1,device=None):return (torch.zeros(batch,40,2,device=device),*(torch.zeros(batch,64,4*d,device=device) for d in self.dilations),torch.zeros(batch,64,1,device=device))
 def forward(self,x,states):
  if x.ndim!=3 or x.shape[1]!=40 or x.shape[2]<3 or x.shape[2]%3:raise ValueError('Expected[B,40,3*N]')
  if len(states)!=7:raise ValueError('Expected seven independently retained states')
  joined=torch.cat((states[0],x),dim=2);y=F.relu(self.stem(joined));updated=[joined[:,:,-2:]]
  for i,(dw,pw,d) in enumerate(zip(self.dw,self.pw,self.dilations)):
   joined=torch.cat((states[i+1],y),dim=2);updated.append(joined[:,:,-4*d:]);y=F.relu(pw(F.relu(dw(joined)))+y)
  joined=torch.cat((states[-1],y),dim=2);updated.append(joined[:,:,-1:]);return self.head(joined),tuple(updated)
 def geometry(self):return dict(dilations=list(self.dilations),parameters=sum(p.numel() for p in self.parameters()),receptive_field_frames=3+3*(sum(4*d for d in self.dilations)+1),state_elements=80+64*(sum(4*d for d in self.dilations)+1),conv_mac_per_call=40*64*3+5*(64*5+64*64)+64*2,right_context_frames=0)
