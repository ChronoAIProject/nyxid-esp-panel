"""Causal local-frequency stem prototype; original temporal stack unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from causal_binary40_span_v1 import Binary40Span, CONTROL

class LocalStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=nn.Conv2d(1,8,(3,5),stride=(3,2),padding=(0,2))
        self.project=nn.Conv1d(160,64,1)
    def forward(self,x):
        # x B,F,T; Conv2d axes are time, ordered frontend channels.
        y=F.relu(self.local(x.transpose(1,2).unsqueeze(1)))
        y=y.permute(0,1,3,2).reshape(x.shape[0],160,-1)
        return self.project(y)

class Binary40LocalStem(Binary40Span):
    def __init__(self):
        super().__init__(CONTROL)
        self.stem=LocalStem()
    def geometry(self):
        g=super().geometry()
        g.update(parameters=sum(p.numel() for p in self.parameters()),stem='8 local maps, frequency kernel5/stride2, flatten160 to64 projection',conv_mac_per_call=20*8*3*5+160*64+5*(64*5+64*64)+64*2)
        return g
