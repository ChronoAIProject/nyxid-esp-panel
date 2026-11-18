"""Channel-BN mechanical research prototype, not SSN or a BC-ResNet reproduction."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from causal_broadcast_residual_v1 import CausalBroadcastResidual,DILATIONS,CHANNELS

class CausalBroadcastChannelBN(CausalBroadcastResidual):
    def __init__(self):
        super().__init__()
        self.stem_bn=nn.BatchNorm2d(CHANNELS,eps=1e-5,momentum=.1)
        self.spectral_bn=nn.ModuleList(nn.BatchNorm2d(CHANNELS,eps=1e-5,momentum=.1) for _ in DILATIONS)
        self.temporal_bn=nn.ModuleList(nn.BatchNorm1d(CHANNELS,eps=1e-5,momentum=.1) for _ in DILATIONS)

    def forward(self,x,states):
        if x.ndim!=3 or x.shape[1]!=40 or x.shape[2]<3 or x.shape[2]%3 or len(states)!=7:
            raise ValueError('Expected [B,40,3*N] with seven original cache tensors')
        joined=torch.cat((states[0],x),dim=2)
        y=F.relu(self.stem_bn(self.stem(joined.transpose(1,2).unsqueeze(1))))
        updated=[joined[:,:,-2:]]
        for i,(spectral,temporal,pointwise,d) in enumerate(zip(self.spectral,self.temporal,self.pointwise,DILATIONS)):
            s=F.relu(self.spectral_bn[i](spectral(y)))
            joined=torch.cat((states[i+1],s.mean(dim=3)),dim=2)
            updated.append(joined[:,:,-4*d:])
            branch=pointwise(F.relu(self.temporal_bn[i](temporal(joined))))
            y=F.relu(y+s+branch.unsqueeze(3))
        joined=torch.cat((states[-1],y.mean(dim=3)),dim=2)
        updated.append(joined[:,:,-1:])
        return self.head(joined),tuple(updated)

    def folded_eval(self):
        if self.training:raise ValueError('Freeze statistics before folding')
        out=CausalBroadcastResidual().to(self.head.weight.device).eval()
        out.stem=torch.nn.utils.fuse_conv_bn_eval(self.stem,self.stem_bn)
        for i in range(len(DILATIONS)):
            out.spectral[i]=torch.nn.utils.fuse_conv_bn_eval(self.spectral[i],self.spectral_bn[i])
            out.temporal[i]=torch.nn.utils.fuse_conv_bn_eval(self.temporal[i],self.temporal_bn[i])
            out.pointwise[i]=copy.deepcopy(self.pointwise[i])
        out.head=copy.deepcopy(self.head)
        return out
