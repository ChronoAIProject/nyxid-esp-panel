"""Past-only per-frame channel normalization before causal encoder."""
import torch
from torch import nn
from causal_cached_encoder_v1 import CausalCachedEncoder
class CausalFrameNormalizedEncoder(nn.Module):
 def __init__(self,eps=1e-5):
  super().__init__();self.eps=float(eps);self.encoder=CausalCachedEncoder()
 def normalize(self,x):
  if x.ndim!=3 or x.shape[1]!=64:raise ValueError('Expected [B,64,T]')
  mean=x.mean(1,keepdim=True);var=(x-mean).square().mean(1,keepdim=True)
  return (x-mean)/torch.sqrt(var+self.eps)
 def forward(self,x,states):return self.encoder(self.normalize(x),states)
