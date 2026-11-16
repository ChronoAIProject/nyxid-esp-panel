"""Causal per-frame TTS-unit readout, retained for future integer deployment.

Vocabulary dense mapping is fixed from training-only metadata,0=unallocatedtime.
No CTCblank semantics or acousticphone truth. Decoder/event policy separate.
"""
import torch
from torch import nn
from causal_cached_encoder_v1 import CausalCachedEncoder
class CausalGeneratorUnitModel(nn.Module):
 def __init__(self,classes):
  super().__init__();self.encoder=CausalCachedEncoder();self.readout=nn.Conv1d(64,classes,1)
 def forward(self,x,states):
  encoded,newstates=self.encoder(x,states);return self.readout(encoded),newstates

def soft_target_loss(logits,probabilities):
 if logits.shape!=probabilities.shape or logits.ndim!=3:raise ValueError('Expected matching [B,C,T]')
 return -(probabilities*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
