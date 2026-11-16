"""Same initialized causal model with parameter-free current-frame normalization."""
import torch
from causal_generator_unit_model_v1 import CausalGeneratorUnitModel
class NormalizedGeneratorUnitModel(CausalGeneratorUnitModel):
 def forward(self,x,states):
  mean=x.mean(1,keepdim=True);centered=x-mean
  normalized=centered*torch.rsqrt(centered.square().mean(1,keepdim=True)+1e-5)
  return super().forward(normalized,states)
