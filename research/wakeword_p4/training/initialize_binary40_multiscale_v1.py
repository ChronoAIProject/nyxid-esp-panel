"""Fresh function-preserving initialization; never reads trained weights."""
import torch
from causal_binary40_span_v1 import Binary40Span,CONTROL
from causal_binary40_multiscale_v1 import Binary40Multiscale
def fresh_pair(seed=726):
 torch.manual_seed(seed);control=Binary40Span(CONTROL);treatment=Binary40Multiscale()
 with torch.no_grad():
  for name,value in control.state_dict().items():
   if not name.startswith('head.'):treatment.state_dict()[name].copy_(value)
  treatment.head.weight.zero_();treatment.head.weight[:,320:,:].copy_(control.head.weight);treatment.head.bias.copy_(control.head.bias)
 return control,treatment
