"""Proposed exact greedy unit event protocol, fixed before model training.

One frame10ms; originalTTS IDs, None=unallocatedtime/untrusted.
Confidence passed as actualUINT8softmaxcode. No float-score rounding.
"""
from numbers import Integral
from ordered_generator_matcher_v1 import PATTERN,BOUNDARIES
class UnitEventPolicy:
 def __init__(self,confidence_code=128,max_span_frames=200,refractory_frames=200):
  if not isinstance(confidence_code,Integral) or not 0<=confidence_code<=255:raise ValueError('UINT8 confidence threshold')
  self.threshold=int(confidence_code);self.max_span=max_span_frames;self.refractory=refractory_frames;self.frame=0;self.last_event=-refractory_frames;self.clear()
 def clear(self):self.recent=[];self.last_token=None
 def push(self,token,confidence_code):
  if not isinstance(confidence_code,Integral) or not 0<=confidence_code<=255:raise ValueError('ActualUINT8 required')
  if token is not None and (not isinstance(token,Integral) or token<0):raise ValueError('OriginaltokenID orNone')
  self.frame+=1
  if self.frame-self.last_event<self.refractory:self.clear();return False
  if token is None or confidence_code<self.threshold:self.clear();return False
  token=int(token)
  if self.recent and self.frame-self.recent[0][1]>self.max_span:self.clear()
  if token==self.last_token:return False
  self.last_token=token
  hit=tuple(t for t,_ in self.recent)==PATTERN and token in BOUNDARIES
  if hit:self.last_event=self.frame;self.clear();return True
  self.recent.append((token,self.frame));self.recent=self.recent[-len(PATTERN):]
  return False
