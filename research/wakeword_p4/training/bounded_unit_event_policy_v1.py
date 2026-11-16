"""Fixed two-frame uncertainty wait; trusted alternatives never skipped."""
from numbers import Integral
from ordered_generator_matcher_v1 import PATTERN,BOUNDARIES
class BoundedUnitEventPolicy:
 def __init__(self,confidence_code=128,max_span_frames=200,refractory_frames=200,max_uncertain_frames=2):
  if not isinstance(confidence_code,Integral) or not 0<=confidence_code<=255:raise ValueError('UINT8 threshold')
  if max_uncertain_frames!=2:raise ValueError('This preregistered diagnostic fixes two frames')
  self.threshold=int(confidence_code);self.max_span=max_span_frames;self.refractory=refractory_frames;self.frame=0;self.last_event=-refractory_frames;self.clear()
 def clear(self):self.recent=[];self.last_token=None;self.uncertain=0
 def push(self,token,confidence_code):
  if not isinstance(confidence_code,Integral) or not 0<=confidence_code<=255:raise ValueError('Actual UINT8 confidence required')
  if token is not None and (not isinstance(token,Integral) or token<0):raise ValueError('Original token or None')
  self.frame+=1
  if self.frame-self.last_event<self.refractory or token is None:self.clear();return False
  if self.recent and self.frame-self.recent[0][1]>self.max_span:self.clear()
  if confidence_code<self.threshold:
   self.uncertain+=1
   if self.uncertain>2:self.clear()
   return False
  self.uncertain=0;token=int(token)
  if token==self.last_token:return False
  self.last_token=token
  if tuple(t for t,_ in self.recent)==PATTERN and token in BOUNDARIES:self.last_event=self.frame;self.clear();return True
  self.recent.append((token,self.frame));self.recent=self.recent[-len(PATTERN):];return False
