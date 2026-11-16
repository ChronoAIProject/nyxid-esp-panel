"""Exact ordered synthetic-unit reference matcher; not yet an acoustic detector.

OriginalTTSIDs. Collapse adjacent repeats; never remove internalspace/stress.
Require explicit boundary after Nicks to avoid accepting a rightcontinuation.
No fuzzy/edit/beam aliases; no confidence selection implemented here.
"""
PATTERN=(50,156,47,102,16,56,156,102,53,61)
BOUNDARIES=frozenset((0,3,4,6,16))
class OrderedMatcher:
 def __init__(self):self.recent=[];self.last=None;self.frames=0
 def push(self,token):
  if not isinstance(token,int) or token<0:raise ValueError('Original nonnegative integer token required')
  self.frames+=1
  if token==self.last:return False
  self.last=token
  event=tuple(self.recent)==PATTERN and token in BOUNDARIES
  self.recent.append(token);self.recent=self.recent[-len(PATTERN):]
  return event
