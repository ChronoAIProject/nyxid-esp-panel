"""Cycle-delayed training-only exponentiated group weights; fixed eta0.1."""
import copy
import numpy as np
class GroupRisk:
 def __init__(self):
  self.log_weights=np.zeros(16,np.float64);self.losses={}
 def multiplier(self,group):
  assert 0<=group<16
  x=np.exp(self.log_weights-self.log_weights.max());return float(16*x[group]/x.sum())
 def observe(self,group,unweighted_loss):
  assert 0<=group<16 and group not in self.losses
  assert np.isfinite(unweighted_loss) and unweighted_loss>=0
  self.losses[group]=float(unweighted_loss)
  if len(self.losses)==16:
   self.log_weights+=.1*np.array([self.losses[g] for g in range(16)])
   self.log_weights-=self.log_weights.max();self.losses={}
 def state(self):return dict(log_weights=self.log_weights.tolist(),losses=copy.deepcopy(self.losses))
 def restore(self,s):
  self.log_weights=np.asarray(s['log_weights'],dtype=np.float64).copy();assert self.log_weights.shape==(16,) and np.isfinite(self.log_weights).all()
  self.losses={int(k):float(v) for k,v in s['losses'].items()}
