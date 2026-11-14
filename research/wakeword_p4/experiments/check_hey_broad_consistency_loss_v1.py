"""Independent analytic loss and both-logit gradient check, optimizer0."""
import json,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from hey_broad_consistency_loss_v1 import components

def main():
 f=np.array([-1e4,-80,-8,-1,0,1,8,80,1e4,0,8,-8],np.float64)[:,None];q=np.array([1e4,-72,0,7,0,-7,0,72,-1e4,8,0,0],np.float64)[:,None];y=np.array([0,1]*6,np.float64)[:,None];w=np.where(y[:,0]>0,1.,20.);checks=[]
 for coefficient in [0.,1.]:
  a=tf.Variable(f);b=tf.Variable(q)
  with tf.GradientTape() as tape:total,s,c=components(a,b,tf.constant(y),tf.constant(w),coefficient,tf)
  actual=tape.gradient(total,[a,b]);n=len(f);d=(f-q)/8;h=np.where(np.abs(d)<=1,.5*d*d,np.abs(d)-.5).mean();bce=lambda z:np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z)));supervised=np.mean(w[:,None]*(bce(f)+bce(q))*.5)
  sigmoid=lambda z:np.exp(-np.logaddexp(0,-z));dg=np.clip(d,-1,1)/(8*n)
  expected=[w[:,None]*(sigmoid(f)-y)/(2*n)+coefficient*dg,w[:,None]*(sigmoid(q)-y)/(2*n)-coefficient*dg]
  assert np.isclose(float(total),supervised+coefficient*h,rtol=1e-14,atol=1e-14) and np.isclose(float(s),supervised,rtol=1e-14) and np.isclose(float(c),h,rtol=1e-14)
  for x,z in zip(actual,expected,strict=True):assert np.allclose(x.numpy(),z,rtol=1e-13,atol=1e-13)
  assert np.array_equal(a.numpy(),f) and np.array_equal(b.numpy(),q)
  checks.append(dict(coefficient=coefficient,loss=float(total),max_gradient_error=max(float(np.max(np.abs(x.numpy()-z))) for x,z in zip(actual,expected)),variables_unchanged=True))
 out=Path('research/wakeword_p4/results/4090-hey-broad-consistency-loss-check-v1.json')
 with out.open('x') as target:json.dump(dict(checks=checks,optimizer_updates=0,tensorflow=tf.__version__,source_sha256=hashlib.sha256(Path(__file__).with_name('hey_broad_consistency_loss_v1.py').read_bytes()).hexdigest(),limits='Analytic float64 logit fixtures only, no model training or candidate admission.'),target,indent=2);target.write('\n')
 print(checks)
if __name__=='__main__':main()
