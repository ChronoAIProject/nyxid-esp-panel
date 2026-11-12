"""Independent value/gradient checks, no dataset access or network updates."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from temporal_tail_risk import temporal_tail_risk
rng=np.random.default_rng(726);labels=np.array([0.,1.]);weights=np.array([20.,1.])
cases=[np.zeros((2,5)),rng.normal(size=(2,5)),np.array([[1000.,-1000.,20.,-20.,0.],[-1000.,1000.,-20.,20.,0.]]),np.full((2,5),20.)]
errors=[]
def reference(z):
 a=(1-2*labels[:,None])*z;b=np.logaddexp(0,a);m=b.max(axis=1,keepdims=True)
 e=np.exp(b-m);p=e/e.sum(axis=1,keepdims=True)
 risks=np.log(e.sum(axis=1))+m[:,0]-np.log(5.)
 sigmoid=np.exp(-np.logaddexp(0,-a))
 gradient=weights[:,None]/2*p*sigmoid*(1-2*labels[:,None])
 assert np.all(risks>=b.mean(axis=1)-1e-12) and np.all(risks<=b.max(axis=1)+1e-12)
 return np.mean(weights*risks),gradient
for z in cases:
 variable=tf.Variable(z,dtype=tf.float64)
 with tf.GradientTape() as tape: loss=temporal_tail_risk(variable,labels,weights)
 gradient=tape.gradient(loss,variable).numpy();value,expected=reference(z)
 np.testing.assert_allclose(loss.numpy(),value,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(gradient,expected,rtol=1e-10,atol=1e-10)
 assert np.all(gradient[0]>=0) and np.all(gradient[1]<=0)
 np.testing.assert_allclose(temporal_tail_risk(z[:,::-1],labels,weights).numpy(),value,rtol=1e-10,atol=1e-10)
 errors.append(float(np.max(np.abs(gradient-expected))))
z=cases[1];_,expected=reference(z)
for i in range(2):
 for j in range(5):
  hi=z.copy();lo=z.copy();hi[i,j]+=1e-5;lo[i,j]-=1e-5
  fd=(reference(hi)[0]-reference(lo)[0])/2e-5
  np.testing.assert_allclose(fd,expected[i,j],rtol=1e-7,atol=1e-9)
# A correct-label shift lowers absolute risk; this objective is not shift-invariant.
assert reference(z+(2*labels[:,None]-1)*3)[0]<reference(z)[0]
output={'cases':len(cases),'finite_difference_components':10,'max_gradient_abs_error':max(errors),
 'mean_max_bounds_passed':True,'temporal_permutation_passed':True,'correct_label_shift_lowers_loss':True,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'implementation_sha256':hashlib.sha256(Path('research/wakeword_p4/training/temporal_tail_risk.py').read_bytes()).hexdigest(),
 'limits':'Formula only. No sampler feasibility, actual-network gradients, smoke, formal training or acceptance.'}
p=Path('research/wakeword_p4/results/4090-temporal-tail-risk-formula-preflight-v1.json');assert not p.exists()
p.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
