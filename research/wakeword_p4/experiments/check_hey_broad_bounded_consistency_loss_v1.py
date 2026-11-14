"""Analytic standalone bounded loss/gradient contract, no models or optimizer."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_bounded_consistency_loss_v1 import components
 from hey_broad_consistency_loss_v1 import components as original
 tf=helper.tf;reports=[]
 values=np.asarray([-128,-64,-16,-8,-7.99,-1,0,1,7.99,8,16,64,128],np.float64).reshape(-1,1)
 for dtype in (np.float32,np.float64):
  for coefficient in (0,1):
   for name,a,b in [('same',values,values*2),('opposed',values,-values),('shift',values,values+1)]:
    x=tf.constant(a.astype(dtype));z=tf.constant(b.astype(dtype));labels=tf.constant((np.arange(len(a))%2).reshape(-1,1).astype(dtype));weight=tf.constant(np.where(np.arange(len(a))%2,1,20).astype(dtype))
    with tf.GradientTape(persistent=True) as tape:
     tape.watch((x,z));total,s,c=components(x,z,x,z,labels,weight,coefficient,tf)
    gradients=[g.numpy() for g in tape.gradient(total,(x,z))];cg=[g.numpy() for g in tape.gradient(c,(x,z))];_,old_s,_=original(x,z,labels,weight,0,tf);assert np.array_equal(s.numpy(),old_s.numpy())
    af=x.numpy().astype(float);bf=z.numpy().astype(float);y=labels.numpy().astype(float);w=weight.numpy().astype(float).reshape(-1,1);n=len(a);cl=np.clip(af,-8,8);cr=np.clip(bf,-8,8);gap=(cl-cr)/8;terms=np.where(np.abs(gap)<=1,.5*gap*gap,np.abs(gap)-.5);expected_c=terms.mean();base=np.clip(gap,-1,1)/8/n
    dc=[base*((af>=-8)&(af<=8)),-base*((bf>=-8)&(bf<=8))];ds=[w*(1/(1+np.exp(-af))-y)/(2*n),w*(1/(1+np.exp(-bf))-y)/(2*n)];expected_s=float(np.mean(w*(np.logaddexp(0,af)-y*af+np.logaddexp(0,bf)-y*bf)/2))
    errors=[abs(float(c)-expected_c),abs(float(s)-expected_s)]
    for actual,expected in zip(cg,dc,strict=True):np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6);errors.append(float(np.max(np.abs(actual-expected))))
    for actual,expected_sg,expected_cg in zip(gradients,ds,dc,strict=True):np.testing.assert_allclose(actual,expected_sg+coefficient*expected_cg,rtol=2e-6,atol=2e-6);errors.append(float(np.max(np.abs(actual-expected_sg-coefficient*expected_cg))))
    np.testing.assert_allclose([float(s),float(c)],[expected_s,expected_c],rtol=2e-6,atol=2e-6);reports.append(dict(dtype=np.dtype(dtype).name,coefficient=coefficient,fixture=name,points=n,supervised_original_exact=True,max_abs_error=max(errors)))
 # Explicit semantics: identical nominal saturation gives zero C; wrong-label raw BCE still trains.
 x=tf.constant([[-64.]],tf.float64);q=tf.constant([[-128.]],tf.float64)
 with tf.GradientTape(persistent=True) as tape:
  tape.watch((x,q));_,s,c=components(x,q,x,q,tf.ones_like(x),tf.ones([1],tf.float64),1,tf)
 assert float(c)==0 and all(float(g)==0 for g in tape.gradient(c,(x,q)));assert all(float(g)<0 for g in tape.gradient(s,(x,q)))
 source=Path(__file__).with_name('hey_broad_bounded_consistency_loss_v1.py')
 with (R/'4090-hey-broad-bounded-consistency-loss-check-v1.json').open('x') as f:json.dump(dict(files={str(p):sha(p) for p in (source,Path(__file__))},checks=reports,same_saturation_c_zero=True,confident_wrong_raw_bce_gradient_preserved=True,optimizer_updates=0,model_invocations=0,limits='Standalone nominal bounded loss math only, not full deployment/head/network gradient correctness, actual training effect, recognition or candidate admission. Primitive arithmetic check tolerances do not alter original wholewave gates.'),f,indent=2);f.write('\n')
 print('12 analytic loss/gradient fixtures and saturation semantics passed')
if __name__=='__main__':main()
