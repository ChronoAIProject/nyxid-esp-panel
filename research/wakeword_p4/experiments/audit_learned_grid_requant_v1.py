"""Constructed dynamic-step forward/oracle and explicit gradient audit."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import tensorflow as tf
from learned_grid_requant_v1 import requantize
from probe_hey_broad_integer_requant_v2 import fixed
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=[];rng=np.random.default_rng(20261114)
 for ratio in (0.,math.log(.5),math.log(2),math.log(.73),math.log(1.31)):
  for zero,low in ((-128,-128),(38,-128),(-9,-128),(0,0)):
   for ins in (.08132973313331604,.19395044445991516):
    weights=np.array([.00013,.0037,.031],np.float64);base=.12956000864505768
    acc=rng.integers(-20000,20001,(17,3)).astype(np.float64);dy=rng.normal(size=acc.shape);step=float(np.float32(base*math.exp(ratio)));m=ins*weights/step
    codes=np.stack([fixed(acc[:,j].astype(np.int64),v,False) for j,v in enumerate(m)],axis=1)+zero;centered=np.clip(codes,low,127)-zero;expected=centered*step
    a=tf.constant(acc);i=tf.constant(ins,tf.float64);r=tf.constant(ratio,tf.float64)
    with tf.GradientTape() as tape:
     tape.watch([a,i,r]);y=requantize(a,i,r,weights,base,zero,low,127,tf);loss=tf.reduce_sum(y*dy)
    ga,gi,gr=tape.gradient(loss,[a,i,r]);assert all(x is not None and np.isfinite(x.numpy()).all() for x in (ga,gi,gr))
    np.testing.assert_array_equal(y.numpy(),expected)
    u=acc*m;bottom=low-zero;top=127-zero;active=(u>=bottom)&(u<=top)
    np.testing.assert_allclose(ga.numpy(),dy*ins*weights*active,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(gi.numpy(),(dy*acc*weights*active).sum(),rtol=1e-13,atol=1e-13)
    gs=np.where(u<bottom,bottom,np.where(u>top,top,centered-u));oracle=(dy*gs).sum()*step/math.sqrt(acc.size*max(abs(bottom),abs(top)))
    np.testing.assert_allclose(gr.numpy(),oracle,rtol=1e-12,atol=1e-12)
    rows.append(dict(ratio=ratio,zero=zero,input_step=ins,passed=True,step_gradient=float(gr.numpy())))
 out=dict(helper_sha256=sha(Path(__file__).with_name('learned_grid_requant_v1.py')),auditor_sha256=sha(__file__),cases=rows,all_passed=True,model_invocations=0,optimizer_updates=0,limits='Constructed float64 accumulator primitive with reused integer oracle. Dynamic tensor input/output steps and explicit surrogate gradients only. No complete trainable encoder, per-step state/bias-gradient integration, trained export, GPU performance or candidate parity proven. Callback multiplier is offline feasibility implementation.')
 (R/'4090-learned-grid-requant-math-preflight-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),all_passed=True)))
if __name__=='__main__':main()
