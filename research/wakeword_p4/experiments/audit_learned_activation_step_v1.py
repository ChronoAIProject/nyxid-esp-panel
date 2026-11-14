"""Independent NumPy surrogate oracle vs TF helper; no media/model/optimizer."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import tensorflow as tf
from learned_activation_step_v1 import quantize
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=[]
 for dtype in (np.float32,np.float64):
  for zero in (-128,0,38,127):
   for ratio in (-4.,0.,4.):
    base=.125
    step=base*math.exp(max(-math.log(16),min(math.log(16),ratio)))
    u=np.array([-400,-128-zero-1,-128-zero,-2.5,-.2,0,.2,2.5,127-zero,128-zero,400],dtype=dtype)
    x=np.array(u*step,dtype=dtype);dy=np.linspace(-1,1,len(x),dtype=dtype)
    # Recompute effective u from actual typed input and typed step.
    st=np.array(base,dtype=dtype)*np.exp(np.clip(np.array(ratio,dtype=dtype),np.array(-math.log(16),dtype=dtype),np.array(math.log(16),dtype=dtype)))
    actual_u=x/st;lo=-128-zero;hi=127-zero
    expected=st*np.clip(np.rint(actual_u),lo,hi)
    dx=dy*((actual_u>=lo)&(actual_u<=hi))
    ds=np.where(actual_u<lo,lo,np.where(actual_u>hi,hi,np.rint(actual_u)-actual_u))
    dr=float(np.sum(dy*ds)*st/math.sqrt(len(x)*max(abs(lo),abs(hi)))) if abs(ratio)<math.log(16) else 0.
    tx=tf.constant(x);tr=tf.constant(ratio,dtype=tx.dtype)
    with tf.GradientTape() as tape:
     tape.watch([tx,tr]);out=quantize(tx,tr,base,zero,tf);loss=tf.reduce_sum(out*dy)
    gx,gr=tape.gradient(loss,[tx,tr])
    tol=3e-5 if dtype==np.float32 else 1e-12
    np.testing.assert_allclose(out.numpy(),expected,rtol=tol,atol=tol)
    np.testing.assert_allclose(gx.numpy(),dx,rtol=0,atol=0)
    np.testing.assert_allclose(gr.numpy(),dr,rtol=tol,atol=tol)
    assert np.isfinite(out.numpy()).all() and np.isfinite(gr.numpy())
    rows.append(dict(dtype=str(dtype),zero=zero,log_ratio=ratio,passed=True))
 # Explicit derivative distinction: finite differences on a fixed rounding cell.
 x=tf.constant([.03],tf.float64);r=tf.constant(0.,tf.float64)
 with tf.GradientTape() as tape:
  tape.watch(r);v=quantize(x,r,.125,0,tf)
 surrogate=float(tape.gradient(v,r).numpy())
 eps=1e-6
 fd=float(((quantize(x,r+eps,.125,0,tf)-quantize(x,r-eps,.125,0,tf))/(2*eps)).numpy().item())
 assert fd==0 and surrogate!=0
 out=dict(helper_sha256=sha(Path(__file__).with_name('learned_activation_step_v1.py')),auditor_sha256=sha(__file__),cases=rows,all_passed=True,finite_difference_counterexample=dict(actual_forward_derivative=fd,surrogate_derivative=surrogate),model_invocations=0,optimizer_updates=0,limits='Constructed primitive only. Explicit surrogate differs from actual finite differences. No complete integer graph, shared-state step binding, export parity, training efficacy or candidate qualification proven.')
 (R/'4090-learned-activation-step-math-preflight-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
