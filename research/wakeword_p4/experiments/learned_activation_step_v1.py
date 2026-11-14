"""Bounded log-step LSQ-style surrogate; not an integer-network forward."""
import math

def quantize(value,log_ratio,base_step,zero,tf):
    """Fixed zero point; step=base*exp(clip(log_ratio,log(1/16),log(16))).

    Forward ties-to-even affine INT8. Backward uses an explicitly surrogate
    LSQ-style step derivative, normalized by sqrt(N*max centered code magnitude).
    This does NOT equal the derivative of the discontinuous forward function.
    """
    dtype=value.dtype
    if log_ratio.dtype!=dtype:raise TypeError('value and log_ratio dtype differ')
    if not math.isfinite(base_step) or base_step<=0:raise ValueError('invalid base step')
    if not isinstance(zero,int) or not -128<=zero<=127:raise ValueError('invalid zero')
    if log_ratio.shape.rank!=0:raise ValueError('log_ratio must be scalar')
    @tf.custom_gradient
    def apply(x,r):
        tf.debugging.assert_all_finite(x,'nonfinite activation')
        tf.debugging.assert_all_finite(r,'nonfinite log-step')
        low=tf.constant(-math.log(16),dtype);high=tf.constant(math.log(16),dtype)
        step=tf.constant(base_step,dtype)*tf.exp(tf.clip_by_value(r,low,high))
        u=x/step;bottom=tf.constant(-128-zero,dtype);top=tf.constant(127-zero,dtype)
        code=tf.clip_by_value(tf.round(u),bottom,top)
        y=step*code
        def grad(dy):
            inside=(u>=bottom)&(u<=top)
            gx=dy*tf.cast(inside,dtype)
            gs=tf.where(u<bottom,bottom,tf.where(u>top,top,tf.round(u)-u))
            n=tf.cast(tf.size(x),dtype)
            norm=tf.math.rsqrt(n*tf.constant(max(abs(-128-zero),abs(127-zero)),dtype))
            gr=tf.reduce_sum(dy*gs)*step*norm*tf.cast((r>low)&(r<high),dtype)
            return gx,gr
        return y,grad
    return apply(value,log_ratio)
