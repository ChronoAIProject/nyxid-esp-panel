"""Explicit int32 double-rounding forward, declared continuous clipped STE."""
import numpy as np
from probe_hey_broad_integer_requant_v2 import multiplier

def requantize_ste(accumulator,real_multipliers,zero,low,high,output_scale,tf):
    factors=[multiplier(v) for v in np.atleast_1d(real_multipliers)]
    q=tf.constant([v[0] for v in factors],tf.int64)
    shift=np.asarray([v[1] for v in factors],np.int64)
    left=tf.constant(1<<np.maximum(shift,0),tf.int64)
    right=tf.constant(np.maximum(-shift,0),tf.int64)
    mask=tf.bitwise.left_shift(tf.ones_like(right),right)-1
    slope=tf.constant(np.atleast_1d(real_multipliers)*output_scale,tf.float64)
    @tf.custom_gradient
    def apply(value):
        rounded=tf.round(value)
        tf.debugging.assert_equal(value,rounded)
        tf.debugging.assert_greater_equal(value,tf.constant(-(1<<31),value.dtype))
        tf.debugging.assert_less(value,tf.constant(1<<31,value.dtype))
        x=tf.cast(rounded,tf.int64)
        tf.debugging.assert_greater_equal(x*left,tf.constant(-(1<<31),tf.int64))
        tf.debugging.assert_less(x*left,tf.constant(1<<31,tf.int64))
        product=x*left*q
        adjusted=product+tf.where(product>=0,tf.constant(1<<30,tf.int64),tf.constant(1-(1<<30),tf.int64))
        highmul=tf.where(adjusted>=0,adjusted//(1<<31),-((-adjusted)//(1<<31)))
        remainder=tf.bitwise.bitwise_and(highmul,mask)
        threshold=tf.bitwise.right_shift(mask,1)+tf.cast(highmul<0,tf.int64)
        code=tf.bitwise.right_shift(highmul,right)+tf.cast(remainder>threshold,tf.int64)+zero
        result=tf.cast(tf.clip_by_value(code,low,high)-zero,value.dtype)*tf.cast(output_scale,value.dtype)
        def grad(dy):
            # Declared surrogate, not derivative of discontinuous integer forward.
            continuous=value*tf.cast(slope,value.dtype)+tf.cast(zero*output_scale,value.dtype)
            active=(continuous>=low*output_scale)&(continuous<=high*output_scale)
            return dy*tf.cast(slope,value.dtype)*tf.cast(active,value.dtype)
        return result,grad
    return apply(accumulator)
