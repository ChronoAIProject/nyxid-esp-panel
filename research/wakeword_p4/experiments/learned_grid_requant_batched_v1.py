"""Dynamic tensor-step integer forward with declared surrogate gradients.
Python multiplier callback is an offline training feasibility bridge, not deployable
or a performance claim. Exported float32 scale serialization is part of forward.
"""
import math
import numpy as np
from probe_hey_broad_integer_requant_v2 import multiplier

def requantize(accumulator,input_step,log_ratio,weight_steps,base_output_step,zero,low,high,tf):
 dtype=accumulator.dtype
 if input_step.dtype!=dtype or log_ratio.dtype!=dtype:raise TypeError('mixed dtypes')
 weights=tf.constant(np.atleast_1d(weight_steps),dtype)
 def factors(m):
  pairs=[multiplier(x) for x in np.atleast_1d(m)]
  return np.array([x[0] for x in pairs],np.int64),np.array([x[1] for x in pairs],np.int64)
 @tf.custom_gradient
 def apply(acc,ins,r):
  tf.debugging.assert_positive(ins);tf.debugging.assert_all_finite(r,'nonfinite ratio')
  bound=tf.constant(math.log(16),dtype)
  nominal=tf.constant(base_output_step,dtype)*tf.exp(tf.clip_by_value(r,-bound,bound))
  # Actual FlatBuffer scale is float32, even when training arithmetic is float64.
  step=tf.cast(tf.cast(nominal,tf.float32),dtype)
  m=ins*weights/step
  q,shift=tf.numpy_function(factors,[m],[tf.int64,tf.int64]);q.set_shape(weights.shape);shift.set_shape(weights.shape)
  left=tf.bitwise.left_shift(tf.ones_like(shift),tf.maximum(shift,0));right=tf.maximum(-shift,0)
  tf.debugging.assert_less_equal(right,tf.constant(31,tf.int64))
  tf.debugging.assert_equal(acc,tf.round(acc));tf.debugging.assert_all_finite(acc,'nonfinite accumulator')
  x=tf.cast(acc,tf.int64);tf.debugging.assert_greater_equal(x*left,tf.constant(-(1<<31),tf.int64));tf.debugging.assert_less(x*left,tf.constant(1<<31,tf.int64))
  prod=x*left*q;nudge=tf.where(prod>=0,tf.constant(1<<30,tf.int64),tf.constant(1-(1<<30),tf.int64));adjusted=prod+nudge
  h=tf.where(adjusted>=0,adjusted//(1<<31),-((-adjusted)//(1<<31)))
  mask=tf.bitwise.left_shift(tf.ones_like(right),right)-1
  rem=tf.bitwise.bitwise_and(h,mask);threshold=tf.bitwise.right_shift(mask,1)+tf.cast(h<0,tf.int64)
  codes=tf.bitwise.right_shift(h,right)+tf.cast(rem>threshold,tf.int64)+zero
  centered=tf.cast(tf.clip_by_value(codes,low,high)-zero,dtype)
  y=centered*step
  def grad(dy):
   u=acc*m;bottom=tf.constant(low-zero,dtype);top=tf.constant(high-zero,dtype)
   active=(u>=bottom)&(u<=top);mask=tf.cast(active,dtype)
   ga=dy*ins*weights*mask
   gi=tf.reduce_sum(dy*acc*weights*mask)
   gs=tf.where(u<bottom,bottom,tf.where(u>top,top,centered-u))
   norm=tf.math.rsqrt(tf.cast(tf.reduce_prod(tf.shape(acc)[-2:]),dtype)*tf.constant(max(abs(low-zero),abs(high-zero)),dtype))
   gr=tf.reduce_sum(dy*gs)*step*norm*tf.cast((r>-bound)&(r<bound),dtype)
   return ga,gi,gr
  return y,grad
 return apply(accumulator,input_step,log_ratio)
