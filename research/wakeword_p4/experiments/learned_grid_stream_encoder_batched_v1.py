"""Offline differentiable complete temporal encoder with nine shared log-steps.
Trainable folded FP32 kernels/bias, fixed per-channel weight grids and nine
activation steps. This is offline encoder integration, no FC or optimizer.
"""
import math
import numpy as np
from learned_grid_requant_batched_v1 import requantize

def forward(input_codes,configs,ratios,tf,parameters):
 dtype=tf.float64
 def step(j):
  if j<0:return tf.constant(configs[0]['input_step'],dtype)
  base=tf.constant(configs[j]['output_step'],dtype);bound=tf.constant(math.log(16),dtype)
  s=base*tf.exp(tf.clip_by_value(ratios[j],-bound,bound))
  # Preserve identity surrogate through actual float32 serialized step.
  serialized=tf.cast(tf.cast(s,tf.float32),dtype)
  return s+tf.stop_gradient(serialized-s)
 def rounding(x):return x+tf.stop_gradient(tf.round(x)-x)
 x=(tf.constant(input_codes,dtype)-configs[0]['input_zero'])*step(-1);outputs=[]
 for j,c in enumerate(configs):
  ins=step(j-1);outstep=step(j)
  # Representation is real-valued; centered integer code identities carry STE.
  codes=rounding(x/ins)
  weight,bias_real=parameters[j]
  shape=[1]*4;shape[3 if c['kind']=='DEPTHWISE_CONV_2D' else 0]=len(c['weight_steps'])
  weight_scale=tf.constant(np.array(c['weight_steps']).reshape(shape),dtype)
  continuous=weight/weight_scale
  actual=tf.cast(tf.clip_by_value(tf.round(tf.cast(weight,tf.float32)/tf.cast(weight_scale,tf.float32)),-127.,127.),dtype)
  k=continuous+tf.stop_gradient(actual-continuous)
  hist=k.shape[1]-3 if j==0 else k.shape[1]-1
  codes=tf.pad(codes,[[0,0],[hist,0],[0,0]])
  frames=tf.signal.frame(codes,k.shape[1],3 if j==0 else 1,axis=1)
  if c['kind']=='DEPTHWISE_CONV_2D':acc=tf.reduce_sum(frames*k[0,:,0,:],axis=2)
  else:acc=tf.einsum('bntc,otc->bno',frames,k[:,:,0,:])
  ws=tf.constant(c['weight_steps'],dtype)
  bias_scale=ins*ws;serialized=tf.cast(tf.cast(bias_scale,tf.float32),dtype)
  bias_scale=bias_scale+tf.stop_gradient(serialized-bias_scale)
  bias=tf.clip_by_value(rounding(bias_real/bias_scale),tf.constant(-(1<<31),dtype),tf.constant((1<<31)-1,dtype))
  acc=acc+bias
  x=requantize(acc,ins,ratios[j],c['weight_steps'],c['output_step'],c['output_zero'],max(-128,c['output_zero']) if c['kind']=='CONV_2D' else -128,127,tf)
  outputs.append((x/outstep+c['output_zero']))
 return outputs
