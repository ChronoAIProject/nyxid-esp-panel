"""Offline differentiable complete temporal encoder with nine shared log-steps.
Original quantized kernels and real quantized biases are immutable feasibility
constants. This is scale-path integration, not trainable-weight QAT or FC.
"""
import math
import numpy as np
from learned_grid_requant_v1 import requantize

def forward(input_codes,configs,ratios,tf):
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
  k=c['kernel'];hist=k.shape[1]-3 if j==0 else k.shape[1]-1
  codes=tf.pad(codes,[[hist,0],[0,0]])
  frames=tf.signal.frame(codes,k.shape[1],3 if j==0 else 1,axis=0)
  if c['kind']=='DEPTHWISE_CONV_2D':acc=tf.reduce_sum(frames*tf.constant(k[0,:,0,:],dtype),axis=1)
  else:acc=tf.einsum('ntc,otc->no',frames,tf.constant(k[:,:,0,:],dtype))
  ws=tf.constant(c['weight_steps'],dtype)
  bias_scale=ins*ws;serialized=tf.cast(tf.cast(bias_scale,tf.float32),dtype)
  bias_scale=bias_scale+tf.stop_gradient(serialized-bias_scale)
  bias=rounding(tf.constant(c['bias_real'],dtype)/bias_scale)
  acc=acc+bias
  x=requantize(acc,ins,ratios[j],c['weight_steps'],c['output_step'],c['output_zero'],max(-128,c['output_zero']) if c['kind']=='CONV_2D' else -128,127,tf)
  outputs.append((x/outstep+c['output_zero']))
 return outputs
