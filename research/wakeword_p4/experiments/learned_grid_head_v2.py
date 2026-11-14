"""Raw-sigmoid model head; integer FC source scalar rounding, offline STE.
Returns unbounded quantized-parameter preclip logit separately from actual
requantized logit; diagnostic gradients do not prove recognition efficacy.
"""
import math
import numpy as np

def forward(last_codes,last_base_step,last_ratio,config,parameters,pooled,tf):
 d=tf.float64;bound=tf.constant(math.log(16),d)
 nominal=tf.constant(last_base_step,d)*tf.exp(tf.clip_by_value(last_ratio,-bound,bound));serialized=tf.cast(tf.cast(nominal,tf.float32),d);step=nominal+tf.stop_gradient(serialized-nominal)
 def rounding(x):return x+tf.stop_gradient(tf.round(x)-x)
 centered=last_codes-tf.constant(config['input_zero'],d)
 frames=tf.signal.frame(tf.pad(centered,[[29,0],[0,0]]),30,1,axis=0)
 if pooled:
  continuous=tf.reduce_mean(frames,axis=1);total=tf.reduce_sum(frames,axis=1)
  raw_total=total+30*config['input_zero']
  actual=tf.sign(raw_total)*tf.floor(tf.abs(raw_total)/30+tf.constant(.5,d))-config['input_zero']
  embedding=continuous+tf.stop_gradient(actual-continuous)
 else:embedding=tf.reshape(frames,[tf.shape(frames)[0],1800])
 w,b=parameters;ws=tf.constant(config['weight_step'],d)
 cw=w/ws;wc=tf.cast(tf.clip_by_value(tf.round(tf.cast(w,tf.float32)/tf.cast(ws,tf.float32)),-127.,127.),d);wc=cw+tf.stop_gradient(wc-cw)
 bs=step*ws;serial=tf.cast(tf.cast(bs,tf.float32),d);bs=bs+tf.stop_gradient(serial-bs)
 bc=tf.clip_by_value(rounding(b/bs),tf.constant(-(1<<31),d),tf.constant((1<<31)-1,d))
 acc=tf.linalg.matmul(embedding,wc,transpose_b=True)+bc
 tf.debugging.assert_equal(acc,tf.round(acc));tf.debugging.assert_greater_equal(acc,tf.constant(-(1<<31),d));tf.debugging.assert_less(acc,tf.constant(1<<31,d))
 raw=acc*step*ws
 u=raw/tf.constant(config['output_step'],d)
 rounded=tf.sign(u)*tf.floor(tf.abs(u)+tf.constant(.5,d))
 actual=tf.clip_by_value(rounded+config['output_zero'],-128.,127.)
 deployed=(actual-config['output_zero'])*config['output_step']
 logit=raw+tf.stop_gradient(deployed-raw)
 return embedding+config['input_zero'],actual,raw,logit
