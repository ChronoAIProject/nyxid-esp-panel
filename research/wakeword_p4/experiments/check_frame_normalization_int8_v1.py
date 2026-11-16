"""Normalization-only builtin INT8 feasibility, analytic input not recognition."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_NUM_INTRAOP_THREADS']='1';os.environ['TF_NUM_INTEROP_THREADS']='1'
import hashlib,json
from pathlib import Path
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

def main():
 x=tf.keras.Input(batch_shape=(1,3,1,64),name='frame');y=tf.keras.layers.Lambda(lambda z:(z-tf.reduce_mean(z,axis=-1,keepdims=True))*tf.math.rsqrt(tf.reduce_mean(tf.square(z-tf.reduce_mean(z,axis=-1,keepdims=True)),axis=-1,keepdims=True)+1e-5))(x);m=tf.keras.Model(x,y);c=tf.lite.TFLiteConverter.from_keras_model(m);c.optimizations=[tf.lite.Optimize.DEFAULT]
 def rep():
  rng=np.random.default_rng(726)
  for _ in range(64):yield [rng.normal(0,10,(1,3,1,64)).astype(np.float32)]
 c.representative_dataset=rep;c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
 out=dict(schema='frame-normalization-int8-feasibility-v1',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),trained=False,recognition_calls=0,limitations='Normalization-only analytic calibration; does not prove full cache graph or real-wave accuracy.')
 try:
  raw=c.convert();p=Path('.local/wakeword/frame-normalization-feasibility-v1.tflite');assert not p.exists();p.write_bytes(raw);it=Interpreter(model_path=str(p),num_threads=1);it.allocate_tensors();d=it.get_tensor_details();integer=all(t['dtype'] in [np.int8,np.int32,np.int64] for t in d);out.update(converted=True,all_tensors_integer=integer,ops=[x['op_name'] for x in it._get_ops_details()],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());out['noninteger_tensors']=[dict(name=t['name'],dtype=str(t['dtype'])) for t in d if t['dtype'] not in [np.int8,np.int32,np.int64]]
 except Exception as e:out.update(converted=False,error_type=type(e).__name__,error=str(e))
 with Path('research/wakeword_p4/results/4090-frame-normalization-int8-feasibility-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
