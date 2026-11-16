"""Full initial-weight normalized cache graph integer feasibility only."""
import os
os.environ['TF_NUM_INTRAOP_THREADS']='1';os.environ['TF_NUM_INTEROP_THREADS']='1'
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from build_normalized_keras_v1 import build_normalized

def main():
 torch.set_num_threads(1);tf.config.set_visible_devices([],'GPU');root=Path('.local/wakeword/causal-generator-normalized-v1');cp=root/'initial.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==0
 model=build_normalized(state['model']);inputs={v.name.split(':')[0]:v for v in model.inputs};c=tf.lite.TFLiteConverter.from_keras_model(model);c.optimizations=[tf.lite.Optimize.DEFAULT]
 def rep():
  rng=np.random.default_rng(726)
  for _ in range(64):yield {k:rng.normal(0,1 if k!='chunk' else 10,tuple(v.shape)).astype(np.float32) for k,v in inputs.items()}
 c.representative_dataset=rep;c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
 out=dict(schema='normalized-full-cache-integer-feasibility-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),cp,Path('research/wakeword_p4/experiments/build_normalized_keras_v1.py')]},optimizer_updates=0,recognition_calls=0,limitations='Initial learned weights, analytic random feature/cache representatives. Not real-state calibration, trained export, integer accuracy, reset feedback or P4memory evidence.')
 try:
  raw=c.convert();path=root/'initial-full-graph-int8-feasibility-v2.tflite';assert not path.exists();path.write_bytes(raw);it=Interpreter(model_path=str(path),num_threads=1);it.allocate_tensors();details=it.get_tensor_details();runner=it.get_signature_runner();out.update(converted=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),all_tensors_integer=all(x['dtype'] in [np.int8,np.int32,np.int64] for x in details),input_names=sorted(runner.get_input_details()),output_names=sorted(runner.get_output_details()),ops=[x['op_name'] for x in it._get_ops_details()]);out['noninteger_tensors']=[x['name'] for x in details if x['dtype'] not in [np.int8,np.int32,np.int64]]
 except Exception as e:out.update(converted=False,error_type=type(e).__name__,error=str(e))
 with Path('research/wakeword_p4/results/4090-normalized-full-cache-integer-feasibility-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
