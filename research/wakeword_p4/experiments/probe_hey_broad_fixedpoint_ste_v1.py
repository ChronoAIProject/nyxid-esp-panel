"""Saved actual accumulator forward and analytic gradient primitive checks."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import probe_hey_context_transfer_shapes_v1 as helper
 from hey_broad_fixedpoint_ste_v1 import requantize_ste
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;arm=json.loads((R/'4090-hey-broad-equalized-export-result-v1.json').read_text())['arms'][0];i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()};groups=[]
 source=Path('.local/wakeword/hey-broad-integer-requant-v2.npz')
 with np.load(source) as a:
  for op in arm['ops']:
   if op['op'] not in ['CONV_2D','DEPTHWISE_CONV_2D']:continue
   ix,wi,bi=op['inputs'];oy=op['outputs'][0];ins,_=d[ix]['quantization'];os,oz=d[oy]['quantization'];ws=d[wi]['quantization_parameters']['scales'].astype(np.float64);m=ins*ws/os;low=max(-128,oz) if op['op']=='CONV_2D' else -128;high=127
   values=a[f'{oy}_accumulator'].astype(np.float64);x=tf.Variable(values,dtype=tf.float64);weights=np.cos(np.arange(values.size)).reshape(values.shape)
   with tf.GradientTape() as tape:
    y=requantize_ste(x,m,oz,low,high,os,tf);loss=tf.reduce_sum(y*weights)
   gradient=tape.gradient(loss,x).numpy();expected=(a[f'{oy}_double_rounding'].astype(np.float64)-oz)*os
   assert np.array_equal(y.numpy(),expected),oy
   continuous=values*(m*os)+oz*os;analytic=weights*(m*os)*((continuous>=low*os)&(continuous<=high*os));assert np.array_equal(gradient,analytic),oy
   assert np.array_equal(x.numpy(),values)
   groups.append(dict(index=oy,points=int(values.size),forward_exact=True,analytic_gradient_exact=True,unchanged=True))
 result=dict(schema='hey-broad-fixedpoint-ste-result-v1',primitive_sha256=sha(Path(__file__).with_name('hey_broad_fixedpoint_ste_v1.py')),probe_sha256=sha(__file__),saved_accumulators_sha256=sha(source),groups=groups,limits='Actual80160saved accumulators only, no independent synthetic boundary/tie/saturation stress yet. Analytic declaredSTE exact, not derivative of integer forward. No model invoke/optimizer/export or fullgraph training validation.')
 p=R/'4090-hey-broad-fixedpoint-ste-result-v1.json'
 with p.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
