"""Standalone scalar FC numerical/declared-gradient checks; optimizer zero."""
import math,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_scalar_fc_ste_v1 import scalar_fc_ste
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf
 source=Path(__file__).with_name('hey_broad_scalar_fc_ste_v1.py');compile=R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json';rows=[]
 configurations=[dict(name='signed-half-ties',multiplier=.5,zero=-128,low=-128,high=127,output_scale=.125),dict(name='inclusive-saturation',multiplier=.25,zero=-2,low=-4,high=4,output_scale=.5)]
 for arm in json.loads(compile.read_text())['arms']:
  assert sha(arm['path'])==arm['sha256'];i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()}
  si,_=d[64]['quantization'];sw=float(d[34]['quantization_parameters']['scales'][0]) if False else None
  op=next(o for o in i._get_ops_details() if o['op_name']=='FULLY_CONNECTED');wi=int(op['inputs'][1]);sw=float(d[wi]['quantization_parameters']['scales'][0]);so,zero=d[65]['quantization']
  configurations.append(dict(name=arm['target']+'/'+arm['condition'],multiplier=si*sw/so,zero=zero,low=max(-128,zero),high=min(127,int(np.rint(6/so))+zero),output_scale=so,model_sha256=arm['sha256']))
 for cfg in configurations:
  m=cfg['multiplier'];zero=cfg['zero'];lo=cfg['low'];hi=cfg['high'];scale=cfg['output_scale']
  integers=set(range(-1024,1025))|{-(1<<31),(1<<31)-1}
  for edge in (lo-zero,hi-zero,0):
   center=math.floor(edge/m)
   integers.update(n for n in range(center-3,center+4) if -(1<<31)<=n<(1<<31))
  values=np.asarray(sorted(integers),np.float64);weights=np.linspace(-1,1,len(values),dtype=np.float64);signatures=[]
  for repeat in range(2):
   x=tf.constant(values,tf.float64)
   with tf.GradientTape() as tape:
    tape.watch(x);y=scalar_fc_ste(x,m,zero,lo,hi,scale,tf);loss=tf.reduce_sum(y*weights)
   gradient=tape.gradient(loss,x).numpy();actual=y.numpy()
   expected=np.asarray([(min(hi,max(lo,(math.floor(abs(float(n)*m)+.5)*(1 if n*m>0 else -1 if n*m<0 else 0))+zero))-zero)*scale for n in values],np.float64)
   slopes=np.asarray([weights[k]*(m*scale) if lo<=float(n)*m+zero<=hi else 0 for k,n in enumerate(values)])
   assert np.array_equal(actual,expected) and np.array_equal(gradient,slopes),cfg['name']
   signatures.append(hashlib.sha256(actual.tobytes()+gradient.tobytes()).hexdigest())
  assert signatures[0]==signatures[1]
  rows.append(dict(**cfg,values=len(values),forward_exact=True,declared_gradient_exact=True,repeat_exact=True,signature=signatures[0]))
 rejected=[]
 for value in (.5,float('nan'),float('inf'),float(1<<31),float(-(1<<31)-1)):
  try:scalar_fc_ste(tf.constant([value],tf.float64),.5,0,-128,127,.125,tf).numpy()
  except tf.errors.InvalidArgumentError:rejected.append(str(value));continue
  raise AssertionError('Invalid accumulator accepted')
 out=R/'4090-hey-broad-scalar-fc-ste-check-v1.json'
 with out.open('x') as f:json.dump(dict(files={str(p):sha(p) for p in (source,Path(__file__),compile)},checks=rows,invalid_accumulators_rejected=rejected,optimizer_updates=0,model_invocations=0,all_passed=True,limits='Standalone primitive and model metadata only, no shared-network integration, autodiff equivalence to integer discontinuities, complete deployment head/probability parity, recognition or candidate qualification. Does not diagnose current failure or change training.'),f,indent=2);f.write('\n')
 print('Scalar FC primitive:',len(rows),'forward/gradient/repeat grids passed')
if __name__=='__main__':main()
