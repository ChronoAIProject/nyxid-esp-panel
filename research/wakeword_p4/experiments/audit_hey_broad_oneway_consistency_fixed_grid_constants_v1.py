"""Independent H5 dataset-to-buffer audit; no model construction or compilation."""
import json,hashlib
from pathlib import Path
import h5py,numpy as np
import tensorflow as tf
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter,OpResolverType
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=R/'4090-hey-broad-oneway-consistency-fixed-grid-compile-plan-v1.json'; recovered=R/'4090-hey-broad-oneway-consistency-fixed-grid-compile-result-v1.json'
 lock=json.loads(plan.read_text()); metadata=json.loads(recovered.read_text()); reports=[]
 for arm,receipt in zip(lock['arms'],metadata['arms'],strict=True):
  assert (arm['target'],arm['condition'])==(receipt['target'],receipt['condition'])
  assert sha(arm['weights'])==receipt['final_h5_sha256']; assert sha(receipt['path'])==receipt['sha256']
  with h5py.File(arm['weights']) as f:
   def read(layer,var=0):return f[f'layers/{layer}/vars/{var}'][()]
   params=[(np.moveaxis(read('stream/cell'),3,0),np.zeros(30,np.float32))]
   for i in range(4):
    suffix='' if i==0 else f'_{i}'
    params.append((np.moveaxis(read('depthwise_conv2d'+suffix),3,0),read('depthwise_conv2d'+suffix,1)))
    gamma,beta,mean,var=[read('batch_normalization'+suffix,j) for j in range(4)]
    # Original model BN epsilon is 0.001. Execute only frozen folding arithmetic.
    scale=tf.constant(gamma)*tf.math.rsqrt(tf.constant(var)+tf.constant(.001,tf.float32))
    w=(tf.constant(read('conv2d'+suffix))*scale[None,None,None,:]).numpy()
    bias=(tf.constant(beta)-tf.constant(mean)*scale).numpy()
    params.append((np.moveaxis(w,3,0),bias))
   params.append(((read('dense')*np.float32(.375)).T,read('dense',1)*np.float32(.375)+np.float32(3)))
  model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(Path(receipt['path']).read_bytes(),0)); tensors=model.subgraphs[0].tensors
  interpreter=Interpreter(model_path=receipt['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors()
  checks=[]
  ops=[x for x in arm['template']['ops'] if x['op'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')]
  for op,pair in zip(ops,params,strict=True):
   for ordinal,(index,value) in enumerate(zip(op['inputs'][1:],pair,strict=True)):
    t=tensors[index];q=t.quantization;shape=[1]*value.ndim;shape[q.quantizedDimension]=len(q.scale)
    precision=np.float64 if ordinal else np.float32
    scales=np.asarray(q.scale,np.float32).astype(precision).reshape(shape)
    zero=np.asarray(q.zeroPoint,np.float32).astype(precision).reshape(shape)
    codes=np.rint(value.astype(precision)/scales)+zero
    lo,hi=(-(2**31),2**31-1) if ordinal else (-127,127)
    clipped=int(np.count_nonzero((codes<lo)|(codes>hi)))
    encoded=np.clip(codes,lo,hi).astype('<i4' if ordinal else 'i1')
    actual=bytes(model.buffers[t.buffer].data)
    assert encoded.shape==tuple(t.shape)
    assert encoded.tobytes()==actual,(arm['target'],arm['condition'],index)
    assert interpreter.get_tensor(index).tobytes()==actual
    checks.append(dict(index=int(index),values=int(encoded.size),clipped=clipped,sha256=hashlib.sha256(actual).hexdigest(),buffer_exact=True,runtime_constant_exact=True))
  reports.append(dict(target=arm['target'],condition=arm['condition'],model_sha256=receipt['sha256'],checks=checks))
 dest=R/'4090-hey-broad-oneway-consistency-fixed-grid-constant-audit-v1.json'
 with dest.open('x') as f:json.dump(dict(plan_sha256=sha(plan),recovered_sha256=sha(recovered),arms=reports,all_passed=True,limits='Direct H5 independent dataset mapping; frozen BN arithmetic uses TensorFlow rsqrt to match declared float32 arithmetic. All80 buffers independently encoded and allocated builtinref constant reads exact. No model invocation, compilation or numerical/quality admission.'),f,indent=2);f.write('\n')
 print(str(dest),sha(dest))
if __name__=='__main__':main()
