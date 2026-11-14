"""All actual encoder grid boundary/saturation and independent integer oracle."""
import json,math,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oracle(x,m):
 f,e=math.frexp(float(m));q=math.floor(f*(1<<31)+.5)
 if q==(1<<31):q//=2;e+=1
 if e< -31:q=0;e=0
 prod=int(x)*(1<<max(e,0))*q
 h=(prod+(1<<30))//(1<<31)
 r=max(-e,0)
 if not r:return h
 return (1 if h>=0 else -1)*((abs(h)+(1<<(r-1)))//(1<<r))
def main():
 import probe_hey_context_transfer_shapes_v1 as helper
 from hey_broad_fixedpoint_ste_v2 import requantize_ste
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;groups=[];total=0
 for arm in json.loads((R/'4090-hey-broad-equalized-export-result-v1.json').read_text())['arms']:
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()}
  for op in arm['ops']:
   if op['op'] not in ['CONV_2D','DEPTHWISE_CONV_2D']:continue
   ix,wi,_=op['inputs'];oy=op['outputs'][0];ins,_=d[ix]['quantization'];os,z=d[oy]['quantization'];m=ins*d[wi]['quantization_parameters']['scales'].astype(np.float64)/os;lo=max(-128,z) if op['op']=='CONV_2D' else -128
   for channel,v in enumerate(m):
    left=max(math.frexp(float(v))[1],0);minimum=-(1<<31)//(1<<left);maximum=((1<<31)-1)//(1<<left)
    vals={minimum,maximum,-1,0,1}
    # Integer accumulator neighbors around all output levels and half-levels,
    # plus both representable boundaries and saturation, no candidate scores.
    for level in np.arange(-257,258)/2:
     center=math.floor((float(level)-z)/float(v))
     vals.update(x for x in range(center-2,center+4) if minimum<=x<=maximum)
    a=np.asarray(sorted(vals),np.float64)[:,None];x=tf.Variable(a);weights=np.cos(np.arange(len(a)))[:,None]
    with tf.GradientTape() as tape:
     y=requantize_ste(x,[v],z,lo,127,os,tf);loss=tf.reduce_sum(y*weights)
    grad=tape.gradient(loss,x).numpy();codes=np.asarray([min(127,max(lo,oracle(int(n),v)+z)) for n in a[:,0]],np.float64)[:,None]
    expected=(codes-z)*os;assert np.array_equal(y.numpy(),expected),(arm['target'],oy,channel)
    continuous=a*(v*os)+z*os;analytic=weights*(v*os)*((continuous>=lo*os)&(continuous<=127*os));assert np.array_equal(grad,analytic)
    if left:
     try:requantize_ste(tf.constant([[float((1<<31)-1)]],tf.float64),[v],z,lo,127,os,tf)
     except tf.errors.InvalidArgumentError:pass
     else:raise AssertionError('Invalid shifted int32 input accepted')
    groups.append(dict(target=arm['target'],index=oy,channel=channel,points=len(a),forward_exact=True,analytic_gradient_exact=True));total+=len(a)
 result=dict(schema='hey-broad-fixedpoint-boundaries-result-v2',primitive_sha256=sha(Path(__file__).with_name('hey_broad_fixedpoint_ste_v2.py')),probe_sha256=sha(__file__),groups=len(groups),points=total,all_forward_exact=True,all_analytic_gradient_exact=True,coverage=groups,limits='Independent scalar integer oracle on actual encoder grids, output levels/halves and integer-neighbor saturation/legal shifted-int32 endpoints and invalid shifted input rejection. Not exhaustive int32 space or fullgraph gradient validation; no optimizer/model export/new data.')
 with (R/'4090-hey-broad-fixedpoint-boundaries-result-v2.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='coverage'},indent=2))
if __name__=='__main__':main()
