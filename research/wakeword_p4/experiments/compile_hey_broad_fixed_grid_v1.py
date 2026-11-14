"""Compile final H5 constants into original fixed-grid TFLite, no PTQ or invoke."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-fixed-grid-compile-plan-v1.json';OUT=Path('.local/wakeword/hey-broad-fixed-grid-compile-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,bytes):return {'bytes_sha256':hashlib.sha256(x).hexdigest()}
 if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
 if hasattr(x,'__dict__'):return {k:normalize(v) for k,v in vars(x).items() if not(type(x).__name__=='BufferT' and k=='data')}
 return x
def structural_hash(model):return hashlib.sha256(json.dumps(normalize(model),sort_keys=True).encode()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-paired-qat-formal-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files']);result=R/'4090-hey-broad-paired-qat-formal-result-v1.json';template=R/'4090-hey-broad-equalized-export-result-v1.json'
 for p in [parent,result,template,Path(__file__),R/'4090-hey-broad-paired-qat-formal-independent-audit-v1.json',R/'4090-hey-broad-paired-qat-screen-independent-audit-v1.json',R/'4090-hey-broad-paired-qat-grid-drift-v1.json']:files[str(p)]=sha(p)
 arms=[]
 for c in json.loads(result.read_text())['cases']:
  a=next(a for a in json.loads(template.read_text())['arms'] if a['target']==c['target']);f=c['final_repeat'];files[a['path']]=a['sha256'];files[f['weights_path']]=f['weights_sha256'];arms.append(dict(target=c['target'],condition=c['condition'],weights=f['weights_path'],template=a))
 record=dict(files=files,versions=base['versions'],arms=arms,policy='New independently named compile intervention after original failed540. Allfour same frozenfinal10000weights. Original equalized actual TFLite graph/opversions/allquantizationgrids unchanged. Encode shared finalvariables with training-declared nearest-even weight[-127,127]/int32biaswide/frozenBNfold/boundedhead mapping. Replace only20operatorweightbias buffers and sixoriginalinitializer buffers, zeroencodedusing originalactualquantizer. No converter/PTQ/rangeupdate/source selection/optimizer/modelinvoke. Assert unique bufferrefs, exact shapetype/bytes, unchanged structural fingerprint and all unallowed bytes, allocate actualinterpreter metadata. Preserve original failed PTQ models/reports. Separate compiler/runtime/reference/fullwaveadmission required, no newdev/acceptance/promotion.')
 with P.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def run():
 import subprocess,importlib.metadata
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
 import probe_hey_context_transfer_shapes_v1 as helper
 from microwakeword.layers.stream import Stream
 from ai_edge_litert import schema_py_generated as schema
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 import flatbuffers
 tf=helper.tf;OUT.mkdir(exist_ok=False);arms=[]
 for arm in lock['arms']:
  model=helper.create(225);model.load_weights(arm['weights']);before=[v.numpy().copy() for v in model.variables]
  streams=[l for l in model.layers if isinstance(l,Stream)];dw=[l for l in model.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)];pw=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)];bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)];head,=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
  params=[(np.transpose(streams[0].cell.kernel.numpy(),(3,0,1,2)),np.zeros(30,np.float32))]
  for depth,point,norm in zip(dw,pw,bn,strict=True):
   params.append((np.transpose(depth.kernel.numpy(),(3,0,1,2)),depth.bias.numpy()))
   scale=norm.gamma*tf.math.rsqrt(norm.moving_variance+norm.epsilon);kernel=point.kernel*scale[None,None,None,:];bias=norm.beta-norm.moving_mean*scale;params.append((np.transpose(kernel.numpy(),(3,0,1,2)),bias.numpy()))
  params.append(((head.kernel*tf.constant(3/8,tf.float32)).numpy().T,(head.bias*tf.constant(3/8,tf.float32)+3).numpy()))
  data=Path(arm['template']['path']).read_bytes();original=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data,0));graph=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data,0));structure=structural_hash(graph);main=graph.subgraphs[0];refs={}
  for gi,g in enumerate(graph.subgraphs):
   for ti,t in enumerate(g.tensors):refs.setdefault(t.buffer,[]).append((gi,ti))
  ops=[o for o in arm['template']['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']];assert len(params)==len(ops)==10;allowed=set();constants=[]
  for op,pair in zip(ops,params,strict=True):
   for ordinal,(index,value) in enumerate(zip(op['inputs'][1:],pair,strict=True)):
    t=main.tensors[index];q=t.quantization;sc=np.asarray(q.scale,np.float32);zp=np.asarray(q.zeroPoint,np.float32);shape=[1]*value.ndim;shape[q.quantizedDimension]=len(sc);sc=sc.reshape(shape);zp=zp.reshape(shape);bias=ordinal==1;assert tuple(t.shape)==value.shape and t.type==(schema.TensorType.INT32 if bias else schema.TensorType.INT8);dtype=np.int32 if bias else np.int8;calc=np.float64 if bias else np.float32;unclipped=np.rint(value.astype(calc)/sc.astype(calc))+zp.astype(calc);low,high=(-(1<<31),(1<<31)-1) if bias else (-127,127);codes=np.clip(unclipped,low,high).astype(dtype);buffer=t.buffer;assert refs[buffer]==[(0,index)];assert len(graph.buffers[buffer].data)==codes.nbytes;allowed.add(buffer);graph.buffers[buffer].data=np.frombuffer(codes.astype('<i4' if bias else 'i1').tobytes(),np.uint8).copy();constants.append(dict(index=index,buffer=buffer,shape=value.shape,code_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),clipped=int(np.sum((unclipped<low)|(unclipped>high))),values=int(codes.size)))
  initializers=[]
  for index,t in enumerate(graph.subgraphs[1].tensors):
   buffer=t.buffer
   if graph.buffers[buffer].data is None or len(graph.buffers[buffer].data)==0:continue
   assert t.type==schema.TensorType.INT8 and len(t.quantization.scale)==1;assert refs[buffer]==[(1,index)];zero=int(t.quantization.zeroPoint[0]);codes=np.full(tuple(t.shape),zero,np.int8);assert codes.nbytes==len(graph.buffers[buffer].data);allowed.add(buffer);graph.buffers[buffer].data=codes.view(np.uint8).reshape(-1);initializers.append(dict(shape=list(t.shape),zero=zero,all_real_zero=True))
  assert len(initializers)==6 and len(allowed)==26
  assert structural_hash(graph)==structure
  for j,(a,b) in enumerate(zip(original.buffers,graph.buffers,strict=True)):
   if j not in allowed:assert np.array_equal(a.data,b.data),j
  builder=flatbuffers.Builder(0);offset=graph.Pack(builder);builder.Finish(offset,file_identifier=b'TFL3');compiled=bytes(builder.Output());parsed=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(compiled,0));assert structural_hash(parsed)==structure
  for a,b in zip(parsed.buffers,graph.buffers,strict=True):assert np.array_equal(a.data,b.data)
  dest=OUT/f'{arm["target"]}-{arm["condition"]}';dest.mkdir();path=dest/'hey_nyx.tflite';path.write_bytes(compiled)
  interpreter=Interpreter(model_path=str(path),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors();details={d['index']:d for d in interpreter.get_tensor_details()}
  for c in constants:assert hashlib.sha256(interpreter.get_tensor(c['index']).tobytes()).hexdigest()==c['code_sha256']
  assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
  report=dict(arm['template']);report.update(target=arm['target'],condition=arm['condition'],path=str(path),bytes=path.stat().st_size,sha256=sha(path),template_sha256=arm['template']['sha256'],final_h5_sha256=sha(arm['weights']),structure_sha256=structure,only_allowed_buffers_changed=True,original_grids_exact=True,constants=constants,initializers=initializers,compiler_invocations=1,converter_invocations=0,model_invocations=0);arms.append(report);print(json.dumps(dict(target=arm['target'],condition=arm['condition'],bytes=len(compiled),sha256=sha(path))),flush=True)
 with (R/'4090-hey-broad-fixed-grid-compile-result-v1.json').open('x') as f:f.write(json.dumps(dict(plan_sha256=sha(P),arms=arms,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
