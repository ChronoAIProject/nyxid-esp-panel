"""Once-only INT8 preflight on saved untrained pooled heads; synthetic inputs only."""
import argparse,hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-pooled-int8-execution-v1.json';OUT=Path('.local/wakeword/pooled-int8-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-pooled-stream-phase-execution-v1.json';result=R/'4090-pooled-stream-phase-original-result-v1.json'
 p=json.loads(parent.read_text());r=json.loads(result.read_text());assert r['all_passed'] and r['execution_sha256']==sha(parent)
 files=dict(p['files'])
 for f in [parent,result,Path(__file__)]+[Path(c[k]) for c in r['cases'] for k in ['h5_path','references_path']]:files[str(f)]=sha(f)
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,models=r['cases'],inputs=p['inputs'],versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},contract='Original four saved pooled initialized H5, raw sigmoid unchanged. One INT8 variable-quantized builtin-only export/model, int8 input/uint8 output, six realzero initializers. Representative: all4 fixed synthetic256step streams in original order, no boundary resets. No retries/calibration selection. Two fresh interpreters per stream, all startup. Finite/repeat exact; per-stream five-step MAE<=.02,p99<=.1,max<=.25 against saved original raw TF. No optimizer/audio/features/dev/recognition. Passing untrained-head preflight is not trained-model admission.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import numpy as np
  import probe_hey_context_transfer_shapes_v1 as helper
  from microwakeword import mixednet,utils
  from microwakeword.layers import modes
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  from tensorflow.lite.python import schema_py_generated as schema
  tf=helper.tf;inputs=np.fromfile(p['inputs'],dtype='<f4').reshape(4,256,1,3,40);rows=[]
  names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
  for m in p['models']:
   folder=OUT/(m['target']+'-'+m['condition']);folder.mkdir();parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
   model=mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None);model.load_weights(m['h5_path'])
   for l in model.layers:
    if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
   before=[v.numpy().copy() for v in model.variables]
   stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
   states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   for s in states:s.assign(tf.zeros_like(s))
   stream.export(str(folder/'saved_model'));converter=tf.lite.TFLiteConverter.from_saved_model(str(folder/'saved_model'))
   converter.optimizations=[tf.lite.Optimize.DEFAULT];converter._experimental_variable_quantization=True
   converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.uint8
   def representative():
    for case in inputs:
     for value in case:yield [value.copy()]
   converter.representative_dataset=representative;data=converter.convert();path=folder/'model.tflite';path.write_bytes(data)
   g=schema.Model.GetRootAsModel(data,0);sg=g.Subgraphs(0);ops=[names[g.OperatorCodes(sg.Operators(i).OpcodeIndex()).BuiltinCode()] for i in range(sg.OperatorsLength())]
   assert 'AVERAGE_POOL_2D' in ops and not any(n in ops for n in ['CUSTOM','DEQUANTIZE'])
   assert all(sg.Tensors(i).Type()!=schema.TensorType.FLOAT32 for i in range(sg.TensorsLength()))
   initializers=[]
   for si in range(1,g.SubgraphsLength()):
    sub=g.Subgraphs(si)
    for ti in range(sub.TensorsLength()):
     t=sub.Tensors(ti);b=g.Buffers(t.Buffer())
     if b.DataLength():
      assert t.Type()==schema.TensorType.INT8;q=t.Quantization();assert q.ScaleLength()==1 and q.Scale(0)>0
      assert np.all(b.DataAsNumpy().view(np.int8)==q.ZeroPoint(0))
      initializers.append(dict(shape=t.ShapeAsNumpy().tolist(),scale=float(q.Scale(0)),zero=int(q.ZeroPoint(0))))
   assert len(initializers)==6
   codes=np.empty((4,2,256),np.uint8);dequant=np.empty((4,2,256),np.float64);quant=None
   for case in range(4):
    for rep in range(2):
     it=Interpreter(model_content=data,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
     assert a['dtype']==np.int8 and b['dtype']==np.uint8 and list(a['shape'])==[1,3,40] and list(b['shape'])==[1,1]
     scale,zero=a['quantization'];os,oz=b['quantization'];assert scale>0 and os>0;quant=dict(input=[float(scale),int(zero)],output=[float(os),int(oz)])
     for step,value in enumerate(inputs[case]):
      q=np.clip(np.rint(value/scale)+zero,-128,127).astype(np.int8);it.set_tensor(a['index'],q);it.invoke();code=int(it.get_tensor(b['index'])[0,0]);codes[case,rep,step]=code;dequant[case,rep,step]=(code-oz)*os
   with np.load(m['references_path'],allow_pickle=False) as refs:original=refs['tf'].copy()
   np.savez(folder/'references.npz',uint8=codes,dequantized=dequant,tf=original);cases=[]
   for i in range(4):
    error=np.abs(np.convolve(dequant[i,0]-original[i,0].astype(float),np.ones(5)/5,mode='valid'));stats=dict(mean=float(error.mean()),p99=float(np.percentile(error,99)),maximum=float(error.max()))
    cases.append(dict(stream=i,five_step_abs_error=stats,passed=stats['mean']<=.02 and stats['p99']<=.1 and stats['maximum']<=.25))
   repeat=codes[:,0].tobytes()==codes[:,1].tobytes();finite=bool(np.isfinite(dequant).all() and np.isfinite(original).all());assert all(a.tobytes()==v.numpy().tobytes() for a,v in zip(before,model.variables,strict=True))
   row=dict(target=m['target'],condition=m['condition'],h5_sha256=m['h5_sha256'],model_sha256=sha(path),model_bytes=len(data),references_sha256=sha(folder/'references.npz'),ops=ops,initializers=initializers,quantization=quant,repeat_exact=repeat,finite=finite,parameters_unchanged=True,cases=cases,passed=repeat and finite and all(c['passed'] for c in cases));rows.append(row)
   (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),models=rows,all_passed=all(r['passed'] for r in rows),limits=p['contract']),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
