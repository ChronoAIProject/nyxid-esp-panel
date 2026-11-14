"""Once-only raw sigmoid internal-state FP32 diagnostic, four fixed trained H5s."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-fp32-internal-software-execution-v1.json';OUT=Path('.local/wakeword/fp32-internal-software-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 source=R/'4090-fp32-original-paired-route-prospective-v1.json';models=json.loads(source.read_text())['models'];inp=Path('.local/wakeword/fp32-internal-stage-intake-v1/inputs.bin')
 paths=[Path(__file__),source,inp,Path('research/wakeword_p4/experiments/probe_hey_context_transfer_shapes_v1.py'),Path('research/wakeword_p4/training/gpu_python.py')]+list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))+[Path(m['path']) for m in models]
 for m in models:assert sha(m['path'])==m['sha256']
 assert sha(inp)=='0f7eff41940be4149be682f9a24755d42646bbe08e211c5fbdd1638059d3c9ac'
 d=dict(files={str(p):sha(p) for p in paths},models=models,inputs=str(inp),versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},device='CPU, original helper two intra-op threads; GPU hidden',contract='ALL4 original final10000 raw Dense sigmoid, unchanged arrays/BN, six real-zero internal states. FP32 builtin-only, no optimization/representative data. Four original synthetic feature streams x256 x2 fresh/reset repeats. Finite/repeat byte exact, probability maxabs<=1e-4. Save all startup/results, failure stops no retry/config selection. No old135, recognition, freshdev or acceptance. Internal states are not external ABI/state-output validation.',retries=0)
 with LOCK.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import numpy as np
  import probe_hey_context_transfer_shapes_v1 as helper
  from microwakeword import utils
  from microwakeword.layers import modes
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  from tensorflow.lite.python import schema_py_generated as schema
  tf=helper.tf;inputs=np.fromfile(p['inputs'],dtype='<f4').reshape(4,256,1,3,40);rows=[]
  names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
  for m in p['models']:
   folder=OUT/(m['target']+'-'+m['condition']);folder.mkdir();model=helper.create(225);model.load_weights(m['path'])
   for layer in model.layers:
    cell=getattr(layer,'cell',layer)
    if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
   original=[v.numpy().copy() for v in model.variables]
   inventory=[dict(name=v.name,shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for v,a in zip(model.variables,original)]
   stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
   states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   for s in states:s.assign(tf.zeros_like(s))
   state_inventory=[dict(name=s.name,shape=list(s.shape),all_zero=bool(np.all(s.numpy()==0))) for s in states];assert all(s['all_zero'] for s in state_inventory)
   stream.export(str(folder/'saved_model'));conv=tf.lite.TFLiteConverter.from_saved_model(str(folder/'saved_model'));conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS];data=conv.convert();modelpath=folder/'model.tflite';modelpath.write_bytes(data)
   g=schema.Model.GetRootAsModel(data,0);sub=g.Subgraphs(0);assert sub.InputsLength()==sub.OutputsLength()==1
   for idx in [sub.Inputs(0),sub.Outputs(0)]:assert sub.Tensors(idx).Type()==schema.TensorType.FLOAT32
   initializers=[]
   for si in range(1,g.SubgraphsLength()):
    sg=g.Subgraphs(si)
    for ti in range(sg.TensorsLength()):
     t=sg.Tensors(ti);b=g.Buffers(t.Buffer())
     if b.DataLength() and t.Type()==schema.TensorType.FLOAT32:
      a=b.DataAsNumpy().view('<f4');assert np.all(a==0);initializers.append(dict(shape=t.ShapeAsNumpy().tolist(),bytes=int(b.DataLength()),all_zero=True))
   assert len(initializers)==6
   ops=[names[g.OperatorCodes(sub.Operators(i).OpcodeIndex()).BuiltinCode()] for i in range(sub.OperatorsLength())];assert not any(x in ops for x in ['QUANTIZE','DEQUANTIZE','CUSTOM'])
   tfout=np.empty((4,2,256),np.float32);liteout=np.empty_like(tfout)
   infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
   for case in range(4):
    for rep in range(2):
     for s in states:s.assign(tf.zeros_like(s))
     it=Interpreter(model_content=data,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();ii=it.get_input_details()[0]['index'];oi=it.get_output_details()[0]['index']
     for step,x in enumerate(inputs[case]):
      tfout[case,rep,step]=np.asarray(infer(x)).reshape(-1)[0];it.set_tensor(ii,x);it.invoke();liteout[case,rep,step]=it.get_tensor(oi).reshape(-1)[0]
   np.savez(folder/'references.npz',tf=tfout,litert=liteout)
   finite=bool(np.isfinite(tfout).all() and np.isfinite(liteout).all());repeat=all(a[:,0].tobytes()==a[:,1].tobytes() for a in [tfout,liteout]);gap=float(np.max(np.abs(tfout.astype(float)-liteout.astype(float))))
   unchanged=all(np.array_equal(a,v.numpy()) for a,v in zip(original,model.variables));assert unchanged
   row=dict(target=m['target'],condition=m['condition'],h5_sha256=m['sha256'],inventory=inventory,states=state_inventory,initializers=initializers,ops=ops,model_sha256=sha(modelpath),references_sha256=sha(folder/'references.npz'),finite=finite,repeat_exact=repeat,max_abs=gap,passed=finite and repeat and gap<=1e-4,parameters_unchanged=unchanged);rows.append(row);(folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['target','condition','max_abs','passed']}),flush=True)
  result=dict(execution_sha256=sha(LOCK),cases=rows,all_passed=all(r['passed'] for r in rows),limits=p['contract']);(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
