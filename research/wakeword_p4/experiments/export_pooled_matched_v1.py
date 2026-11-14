"""Fixed-final matched flat/pool raw sigmoid exports; no score-based retries."""
import argparse,hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-pooled-matched-export-execution-v1.json';OUT=Path('.local/wakeword/pooled-matched-export-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 formal=R/'4090-pooled-matched-formal-execution-v1.json';result=R/'4090-pooled-matched-formal-result-v1.json';audit=R/'4090-pooled-matched-formal-independent-audit-v1.json'
 p=json.loads(formal.read_text());r=json.loads(result.read_text());assert r['all_passed'] and json.loads(audit.read_text())['all_passed'] and r['execution_sha256']==sha(formal)
 source=R/'4090-hey-broad-paired-qat-export-execution-v1.json';material=json.loads(source.read_text())['material'];files=dict(p['files']);weights=[]
 paths=[formal,result,audit,source,Path(__file__)]
 for c in r['cases']:
  receipt=c['final_repeat'];assert c['steps']==receipt['steps']==10000
  path=Path(receipt['weights_path']);assert sha(path)==receipt['weights_sha256'];paths.append(path)
  weights.append(dict(target=c['target'],condition=c['condition'],path=str(path),sha256=sha(path)))
 assert {(w['target'],w['condition']) for w in weights}=={(t,c) for t in ['hey_willow','hey_nyx'] for c in ['flat','pool']}
 for m in material:assert sha(m['path'])==m['sha256'];paths.append(Path(m['path']))
 for f in paths:files[str(f)]=sha(f)
 for f,h in files.items():assert sha(f)==h,f
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,weights=weights,material=material,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},retries=0,contract='All4fixedfinal10000 flat/pool raw Dense sigmoid unchanged, originalBN frozen. One FP32 and one builtinINT8 export/model, zero6states before saving. Representative originaltarget100training arrays/order plus originalzero/26, no intermediate reset. No bound/equalization/head rewrite, newdata or checkpointselection. Static ABI/init gate only; trained full540 probability/phase numerical and runtime/reference, recognition/FAR gates separate. Coarse logit scales reported not tuned. Preserve all failures, no retry/calibration selection.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import probe_hey_context_transfer_shapes_v1 as helper
  from microwakeword import mixednet,utils
  from microwakeword.layers import modes
  from tensorflow.lite.python import schema_py_generated as schema
  tf=helper.tf;names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)};rows=[]
  for weight in p['weights']:
   folder=OUT/(weight['target']+'-'+weight['condition']);folder.mkdir()
   if weight['condition']=='flat':model=helper.create(225)
   else:
    parser=argparse.ArgumentParser();mixednet.model_parameters(parser);model=mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None)
   model.load_weights(weight['path'])
   for l in model.layers:
    if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
   before=[v.numpy().copy() for v in model.variables]
   stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
   states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   for s in states:s.assign(tf.zeros_like(s))
   stream.export(str(folder/'saved_model'));artifacts={}
   for kind in ['fp32','int8']:
    converter=tf.lite.TFLiteConverter.from_saved_model(str(folder/'saved_model'))
    if kind=='int8':
     material=next(m for m in p['material'] if m['target']==weight['target']);a=np.load(material['path'],allow_pickle=False,mmap_mode='r');assert list(a.shape)==material['shape']
     def representative():
      yield [np.zeros((1,3,40),np.float32)];yield [np.full((1,3,40),26,np.float32)]
      for window in a:
       for j in range(0,225,3):yield [np.array(window[:,j:j+3],copy=True)]
     converter.optimizations=[tf.lite.Optimize.DEFAULT];converter._experimental_variable_quantization=True;converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.uint8;converter.representative_dataset=representative
    else:converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    data=converter.convert();path=folder/(kind+'.tflite');path.write_bytes(data);g=schema.Model.GetRootAsModel(data,0);sub=g.Subgraphs(0)
    ops=[names[g.OperatorCodes(sub.Operators(i).OpcodeIndex()).BuiltinCode()] for i in range(sub.OperatorsLength())];assert 'CUSTOM' not in ops
    assert sub.InputsLength()==sub.OutputsLength()==1
    assert sub.Tensors(sub.Inputs(0)).ShapeAsNumpy().tolist()==[1,3,40] and sub.Tensors(sub.Outputs(0)).ShapeAsNumpy().tolist()==[1,1]
    initializers=[]
    for si in range(1,g.SubgraphsLength()):
     sg=g.Subgraphs(si)
     for ti in range(sg.TensorsLength()):
      t=sg.Tensors(ti);buf=g.Buffers(t.Buffer())
      if buf.DataLength():
       if kind=='int8':
        assert t.Type()==schema.TensorType.INT8;q=t.Quantization();zero=int(q.ZeroPoint(0));assert np.all(buf.DataAsNumpy().view(np.int8)==zero)
       else:assert t.Type()==schema.TensorType.FLOAT32 and np.all(buf.DataAsNumpy().view('<f4')==0);zero=0
       initializers.append(dict(shape=t.ShapeAsNumpy().tolist(),zero=zero,all_real_zero=True))
    assert len(initializers)==6
    logit_scale=None
    if kind=='int8':
     assert sub.Tensors(sub.Inputs(0)).Type()==schema.TensorType.INT8 and sub.Tensors(sub.Outputs(0)).Type()==schema.TensorType.UINT8
     assert all(sub.Tensors(i).Type()!=schema.TensorType.FLOAT32 for i in range(sub.TensorsLength())) and 'DEQUANTIZE' not in ops
     logistic,=[sub.Operators(i) for i in range(sub.OperatorsLength()) if names[g.OperatorCodes(sub.Operators(i).OpcodeIndex()).BuiltinCode()]=='LOGISTIC'];logit_scale=float(sub.Tensors(logistic.Inputs(0)).Quantization().Scale(0))
    else:assert sub.Tensors(sub.Inputs(0)).Type()==sub.Tensors(sub.Outputs(0)).Type()==schema.TensorType.FLOAT32
    artifacts[kind]=dict(path=str(path),bytes=len(data),sha256=sha(path),ops=ops,initializers=initializers,logistic_input_scale=logit_scale)
   assert all(a.tobytes()==v.numpy().tobytes() for a,v in zip(before,model.variables,strict=True))
   row=dict(**weight,artifacts=artifacts,parameters_unchanged=True,static_abi_passed=True);rows.append(row);(folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),models=rows,all_static_passed=len(rows)==4,limits=p['contract']),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
