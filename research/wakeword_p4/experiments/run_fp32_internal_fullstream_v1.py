"""Full540 internal raw FP32 numerical stage; execution lock requires Mac evidence."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-fp32-internal-fullstream-execution-v1.json';OUT=Path('.local/wakeword/fp32-internal-fullstream-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def published(p):assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)])==Path(p).read_bytes()
def lock(mac_report):
 # Caller must first authenticate and import the actual Mac run, not a plan.
 mp=Path(mac_report);published(mp);mac=json.loads(mp.read_text())
 assert mac['actual_trained_microinterpreter_run'] is True and mac['all_passed'] is True
 scope=R/'4090-fp32-internal-fullstream-prospective-v1.json';published(scope);s=json.loads(scope.read_text());software=R/'4090-fp32-internal-software-execution-v1.json';p=json.loads(software.read_text());result=R/'4090-fp32-internal-software-original-result-v1.json';r=json.loads(result.read_text());assert r['all_passed']
 from collect_complete_development_v1 import runtime_identity_files
 for f,h in s['files'].items():assert sha(f)==h,f
 for f,h in p['files'].items():assert sha(f)==h,f
 for row in s['ordered_requests']:assert sha(row['path'])==row['sha256'],row['path']
 files=dict(p['files'])
 paths=[mp,scope,software,result,Path(__file__),Path('research/wakeword_p4/training/features.py'),Path('research/wakeword_p4/experiments/collect_complete_development_v1.py'),Path('research/wakeword_p4/experiments/audit_fp32_fullstream_arrays_v1.py')]+runtime_identity_files()+[Path(x['path']) for x in s['ordered_requests']]
 exports=[]
 for m,x in zip(p['models'],r['cases'],strict=True):
  path=Path('.local/wakeword/fp32-internal-software-v1')/(m['target']+'-'+m['condition'])/'model.tflite';assert sha(path)==x['model_sha256'];paths.append(path);exports.append(dict(target=m['target'],condition=m['condition'],path=str(path),sha256=sha(path)))
 for path in paths:files[str(path)]=sha(path)
 d=dict(files=files,models=p['models'],exports=exports,requests=s['ordered_requests'],versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert','soundfile','pymicro-features']},mac_report=str(mp),max_abs=1e-4,retries=0)
 with LOCK.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def run():
 published(LOCK);p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import numpy as np,soundfile as sf
  import probe_hey_context_transfer_shapes_v1 as helper
  from microwakeword import utils
  from microwakeword.layers import modes
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
  tf=helper.tf;cases=[]
  for model_spec,export in zip(p['models'],p['exports'],strict=True):
   assert (model_spec['target'],model_spec['condition'])==(export['target'],export['condition']);models=[]
   for _ in range(2):
    model=helper.create(225);model.load_weights(model_spec['path'])
    for layer in model.layers:
     cell=getattr(layer,'cell',layer)
     if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
    models.append(model)
   model,offline=models;stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False);states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)]);window=tf.function(lambda x:offline(x,training=False),input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
   for index,row in enumerate(p['requests']):
    pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
    raw=features(pcm);values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);tfout=[];lite=[]
    for rep in range(2):
     for state in states:state.assign(tf.zeros_like(state))
     tfout.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)],np.float32))
     it=Interpreter(model_path=export['path'],num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0];assert list(a['shape'])==[1,3,40] and a['dtype']==np.float32 and list(b['shape'])==[1,1] and b['dtype']==np.float32;scores=[]
     for j in range(0,len(values),3):it.set_tensor(a['index'],values[None,j:j+3]);it.invoke();scores.append(float(it.get_tensor(b['index'])[0,0]))
     lite.append(np.asarray(scores,np.float32))
    phase=np.asarray([float(window(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for end in range(225,len(values)+1,3)],np.float32);nominal=np.asarray([float(window(values[None,end-225:end]).numpy()[0,0]) for end in range(225,len(values)+1,3)],np.float32)
    gap=float(np.max(np.abs(tfout[0].astype(float)-lite[0].astype(float))));pe=float(np.max(np.abs(phase.astype(float)-tfout[0][74:].astype(float))));repeat=tfout[0].tobytes()==tfout[1].tobytes() and lite[0].tobytes()==lite[1].tobytes();finite=all(np.isfinite(x).all() for x in [*tfout,*lite,phase,nominal]);clocks=(3*np.arange(1,len(tfout[0])+1)+2)*160;path=OUT/(model_spec['target']+'-'+model_spec['condition']+f'-{index:03}.npz')
    np.savez(path,raw=raw,input=values,tf=tfout[0],tf_repeat=tfout[1],litert=lite[0],litert_repeat=lite[1],offline_phase=phase,offline_nominal=nominal,support_unclamped=clocks,sample_end=np.minimum(clocks,len(pcm)))
    c=dict(target=model_spec['target'],condition=model_spec['condition'],request_order=row['order'],index=index,path=str(path),sha256=sha(path),max_abs=gap,phase_max_abs=pe,repeat_exact=repeat,finite=finite,passed=bool(repeat and finite and max(gap,pe)<=p['max_abs']));cases.append(c)
    with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(c)+'\n')
    print(json.dumps(c),flush=True)
  assert len(cases)==540;(OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),cases=cases,all_passed=all(c['passed'] for c in cases),limits='TF/LiteRT fullstream numerical stage only. Actual fullstream MicroInterpreter/independent array audit still required. No recognition or promotion.'),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
 if sys.argv[1]=='lock':lock(sys.argv[2])
 elif sys.argv[1]=='run':run()
 else:raise ValueError('lock MAC_REPORT or run')
