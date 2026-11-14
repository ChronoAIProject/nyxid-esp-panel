"""Save new pooled initialization, then test synthetic TF streaming/phase only."""
import argparse,hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-pooled-stream-phase-execution-v1.json';OUT=Path('.local/wakeword/pooled-stream-phase-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-pooled-head-feasibility-plan-v1.json';p=json.loads(parent.read_text());inp=Path('.local/wakeword/fp32-internal-stage-intake-v1/inputs.bin')
 files=dict(p['files'])
 for f in [parent,Path(__file__),inp]+list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py')):files[str(f)]=sha(f)
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,models=p['models'],inputs=str(inp),seeds=[202611140+i for i in range(4)],versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy']},max_abs=1e-4,contract='New average-pool head initialization saved before first forward, original encoder/BN copied exactly. All4 original H5, same4synthetic inputs x256steps, all startup, two TF state resets. Independent offline model loaded from saved initialization; phase224+zero against stream74:, nominal225 retained. No export/optimizer/audio/recognition or fresh data. Failure preserved, no seed/head selection/retry.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import numpy as np
  import probe_hey_context_transfer_shapes_v1 as helper
  from microwakeword import mixednet,utils
  from microwakeword.layers import modes
  tf=helper.tf
  def create():
   parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
   return mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None)
  inputs=np.fromfile(p['inputs'],dtype='<f4').reshape(4,256,1,3,40);rows=[]
  for m,seed in zip(p['models'],p['seeds'],strict=True):
   tf.keras.utils.set_random_seed(seed);folder=OUT/(m['target']+'-'+m['condition']);folder.mkdir()
   original=helper.create(225);original.load_weights(m['path']);model=create()
   weighted=lambda net:[l for l in net.layers if l.weights and not isinstance(getattr(l,'cell',l),tf.keras.layers.Dense)]
   for a,b in zip(weighted(original),weighted(model),strict=True):
    assert type(getattr(a,'cell',a))==type(getattr(b,'cell',b));b.set_weights(a.get_weights())
    assert all(x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(a.get_weights(),b.get_weights(),strict=True))
   for l in model.layers:
    if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
   weights=folder/'initial.weights.h5';model.save_weights(weights)
   before=[v.numpy().copy() for v in model.variables]
   inventory=[dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for v in before]
   (folder/'initialization.json').write_text(json.dumps(dict(seed=seed,derived_original_h5=m['sha256'],h5_sha256=sha(weights),inventory=inventory,new_classifier=True),indent=2)+'\n')
   offline=create();offline.load_weights(weights)
   for l in offline.layers:
    if isinstance(l,tf.keras.layers.BatchNormalization):l.trainable=False
   stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
   stream(np.zeros((1,3,40),np.float32),training=False);states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
   window=tf.function(lambda x:offline(x,training=False),input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
   output=np.empty((4,2,256),np.float32);phases=[];nominal=[]
   for case in range(4):
    for rep in range(2):
     for s in states:s.assign(tf.zeros_like(s))
     for j,x in enumerate(inputs[case]):output[case,rep,j]=float(infer(x).numpy()[0,0])
    values=inputs[case].reshape(768,40)
    phases.append([float(window(np.concatenate((values[e-224:e],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for e in range(225,769,3)])
    nominal.append([float(window(values[None,e-225:e]).numpy()[0,0]) for e in range(225,769,3)])
   phases=np.asarray(phases,np.float32);nominal=np.asarray(nominal,np.float32)
   path=folder/'references.npz';np.savez(path,tf=output,offline_phase=phases,offline_nominal=nominal)
   assert all(a.tobytes()==v.numpy().tobytes() for a,v in zip(before,model.variables,strict=True))
   gap=float(np.max(np.abs(phases.astype(float)-output[:,0,74:].astype(float))));repeat=output[:,0].tobytes()==output[:,1].tobytes();finite=all(np.isfinite(x).all() for x in [output,phases,nominal])
   row=dict(target=m['target'],condition=m['condition'],seed=seed,h5_path=str(weights),h5_sha256=sha(weights),references_path=str(path),references_sha256=sha(path),phase_max_abs=gap,repeat_exact=repeat,finite=bool(finite),parameters_unchanged=True,passed=bool(finite and repeat and gap<=p['max_abs']));rows.append(row)
   (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),cases=rows,all_passed=all(r['passed'] for r in rows),limits=p['contract']),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
