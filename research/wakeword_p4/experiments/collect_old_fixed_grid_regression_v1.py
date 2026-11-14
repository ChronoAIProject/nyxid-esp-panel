"""Collect rejected-model old consumed development regression, no threshold selection."""
import argparse,json,hashlib,subprocess,importlib.metadata
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments');P=R/'4090-old-fixed-grid-regression-plan-v1.json';OUT=Path('.local/wakeword/old-fixed-grid-regression-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 prior=R/'4090-hey-broad-fixed-grid-screen-execution-v1.json';old=json.loads(prior.read_text());files=dict(old['files']);synthetic=Path('.local/wakeword/hey-development-replay-v1.json');background=Path('.local/wakeword/hey-paired-earnings-development-v1/manifest.json');models=json.loads((R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json').read_text())['arms'];arms=json.loads((R/'4090-hey-broad-fixed-grid-compile-plan-v1.json').read_text())['arms'];entries=[]
 for m,a in zip(models,arms,strict=True):
  assert (m['target'],m['condition'])==(a['target'],a['condition']);files[m['path']]=m['sha256'];files[a['weights']]=m['final_h5_sha256'];entries.append(dict(id=m['target']+'-'+m['condition'],target=m['target'],model=m['path'],model_sha256=m['sha256'],weights=a['weights'],weights_sha256=m['final_h5_sha256']))
 v30=Path('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite');assert v30.stat().st_size==54440 and sha(v30)=='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c';files[str(v30)]=sha(v30);entries.append(dict(id='v30',target='hey_nyx',model=str(v30),model_sha256=sha(v30),weights=None))
 assert len(json.loads(synthetic.read_text())['records'])==360 and len(json.loads(background.read_text())['records'])==12
 consumed=[];historical=R/'4090-hey-development-custom-execution-v1.json'
 for identifier in ['hey_willow','hey_nyx','v30']:
  receipt=Path('.local/wakeword/hey-development-custom-streams-v1')/identifier/'report.json';evidence=json.loads(receipt.read_text());assert evidence['execution_sha256']==sha(historical)
  for phase,manifest in [('synthetic',synthetic),('background',background)]:
   assert evidence['phases'][phase]['manifest_sha256']==sha(manifest)
   assert [r['record'] for r in evidence['phases'][phase]['records']]==json.loads(manifest.read_text())['records']
  files[str(receipt)]=sha(receipt);consumed.append(dict(path=str(receipt),sha256=sha(receipt)))
 for phase,manifest in [('synthetic',synthetic),('background',background)]:
  for row in json.loads(manifest.read_text())['records']:
   p=Path(row['path'] if phase=='synthetic' else row['wav_path']);h=row['sha256'] if phase=='synthetic' else row['wav_sha256'];assert sha(p)==h;files[str(p)]=h
   if phase=='background':assert sha(row['frontend_path'])==row['frontend_sha256'];files[row['frontend_path']]=row['frontend_sha256']
 for p in [prior,synthetic,background,Path(__file__),E/'old_fixed_grid_regression_v1.md',E/'old_regression_events_v1.py',E/'reduce_old_fixed_grid_regression_v1.py',E/'hey_development_metrics_v1.py',E/'uint8_event_policy_v1.py',E/'complete_feature_stream_v1.py',E/'probe_hey_broad_bounded_float_v1.py',R/'4090-old-regression-events-check-v1.json',R/'4090-hey-development-custom-execution-v1.json',R/'4090-hey-broad-fixed-grid-screen-independent-audit-v1.json']:
  files[str(p)]=sha(p)
 assert json.loads((R/'4090-old-regression-events-check-v1.json').read_text())['all_exact']
 with P.open('x') as f:json.dump(dict(files=files,versions=old['versions'],models=entries,synthetic_manifest=str(synthetic),background_manifest=str(background),role='reused-development-regression',prior_consumed_reports=consumed,policy='Collect original360+12 completeinputs, old4rejectedmodels float+int and v30int only, twicefresh exact, no threshold selection/newdev/acceptance/promotion. Original artifacts unchanged.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 for name,version in lock['versions'].items():assert importlib.metadata.version(name)==version
 import probe_hey_context_transfer_shapes_v1 as helper
 from probe_hey_broad_bounded_float_v1 import bounded
 from microwakeword import utils
 from microwakeword.layers import modes
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 import soundfile as sf,sys
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 tf=helper.tf;OUT.mkdir(exist_ok=False);records=[]
 for entry in lock['models']:
  folder=OUT/entry['id'];folder.mkdir();stream=None
  if entry['weights'] is not None:
   model=helper.create(225);model.load_weights(entry['weights'])
   for layer in model.layers:
    if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
   stream=utils.to_streaming_inference(bounded(model,tf),{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False);states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
   infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
  for phase in ['synthetic','background']:
   sources=json.loads(Path(lock[phase+'_manifest']).read_text())['records']
   for index,row in enumerate(sources):
    wav=row['path'] if phase=='synthetic' else row['wav_path'];assert sha(wav)==lock['files'][wav]
    if phase=='synthetic':
     pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256'];raw=features(pcm)
    else:
     assert sha(row['frontend_path'])==row['frontend_sha256'];raw=np.load(row['frontend_path'],allow_pickle=False)
    assert raw.dtype==np.uint16 and raw.shape==(max(0,(row['samples']+159)//160-2),40)
    values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);runs=[]
    for repeat in range(2):
     i=Interpreter(model_path=entry['model'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();inp=i.get_input_details()[0];out=i.get_output_details()[0];sc,z=inp['quantization'];assert out['quantization']==(1/256,0) and inp['dtype']==np.int8 and out['dtype']==np.uint8
     q=np.clip(np.rint(values/sc)+z,-128,127).astype(np.int8);integer=[];floating=[]
     if stream is not None:
      for state in states:state.assign(tf.zeros_like(state))
     for at in range(0,len(q),3):
      i.set_tensor(inp['index'],q[None,at:at+3]);i.invoke();integer.append(int(i.get_tensor(out['index'])[0,0]))
      if stream is not None:floating.append(float(infer(values[None,at:at+3]).numpy()[0,0]))
     runs.append(dict(uint8=np.asarray(integer,np.uint8),floating=np.asarray(floating,np.float32)))
    path=folder/f'{phase}-{index:03}.npz';clocks=np.minimum((3*np.arange(1,len(runs[0]['uint8'])+1,dtype=np.int64)+2)*160,row['samples']);np.savez_compressed(path,uint8=runs[0]['uint8'],uint8_repeat=runs[1]['uint8'],floating=runs[0]['floating'],float_repeat=runs[1]['floating'],sample_end=clocks)
    assert all(np.array_equal(runs[0][k],runs[1][k]) for k in runs[0]) and np.isfinite(runs[0]['floating']).all()
    receipt=dict(model=entry['id'],target=entry['target'],phase=phase,index=index,source_id=row['source_id'],samples=row['samples'],raw_frames=len(raw),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),final_feature_padding=(-len(raw))%3,steps=len(clocks),repeat_exact=True,arrays_path=str(path),arrays_sha256=sha(path));records.append(receipt)
    with (OUT/'records.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    print(json.dumps(dict(model=entry['id'],phase=phase,completed=index+1)),flush=True)
 assert len(records)==5*372
 with (R/'4090-old-fixed-grid-regression-collection-v1.json').open('x') as f:json.dump(dict(plan_sha256=sha(P),records=records,limits=lock['policy']),f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
