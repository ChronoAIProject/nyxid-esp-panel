"""Actual float streaming on all380 already-consumed development clips, fixed threshold."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
from evaluate import Detector,smooth,event_indices
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');root=base/'model-gpu-v42-ami-extra'
assert sha(root/'hey_nyx.tflite')=='ff15ebf95001433135514babe0929bb75afa79d0e2a9755e0d61d9a6c7d81272'
loaded=tf.saved_model.load(str(root/'saved_model'));signature=loaded.signatures['serving_default'];key=next(iter(signature.structured_input_signature[1]));states=[v for v in loaded.variables if 'states' in v.name];assert states
threshold=.992;detector=Detector(root/'hey_nyx.tflite');arrays={};details=[]
protocols=[('phone','synthetic-phone-development-replay-v1','2ff49bdd98b7b8ccaa2d01bef9f8764db331ad2a08c98ace522a05df48763dc8',80),('vctk','vctk-development-replay-v1','779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be',300)]
for tag,directory,digest,count in protocols:
 manifest=base/directory/'manifest.json';assert sha(manifest)==digest
 rows=json.loads(manifest.read_text())['records'];assert len(rows)==count
 previous=base/f'v42-{tag}-development-historical-threshold-v1';report=json.loads((previous/'report.json').read_text());assert report['model_sha256']==sha(root/'hey_nyx.tflite') and not report['final_data_used']
 for i,(row,old) in enumerate(zip(rows,report['records'])):
  assert row==old['record'] and row['split']=='validation' and row['role']=='development-paired-validation'
  path=manifest.parent/row['path'];assert sha(path)==row['sha256']
  pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
  raw=features(pcm);x=np.clip(raw.astype(np.float32)/25.6,0,26)
  detector.reset();q=smooth(detector.predict(raw),5)
  assert sha(previous/f'{i}-scores.npy')==old['nyx']['score_sha256'];np.testing.assert_array_equal(q,np.load(previous/f'{i}-scores.npy'))
  for state in states:state.assign(tf.zeros_like(state))
  f=[]
  for at in range(0,len(x)-2,3):f.append(float(next(iter(signature(**{key:tf.convert_to_tensor(x[None,at:at+3])}).values())).numpy().reshape(-1)[0]))
  f=smooth(np.asarray(f,dtype=np.float32),5);assert f.shape==q.shape
  arrays[f'{tag}_{i}']=f
  details.append(dict(protocol=tag,index=i,record=row,float_events=event_indices(f,threshold),int8_events=event_indices(q,threshold),float_max_five=float(f.max()),int8_max_five=float(q.max()),max_smoothed_error=float(np.abs(f-q).max())))
  if i%25==0:print(json.dumps(dict(protocol=tag,completed=i+1,total=count)),flush=True)
out=base/'v42-old-development-float-probe-v1';out.mkdir(exist_ok=False);np.savez_compressed(out/'scores.npz',**arrays)
summary={}
for tag,_,_,_ in protocols:
 rows=[d for d in details if d['protocol']==tag];summary[tag]={}
 for mode in ['float','int8']:
  positive=[d for d in rows if d['record']['target']=='hey_nyx'];negative=[d for d in rows if d['record']['target']!='hey_nyx']
  summary[tag][mode]=dict(target_any=sum(bool(d[mode+'_events']) for d in positive),positive_clips=len(positive),negative_false_clips=sum(bool(d[mode+'_events']) for d in negative),negative_clips=len(negative))
r=dict(scope='All380 consumed old development clips; actual exported float vs authenticated INT8, fixed historical threshold',model_sha256=sha(root/'hey_nyx.tflite'),threshold=threshold,scores_sha256=sha(out/'scores.npz'),script_sha256=sha(Path(__file__)),all_int8_traces_exact=True,summary=summary,details=details,limits='Diagnostic only; no threshold selection, new/final data, training or promotion. Actual float vs INT8 includes quantization effects; synthetic boundaries are process references, not phonetic truth. No new FAR evaluation.')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(summary),flush=True)
