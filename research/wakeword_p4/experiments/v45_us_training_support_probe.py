"""All324 US training variants: actual exported float/INT8 full-stream diagnostic."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import Detector,smooth,event_indices
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/model-gpu-v45-ami-extra');pack=Path('.local/wakeword/us-training-features-v1');out=Path('.local/wakeword/v45-us-training-support-v1')
assert sha(root/'hey_nyx.tflite')=='f5d175a0172d4fde70dce746fa358331c3fcdf590139c2e76113fe14c45e30af'
assert sha(pack/'report.json')=='e73f3c3cb95b2c97a079a5cf4b4fe75d7289f588b7febb42947b37631f355f0c'
report=json.loads((pack/'report.json').read_text());assert report['included']==324 and sha(pack/'training.npz')==report['pack_sha256']
loaded=tf.saved_model.load(str(root/'saved_model'));signature=loaded.signatures['serving_default'];key=next(iter(signature.structured_input_signature[1]));states=[v for v in loaded.variables if 'states' in v.name];assert states
out.mkdir(exist_ok=False);detector=Detector(root/'hey_nyx.tflite');arrays={};details=[];thresholds=[.5,.992,.99609375]
for i,r in enumerate(report['records']):
 assert r['pack_index']==i and r['entire_signal_fits'] and r['split']=='training'
 raw=np.load(pack/r['raw'],allow_pickle=False);assert sha(pack/r['raw'])==r['raw_file_sha256'] and hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
 x=np.clip(raw.astype(np.float32)/25.6,0,26);detector.reset();q=detector.predict(raw)
 for state in states:state.assign(tf.zeros_like(state))
 f=[]
 for at in range(0,len(x)-2,3):f.append(float(next(iter(signature(**{key:tf.convert_to_tensor(x[None,at:at+3])}).values())).numpy().reshape(-1)[0]))
 f=np.asarray(f,np.float32);assert f.shape==q.shape and np.isfinite(f).all() and np.isfinite(q).all()
 arrays[f'float_{i}']=f;arrays[f'int8_{i}']=q
 first=150+3*np.arange(len(q[49:])-4);last=first+12
 valid=((first-150)*160<=32000)&((last-150)*160<=32000)&((first+1)*160>=32000+r['signal_samples'])
 modes={}
 for mode,scores in [('float',f),('int8',q)]:
  five=np.convolve(scores[49:].astype(np.float64),np.ones(5)/5,mode='valid');assert len(five)==len(valid)
  modes[mode]=dict(score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),full_max_five=float(five.max()),thresholds={str(t):dict(full_high=bool(five.max()>=t),support_high_positions=int(np.count_nonzero(five[valid]>=t)),event_indices=event_indices(smooth(scores,5),t)) for t in thresholds})
 details.append(dict(index=i,source=r['source'],voice=r['voice'],target=r['target'],variant=r['variant'],label=int(r['target']=='hey_nyx'),valid_support_positions=int(valid.sum()),modes=modes))
 if i%25==0:print(json.dumps(dict(completed=i+1,total=324)),flush=True)
np.savez_compressed(out/'scores.npz',**arrays);groups={}
for d in details:
 for group in ['all/'+str(d['label']),d['voice']+'/'+d['target']]:
  g=groups.setdefault(group,dict(variants=0,zero_support=0,modes={}));g['variants']+=1;g['zero_support']+=int(d['valid_support_positions']==0)
  for mode,v in d['modes'].items():
   for t,h in v['thresholds'].items():
    s=g['modes'].setdefault(mode+'/'+t,dict(full_high=0,support_hit_variants=0,support_high_positions=0,event_clips=0));s['full_high']+=int(h['full_high']);s['support_hit_variants']+=int(h['support_high_positions']>0);s['support_high_positions']+=h['support_high_positions'];s['event_clips']+=int(bool(h['event_indices']))
r=dict(scope='All324UStraining variants; actual exported float and fullINT8, descriptive fixed thresholds only',model_sha256=sha(root/'hey_nyx.tflite'),training_sha256=sha(root/'training.json'),pack_sha256=report['pack_sha256'],pack_report_sha256=sha(pack/'report.json'),script_sha256=sha(Path(__file__)),scores_sha256=sha(out/'scores.npz'),groups=groups,details=details,limits='Reused training fit diagnostic, not validation or phonetic truth. Container-support ledger only, includes native silence and RIR tail. No optimization, threshold selection, new/final data or promotion.')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:groups[k] for k in ['all/0','all/1']}),flush=True)
