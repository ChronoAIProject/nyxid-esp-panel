"""Frozen v46 all-source factorial stream diagnostic after exact control audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import Detector,smooth,event_indices
base=Path('.local/wakeword');root=base/'balanced-rendering-factorial-v1';out=base/'balanced-rendering-factorial-v1-scores';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
auditpath=Path('research/wakeword_p4/results/4090-balanced-rendering-controls-audit-v1.json');audit=json.loads(auditpath.read_text());assert audit['verified']==16128 and audit['all_arrays_pcm_frontend_exact'] and audit['all_complete_containers'];assert sha(root/'controls.json')==audit['controls_report_sha256']
controls=json.loads((root/'controls.json').read_text())['records'];pack=base/'piper-balanced-training-features-v1';old=json.loads((pack/'report.json').read_text());assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
prior=base/'v46-balanced-training-support-v1';pr=json.loads((prior/'report.json').read_text());assert sha(prior/'scores.npz')==pr['scores_sha256']
model=base/'model-gpu-v46-ami-extra/hey_nyx.tflite';assert sha(model)=='5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417';detector=Detector(model);out.mkdir(exist_ok=False);details=[];arrays={};groups={}
with np.load(prior/'scores.npz') as z:
 for i in range(8064):
  for arm in ['quiet','noise15','rir','combined']:
   if arm in ['quiet','combined']:
    k=3*i+(0 if arm=='quiet' else 2);d=old['records'][k];q=z[f'int8_{k}'];assert hashlib.sha256(q.tobytes()).hexdigest()==pr['details'][k]['modes']['int8']['score_sha256'];rawshape=d['raw_shape'];voice=d['voice'];target=d['target'];speed=d['length_multiplier'];signal=d['signal_samples'];rawsha=d['raw_file_sha256']
   else:
    d=controls[2*i+(0 if arm=='noise15' else 1)];assert d['source_index']==i and d['arm']==arm;p=root/d['files']['raw']['path'];assert sha(p)==d['files']['raw']['sha256'];raw=np.load(p,allow_pickle=False);detector.reset();q=detector.predict(raw);rawshape=list(raw.shape);voice=d['voice'];target=d['target'];speed=d['speed'];signal=d['signal_samples'];rawsha=sha(p)
   assert np.isfinite(q).all();key=f'{i}_{arm}';arrays[key]=q;five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');first=150+3*np.arange(len(five));last=first+12;valid=((first-150)*160<=32000)&((last-150)*160<=32000)&((first+1)*160>=32000+signal);label=int(target=='hey_nyx');counts={}
   for t in [.5,.992,.99609375]:
    h=dict(full_trace_events=event_indices(smooth(q,5),t),warmed_restarted_events=event_indices(five,t),warmed_max_five=float(five.max()),container_support_positions=int(np.count_nonzero(five[valid]>=t)));counts[str(t)]=h
    for group in [f'{arm}/label/{label}',f'{arm}/phrase/{target}',f'{arm}/speed/{speed}/label/{label}']:
     g=groups.setdefault(group+'/'+str(t),dict(variants=0,full_trace_event_clips=0,warmed_high_clips=0,warmed_restarted_event_clips=0,container_support_hit_clips=0));g['variants']+=1;g['full_trace_event_clips']+=bool(h['full_trace_events']);g['warmed_high_clips']+=bool(five.max()>=t);g['warmed_restarted_event_clips']+=bool(h['warmed_restarted_events']);g['container_support_hit_clips']+=h['container_support_positions']>0
   details.append(dict(source_index=i,arm=arm,voice=voice,target=target,speed=speed,label=label,raw_file_sha256=rawsha,raw_shape=rawshape,score_sha256=hashlib.sha256(q.tobytes()).hexdigest(),valid_support_positions=int(valid.sum()),thresholds=counts))
  if i%250==0:print(json.dumps(dict(completed_sources=i+1,total=8064)),flush=True)
assert len(details)==32256
np.savez_compressed(out/'scores.npz',**arrays)
r=dict(model_sha256=sha(model),audit_sha256=sha(auditpath),controls_report_sha256=sha(root/'controls.json'),prior_report_sha256=sha(prior/'report.json'),sources=8064,traces=32256,script_sha256=sha(Path(__file__)),scores_sha256=sha(out/'scores.npz'),groups=groups,details=details,final_data_used=False,limits='Training-only frozen-model rendering diagnostic. Quiet/combined original authenticated traces reused. Controls newly inferred. Fixed thresholds descriptive; no training, calibration or promotion. Complete-container support not phonetic truth; noise prefix and headroom/tail effects remain part of intervention.')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in groups.items() if k.endswith('/0.99609375') and '/label/' in k and '/speed/' not in k}),flush=True)
