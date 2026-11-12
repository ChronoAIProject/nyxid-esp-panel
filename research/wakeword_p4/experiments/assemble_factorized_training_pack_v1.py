"""Assemble original-byte 5-condition pack; no synthesis or wake-model inference."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');old=base/'piper-balanced-training-features-v1';controls=base/'balanced-rendering-factorial-v1';out=base/'factorized-training-features-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
r=json.loads((old/'report.json').read_text());c=json.loads((controls/'controls.json').read_text());auditpath=Path('research/wakeword_p4/results/4090-balanced-rendering-controls-audit-v1.json');audit=json.loads(auditpath.read_text());assert sha(controls/'controls.json')==audit['controls_report_sha256'] and audit['verified']==16128 and audit['all_arrays_pcm_frontend_exact']
old_report_hash=sha(old/'report.json')
out.mkdir(exist_ok=False);records=[];xs=[];ys=[]
for i in range(8064):
 for v in range(5):
  if v<3:
   d=r['records'][3*i+v];path=old/d['raw'];digest=d['raw_file_sha256'];end=d['feature_crop_end'];signal=d['signal_samples'];arm=['quiet','noise25','combined'][v];origin={'report_sha256':old_report_hash,'record_index':3*i+v};voice=d['voice'];target=d['target'];speed=d['length_multiplier'];src=d['source']
  else:
   d=c['records'][2*i+v-3];path=controls/d['files']['raw']['path'];digest=d['files']['raw']['sha256'];end=d['feature_crop_end'];signal=d['signal_samples'];arm=d['arm'];origin={'report_sha256':audit['controls_report_sha256'],'record_index':2*i+v-3};voice=d['voice'];target=d['target'];speed=d['speed'];src=d['source']
  assert sha(path)==digest;raw=np.load(path,allow_pickle=False);assert raw.dtype==np.uint16 and raw.shape[1]==40 and 150<=end<=min(350,len(raw));assert (min(350,len(raw))-150)*160<=32000 and (end+1)*160>=32000+signal
  xs.append(raw[end-150:end]);ys.append(int(target=='hey_nyx'));records.append(dict(pack_index=len(records),source_index=i,source=src,voice=voice,target=target,length_multiplier=speed,variant=v,arm=arm,raw='../'+str(path.relative_to(base)),raw_file_sha256=digest,raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),raw_shape=list(raw.shape),feature_crop_end=end,signal_samples=signal,split='training',entire_signal_fits=True,mechanical_pass=True,origin=origin))
 if i%500==0:print(json.dumps(dict(sources=i+1,total=8064)),flush=True)
assert len(records)==40320
np.savez_compressed(out/'training.npz',x=np.asarray(xs,np.uint16),y=np.asarray(ys,np.float32))
report=dict(role='training-only-factorized-rendering-pack',included=40320,attempted_variants=40320,records=records,pack_sha256=sha(out/'training.npz'),script_sha256=sha(Path(__file__)),controls_audit_sha256=sha(auditpath),wake_models_scored=True,prior_scoring_scope='Frozen-v46 training-only diagnostic; no unseen/generalization claim',final_data_used=False,limits='Original raw arrays referenced without rewriting; original and control source quality audits inherited. Uniform5conditions changes original augmentation mass, not additional batch slots. No new model inference/training or promotion.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
