"""Original wholewave frontend on independently admitted Piper Nick renders."""
import hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');ROOT=BASE/'piper-nick-command-rendered-v1';OUT=BASE/'piper-nick-command-features-v1';PLAN=R/'4090-piper-nick-command-feature-plan-v1.json';TRAINING=Path('research/wakeword_p4/training')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def versions():return {n:importlib.metadata.version(n) for n in ['numpy','scipy','soundfile','pymicro-features']}
def plan():
 import pymicro_features,micro_features_cpp
 report=R/'4090-piper-nick-command-render-original-v1.json';audit=R/'4090-piper-nick-command-render-review-v1.json'
 d=read(report);a=read(audit);assert a['original_report_sha256']==sha(report) and a['independently_qualified']==d['quality_pass']==d['cases']==18144 and not a['failures']
 ledger=ROOT/'records.jsonl';assert sha(ledger)==d['records_sha256']
 paths=[Path(__file__),report,audit,ledger,TRAINING/'features.py',Path(pymicro_features.__file__),Path(micro_features_cpp.__file__),R/'4090-piper-nick-command-render-execution-v1.json',R/'4090-piper-nick-command-render-review-execution-v1.json']
 p=dict(schema='piper-nick-command-feature-plan-v1',files={str(f):sha(f) for f in paths},versions=versions(),rows=18144,workers=1,policy='Original wholePCM frontend per row reset, all startup/tail frames, final160hop zero padding only. Separate littleendian uint16 40channel pack; no cropping/augmentation/model scoring or optimization. Preserve positive anchors/pairkeys and all input hashes. One worker CPUmath2, expected2GiB reserve100GiB. Full independent raw frontend reconstruction required before sampler admission; frames>846 block current sampler rather than silently truncate. C161 unchanged.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 assert versions()==p['versions'];subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','2'],check=True)
 sys.path.insert(0,str(TRAINING.resolve()));from features import features
 OUT.mkdir(exist_ok=False);rows=[];offset=0
 with (OUT/'features.bin').open('xb') as packed:
  for index,line in enumerate((ROOT/'records.jsonl').open()):
   r=json.loads(line);assert r['order']==index and r['quality_pass'];wav=Path(r['path']);assert sha(wav)==r['sha256']
   pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
   raw=features(pcm).astype('<u2');assert raw.shape==((len(pcm)+159)//160-2,40) and len(raw)>=225;raw.tofile(packed)
   source=r['source'];assert source['kind']=='negative' and source['engine']=='piper'
   rows.append(dict(order=index,source_id=source['id'],split='training',label=0,target_id='common_negative',engine='piper',conditioning_id=source['pair_key']['conditioning'],source_pair_key=source['pair_key'],positive_anchor=source['positive_anchor'],text=source['text'],variant=r['variant'],offset=offset,frames=len(raw),lead_frame=298,keyword_frame=None,generated_prefix_end_sample=None,generated_container_end_sample=r['generated_container_end_sample'],path=str(wav),sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=len(pcm),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
   offset+=len(raw)
   if len(rows)%1000==0:print(json.dumps(dict(rows=len(rows),total=18144)),flush=True)
 assert len(rows)==18144
 (OUT/'features.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
 result=dict(schema='piper-nick-command-feature-original-v1',plan_sha256=sha(PLAN),rows=len(rows),frames=offset,binary_bytes=(OUT/'features.bin').stat().st_size,binary_sha256=sha(OUT/'features.bin'),metadata_sha256=sha(OUT/'features.json'),max_frames=max(r['frames'] for r in rows),rows_over846=sum(r['frames']>846 for r in rows),training_enabled=False,limits=p['policy'])
 with (R/'4090-piper-nick-command-feature-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
