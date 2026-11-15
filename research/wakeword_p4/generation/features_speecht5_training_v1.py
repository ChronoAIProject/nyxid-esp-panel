"""Extract complete training-only PCM frontend features, never crop speech."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
SRC=BASE/'speecht5-training-rendered-v1';OUT=BASE/'speecht5-training-features-v1'
PLAN=R/'4090-speecht5-training-feature-plan-v1.json'
TRAIN=Path('research/wakeword_p4/training')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def versions():return {n:metadata.version(n) for n in ['numpy','scipy','soundfile','pymicro-features']}
def prepare():
 import pymicro_features,micro_features_cpp
 result=R/'4090-speecht5-training-render-result-v1.json';r=json.loads(result.read_text());assert r['whole_pack_qualified'] and r['requests']==r['mechanically_passed']==1512
 records=SRC/'records.jsonl';assert sha(records)==r['records_sha256']
 files=[Path(__file__),result,records,TRAIN/'features.py',Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)]
 p=dict(schema='speecht5-training-full-feature-plan-v1',files={str(f):sha(f) for f in files},versions=versions(),rows=1512,maximum_frames=846,
  policy='All1512 complete rendered streams, reset MicroFrontend per stream; all startup, source and RIR tail retained. Zero padding only last partial160sample hop; ceil(samples/160)-2 frames,40channels,uint16littleendian. Store all rows including untimed positive commands; eligibility for first paired training experiment is carrier==bare only. Full command rows preserved for future independently planned use. No dev/final data or detector invoked.',training_execution_authorized=False)
 with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(rows=1512,plan_sha256=sha(PLAN))),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 assert versions()==p['versions'];OUT.mkdir(exist_ok=False);sys.path.insert(0,str(TRAIN.resolve()));from features import features
 rows=[];offset=0
 try:
  with (OUT/'features.bin').open('xb') as output:
   for r in map(json.loads,(SRC/'records.jsonl').read_text().splitlines()):
    assert r['role']=='training' and r['passed'];wav=Path(r['path']);assert sha(wav)==r['sha256'];pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and len(pcm)==r['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
    raw=features(pcm).astype('<u2');assert raw.shape==((len(pcm)+159)//160-2,40) and len(raw)<=846
    raw.tofile(output);row=dict(r,offset=offset,frames=len(raw),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),eligible_first_experiment=r['carrier']=='bare');rows.append(row);offset+=len(raw)
    if len(rows)%168==0:print(json.dumps(dict(rows=len(rows),total=1512)),flush=True)
  assert len(rows)==1512 and len({r['id'] for r in rows})==1512
  (OUT/'features.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
  result=dict(plan_sha256=sha(PLAN),rows=len(rows),frames=offset,binary_sha256=sha(OUT/'features.bin'),metadata_sha256=sha(OUT/'features.json'),binary_bytes=(OUT/'features.bin').stat().st_size,all_training_only=True,bare_positive_rows=sum(r['carrier']=='bare' and r['label']==1 for r in rows),bare_negative_rows=sum(r['carrier']=='bare' and r['label']==0 for r in rows),command_positive_rows_preserved_excluded=sum(r['carrier']!='bare' and r['label']==1 for r in rows),training_executed=False,policy=p['policy'])
  with (R/'4090-speecht5-training-feature-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e),completed=len(rows)))+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
