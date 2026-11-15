"""Reuse all existing source-diagnostic negatives for an explicit training experiment."""
import argparse,gzip,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-existing-nearword-features-plan-v1.json';OUT=Path('.local/wakeword/causal-existing-nearword-features-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 source=R/'4090-causal-sentence-diagnostic-behavior-original-v1.json.gz';report=json.loads(gzip.decompress(source.read_bytes()));rows=[r for r in report['records'] if r['model']=='aligned-prefix' and r['label']==0]
 assert len(rows)==len({r['wave_path'] for r in rows})==288
 assert {r['prefix'] for r in rows}=={'Hey Muse','Hey Nick','Hey Next','Hello'}
 paths=[Path(__file__),source,R/'4090-causal-sentence-diagnostic-native-plan-v1.json',R/'4090-causal-sentence-diagnostic-native-original-v1.json',Path('research/wakeword_p4/training/features.py')]+[Path(r['wave_path']) for r in rows]
 p=dict(schema='causal-existing-nearword-features-plan-v1',files={str(x):sha(x) for x in paths},records=rows,expected_new_gib=.05,policy='Reuse all288existing complete negative diagnostic waveforms, not score-filtered; both known training-source voices/fournearword prefixes/fourcarriers/three speeds/three conditions. Explicitly transfer these training-related diagnostic examples to training for the next fixed experiment; never call them independent development or acceptance thereafter. Preserve original diagnostic evidence. No new TTS, PCM rendering/cropping, wake inference or protectedfinal access. Extract fullwave frontend UINT16 features once to a small memmap; keep source hashes/labels/provenance. Goal and candidate independent acceptance remain incomplete.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),waveforms=len(rows))))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
 OUT.mkdir(exist_ok=False);sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 rows=[];offset=0;binary=OUT/'features.bin'
 with binary.open('xb') as f:
  for r in p['records']:
   pcm,rate=sf.read(r['wave_path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples'];raw=features(pcm);assert raw.dtype==np.uint16 and raw.shape[1]==40 and len(raw)<=750
   raw.astype('<u2',copy=False).tofile(f);rows.append(dict(index=len(rows),voice=r['voice'],native_id=r['native_id'],prefix=r['prefix'],carrier=r['carrier'],text=r['text'],length_scale=r['length_scale'],variant=r['variant'],wave_path=r['wave_path'],wave_sha256=r['wave_sha256'],frames=len(raw),offset=offset,label=0,role='training'));offset+=len(raw)
 meta=OUT/'records.json';meta.write_text(json.dumps(rows,indent=2)+'\n')
 report=dict(schema='causal-existing-nearword-features-original-v1',plan_sha256=sha(LOCK),rows=len(rows),records=dict(path=str(meta),sha256=sha(meta)),features=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary),frames=offset),new_tts_requests=0,new_pcm_waveforms=0,detector_invocations=0,policy=p['policy'])
 with (R/'4090-causal-existing-nearword-features-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);x=a.parse_args();globals()[x.action]()
