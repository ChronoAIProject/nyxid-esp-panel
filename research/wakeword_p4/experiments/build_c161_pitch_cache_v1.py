"""Bounded one-pass, training-only pitch feature cache; no derived WAV storage."""
import hashlib,io,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-c161-pitch-cache-plan-v1.json';OUT=Path('.local/wakeword/c161-pitch-cache-v1');META=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json')
PHRASES=('Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def selected():return [r for r in json.loads(META.read_text()) if r['source_pair_key']['carrier']=='bare' and r['text'] in PHRASES]
def cents(row):
 key=json.dumps(dict(engine=row['engine'],pair=row['source_pair_key'],variant=row['variant']),sort_keys=True)
 return (-200,200)[int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2]
def plan():
 rows=selected();assert len(rows)==31644 and all(r['split']=='training' for r in rows)
 assert all(Path(r['path']).exists() for r in rows)
 sox=Path(shutil.which('sox')).resolve()
 files={str(p):sha(p) for p in [META,Path(__file__).resolve(),sox,Path('research/wakeword_p4/training/features.py')]}
 d=dict(schema='c161-pitch-cache-plan-v1',files=files,sox=str(sox),records=len(rows),frames=sum(r['frames'] for r in rows),expected_feature_bytes=sum(r['frames'] for r in rows)*80,max_new_gib=2,reserve_gib=100,policy='All31644 existing C161 broad training-only bare target/Nick/Next/Mix/Muse rows, all3engines/allvariants. One deterministic +/-200cent derivative per row, keyed engine/source_pair/variant independent of label/text; original source sampling unchanged. SoX-D singlethread/.8headroom, fullwave continuous frontend. At most1sample rounding correction at exact-zero tail, record delta; no internal crop/trim or boundary edits. Features uint16 appended once, no WAV retention. Mechanical cache only; no semantic/phonetic boundary proof, training or heldout access. Future training may choose original or derivative with fixed probability; not automatic admission.')
 with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 assert shutil.disk_usage(OUT.parent).free>(p['reserve_gib']+p['max_new_gib'])*1024**3
 import soundfile as sf
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 OUT.mkdir(exist_ok=False);start=time.monotonic();offset=0;rows=selected()
 with (OUT/'features.bin').open('xb') as binary,(OUT/'records.jsonl').open('x') as ledger:
  for i,row in enumerate(rows):
   assert sha(row['path'])==row['sha256']
   x,sr=sf.read(row['path'],dtype='int16');assert sr==16000 and x.ndim==1 and len(x)==row['samples']
   assert hashlib.sha256(x.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
   cmd=[p['sox'],'-D','--single-threaded','-v','0.8',row['path'],'-b','16','-t','wav','-','pitch',str(cents(row))]
   process=subprocess.run(cmd,check=True,capture_output=True);y,rate=sf.read(io.BytesIO(process.stdout),dtype='int16');delta=len(x)-len(y)
   assert rate==16000 and y.ndim==1 and abs(delta)<=1,(i,delta)
   if delta:
    assert not np.any(x[-160:]) and not np.any(y[-160:]),i
    y=np.pad(y,(0,1)) if delta==1 else y[:-1]
   assert len(y)==len(x) and np.max(np.abs(y.astype(np.int32)))<32767,i
   raw=features(y);assert raw.dtype==np.uint16 and raw.shape==(row['frames'],40)
   data=raw.astype('<u2').tobytes();binary.write(data)
   rec=dict(index=i,original_offset=row['offset'],offset=offset,frames=len(raw),cents=cents(row),tail_sample_delta=delta,source_sha256=row['sha256'],pcm_sha256=hashlib.sha256(y.astype('<i2').tobytes()).hexdigest(),features_sha256=hashlib.sha256(data).hexdigest(),stderr=process.stderr.decode(),label=row['label'],engine=row['engine'])
   ledger.write(json.dumps(rec)+'\n');offset+=len(raw)
   if (i+1)%100==0 or i==0:
    binary.flush();ledger.flush();progress=dict(completed=i+1,total=len(rows),frames=offset,elapsed_seconds=time.monotonic()-start)
    (OUT/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
   if (i+1)%1000==0:assert shutil.disk_usage(OUT).free>p['reserve_gib']*1024**3
 assert offset==p['frames'] and (OUT/'features.bin').stat().st_size==p['expected_feature_bytes']
 result=dict(schema='c161-pitch-cache-result-v1',plan_sha256=sha(PLAN),records=len(rows),frames=offset,elapsed_seconds=time.monotonic()-start,files={n:sha(OUT/n) for n in ['features.bin','records.jsonl']},training_started=False,scope='Training-only deterministic derived features, not new independent sources or qualification.')
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
