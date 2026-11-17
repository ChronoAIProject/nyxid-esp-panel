"""Build all4320 paired physical frontend streams; no model or optimizer."""
import hashlib,json,sys,subprocess,shutil,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');META=R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json';PLAN=R/'4090-frozen-history-teacher-cache-plan-v1.json';OUT=Path('.local/wakeword/frozen-history-teacher-cache-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def plan():
 paths=[Path(__file__),META,Path('research/wakeword_p4/training/features.py')];PLAN.write_text(json.dumps(dict(files={str(p):sha(p) for p in paths},policy='All4320inventorysources, both silence/speech physical frontend streams, exact original metadata prefix/native/tail. No skipping/filtering/model/optimizer. Native hashes verified before feature generation. Rawuint16 binary, offsets in full ledger. First source per all30voices: both frontend full/chunk split at target parity, exact array bytes. Keep>=100GiBfree.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 assert shutil.disk_usage('.').free>100*1024**3+1024**3
 import soundfile as sf
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from features import features
 from pymicro_features import MicroFrontend
 d=json.loads(META.read_text());inventory=d['inventory'];byid={r['source']['id']:r['source'] for r in inventory};OUT.mkdir(exist_ok=False)
 streams={h:(OUT/f'{h}.bin').open('xb') for h in ['silence','speech']};offset=0;records=[];checked=set();start=time.monotonic()
 try:
  for i,row in enumerate(inventory):
   s=row['source'];negative=byid[row['context']]
   for src in [s,negative]:assert sha(src['native_path'])==src['native_sha256']
   target,rate=sf.read(s['native_path'],dtype='int16');assert rate==16000 and len(target)==s['native_samples']
   context,rate=sf.read(negative['native_path'],dtype='int16');assert rate==16000
   speech=np.r_[np.zeros(81600,np.int16),context,np.zeros((-len(context))%480+8160,np.int16)].astype(np.int16);assert len(speech)==row['prefix_samples']
   suffix=np.r_[target,np.zeros(16000,np.int16)];voice=(s['engine'],s['speaker_id']);do_check=voice not in checked
   hashes={}
   for h in ['silence','speech']:
    prefix=np.zeros_like(speech) if h=='silence' else speech
    raw=features(np.r_[prefix,suffix]);raw=raw[:len(raw)//3*3];assert raw.shape==(row['frames'],40) and raw.dtype==np.uint16
    if do_check:
     frontend=MicroFrontend();split=np.concatenate([features(prefix,frontend),features(suffix,frontend)]);split=split[:len(split)//3*3];assert np.array_equal(raw,split)
    b=raw.astype('<u2').tobytes();streams[h].write(b);hashes[h]=hashlib.sha256(b).hexdigest()
   if do_check:checked.add(voice)
   records.append(dict(inventory_index=i,source_id=s['id'],offset_frames=offset,frames=row['frames'],feature_sha256=hashes,split_frontend_parity_checked=do_check));offset+=row['frames']
   if (i+1)%500==0:print(json.dumps(dict(rows=i+1,elapsed_seconds=time.monotonic()-start)),flush=True)
 finally:
  for f in streams.values():f.close()
 assert len(checked)==30 and len(records)==4320
 files={h:dict(path=str(OUT/f'{h}.bin'),bytes=(OUT/f'{h}.bin').stat().st_size,sha256=sha(OUT/f'{h}.bin')) for h in streams};assert sum(f['bytes'] for f in files.values())==d['estimated_uint16_feature_bytes']
 result=dict(plan_sha256=sha(PLAN),rows=4320,files=files,records=records,frontend_split_checks=60,elapsed_seconds=time.monotonic()-start,optimizer_updates=0,policy=p['policy'])
 (R/'4090-frozen-history-teacher-cache-original-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=4320,frontend_split_checks=60,files=files)),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
