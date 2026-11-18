"""Recompute every whole-PCM frontend row and validate complete packed offsets."""
import hashlib,json,os,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/piper-nick-command-features-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 receipt=R/'4090-piper-nick-command-feature-execution-v1.json';assert read(receipt)['actual_exit_code']==0
 planpath=R/'4090-piper-nick-command-feature-plan-v1.json';plan=read(planpath)
 for p,h in plan['files'].items():assert sha(p)==h,p
 reportpath=R/'4090-piper-nick-command-feature-original-v1.json';report=read(reportpath);assert report['plan_sha256']==sha(planpath)
 assert sha(B/'features.bin')==report['binary_sha256'] and sha(B/'features.json')==report['metadata_sha256']
 rows=read(B/'features.json');assert len(rows)==report['rows']==18144
 rendered=[json.loads(s) for s in Path('.local/wakeword/piper-nick-command-rendered-v1/records.jsonl').read_text().splitlines()]
 assert len(rendered)==len(rows)
 assert (B/'features.bin').stat().st_size==report['binary_bytes']==report['frames']*40*2
 raw=np.memmap(B/'features.bin',mode='r',dtype='<u2',shape=(report['frames'],40));offset=0;keys=set()
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 for i,(row,source) in enumerate(zip(rows,rendered,strict=True)):
  s=source['source'];assert row['order']==source['order']==i and row['offset']==offset
  assert row['source_id']==s['id'] and row['text']==s['text'] and row['positive_anchor']==s['positive_anchor'] and row['source_pair_key']==s['pair_key']
  assert row['variant']==source['variant'] and row['conditioning_id']==s['pair_key']['conditioning']
  assert row['split']=='training' and row['label']==0 and row['engine']=='piper' and row['target_id']=='common_negative'
  assert row['lead_frame']==298 and row['keyword_frame'] is None and row['generated_prefix_end_sample'] is None
  assert row['generated_container_end_sample']==source['generated_container_end_sample']
  p=Path(row['path']);assert str(p)==source['path'] and sha(p)==row['sha256']==source['sha256']
  pcm,rate=sf.read(p,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']==source['samples']
  assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']==source['pcm_sha256']
  recalculated=features(pcm).astype('<u2');assert recalculated.shape==(row['frames'],40)
  assert len(recalculated)==(len(pcm)+159)//160-2
  assert hashlib.sha256(recalculated.tobytes()).hexdigest()==row['raw_sha256']
  assert np.array_equal(raw[offset:offset+len(recalculated)],recalculated)
  key=(row['source_id'],row['variant']);assert key not in keys;keys.add(key);offset+=len(recalculated)
  if (i+1)%2000==0:print(json.dumps(dict(verified=i+1)),flush=True)
 assert offset==report['frames']
 assert max(x['frames'] for x in rows)==report['max_frames'] and sum(x['frames']>846 for x in rows)==report['rows_over846']
 out=dict(schema='piper-nick-command-feature-independent-review-v1',all_passed=True,rows_verified=18144,frames_verified=offset,sampler846_compatible=report['rows_over846']==0,max_frames=report['max_frames'],plan_sha256=sha(planpath),report_sha256=sha(reportpath),receipt_sha256=sha(receipt),reviewer_sha256=sha(__file__),training_enabled=False,limits='All original serialized PCM frontend outputs recomputed with same pinned frontend implementation, byteexact packed spans and metadata checked; not independent frontend implementation or P4 parity. No model scoring/training/heldout access.')
 with (R/'4090-piper-nick-command-feature-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
