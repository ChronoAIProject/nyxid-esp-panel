"""Authenticate all saved teacher targets and independent physical anchor algebra."""
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-external-representation-target-review-plan-v1.json';REPORT=R/'4090-external-representation-target-cache-original-v1.json';CACHEPLAN=R/'4090-external-representation-target-cache-plan-v1.json';META=R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json';REC=R/'4090-external-representation-teacher-recovered-v3.json';OLD=Path('.local/wakeword/external-representation-teacher-v2/vectors.npz');E=Path('.local/wakeword/external-representation-target-cache-execution-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 d=json.loads(REPORT.read_text());files=[Path(__file__),REPORT,CACHEPLAN,META,REC,OLD,Path(d['arrays']['path']),E/'terminal.json',E/'cache.log']
 with LOCK.open('x') as f:json.dump(dict(files={str(p):sha(p) for p in files},policy='All4320source/raw/mean/normalizedtarget hashes, all600reuse identities, independent physical anchor formula, actualexit0/log binding and archive member checks. No teacher reinference, gradient, threshold choice or stream/P4 claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 d=json.loads(REPORT.read_text());meta=json.loads(META.read_text())['inventory'];old=json.loads(REC.read_text())['records'];by={r['source_id']:r for r in old};terminal=json.loads((E/'terminal.json').read_text())
 assert terminal['actual_exit_code']==0 and terminal['log_sha256']==sha(E/'cache.log') and d['plan_sha256']==sha(CACHEPLAN) and d['metadata_sha256']==sha(META) and d['optimizer_updates']==0 and d['teacher_parameter_versions_unchanged'] and not any(d['strict_loading_info'].values())
 last=json.loads((E/'cache.log').read_text().splitlines()[-1]);assert last['completed'] and last['arrays']==d['arrays'] and last['reused']==600 and last['inferred']==3720
 assert sha(d['arrays']['path'])==d['arrays']['sha256'] and Path(d['arrays']['path']).stat().st_size==d['arrays']['bytes']
 with np.load(d['arrays']['path'],allow_pickle=False) as z:raw=z['raw_prefix'].copy();mean=z['training_mean'].copy();labels=z['labels'].copy();assert set(z.files)=={'raw_prefix','training_mean','labels'}
 with np.load(OLD,allow_pickle=False) as z:previous=z['prefix'].copy()
 assert raw.shape==(4320,768) and raw.dtype==np.float32 and mean.shape==(768,) and mean.dtype==np.float64 and labels.dtype==np.int8 and labels.shape==(4320,) and np.isfinite(raw).all()
 expected_mean=raw.astype(np.float64).sum(0)/4320;assert np.array_equal(mean,expected_mean)
 delta=raw.astype(np.float64)-expected_mean;denominator=np.sqrt(np.square(delta).sum(1));assert np.all(denominator>1e-12);targets=(delta/denominator[:,None]).astype(np.float32);reuse=0;inferred=0
 assert len(d['records'])==len(meta)==4320
 for i,(r,m) in enumerate(zip(d['records'],meta,strict=True)):
  s=m['source'];assert r['inventory_index']==i and r['source_id']==s['id'] and r['native_sha256']==s['native_sha256'] and s['role']=='training'
  for key in ['engine','speaker_id','text','label']:assert r[key]==s[key]
  assert labels[i]==s['label'] and r['frames']==m['frames'] and r['prefix_samples']==m['prefix_samples'] and r['end_sample']==m['reference_samples'] and r['start_sample']==max(0,m['reference_samples']-49440)
  absolute_end=m['prefix_samples']+m['reference_samples'];anchor=(absolute_end+159)//480;release=480*anchor+800
  assert r['student_anchor_zero_based_call']==anchor and r['student_anchor_release_sample']==release and 0<=anchor<m['frames']//3 and absolute_end+480<=release<=absolute_end+4320 and r['teacher_future_samples']==0
  assert hashlib.sha256(raw[i].tobytes()).hexdigest()==r['raw_vector_sha256'] and hashlib.sha256(targets[i].tobytes()).hexdigest()==r['normalized_target_sha256']
  if s['id'] in by:
   o=by[s['id']];assert r['provenance']=='reused-authenticated-prefix' and o['end_sample']==r['end_sample'] and o['start_sample']==r['start_sample'] and np.array_equal(raw[i],previous[o['index']]);reuse+=1
  else:assert r['provenance']=='new-prefix-inference';inferred+=1
 assert reuse==600 and inferred==3720
 normerror=float(np.abs(np.sqrt(np.square(targets.astype(np.float64)).sum(1))-1).max());assert normerror<1e-6
 files=[Path(d['arrays']['path']),REPORT,E/'terminal.json',E/'cache.log'];members={};archive=R/'4090-external-representation-target-evidence-v1.tar.gz'
 with tarfile.open(archive,'x:gz') as t:
  for f in files:
   assert f.is_file() and not f.is_symlink() and f.name not in members;members[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f));t.add(f,arcname=f.name)
 with tarfile.open(archive,'r:gz') as t:
  assert set(t.getnames())==set(members) and len(t.getnames())==len(members)
  for member in t:
   data=t.extractfile(member).read();assert len(data)==members[member.name]['bytes'] and hashlib.sha256(data).hexdigest()==members[member.name]['sha256']
 out=dict(schema='external-representation-target-independent-review-v1',all_passed=True,review_plan_sha256=sha(LOCK),cache_report_sha256=sha(REPORT),actual_gpu_exit_code=0,verified_sources=4320,verified_reused_vectors=reuse,verified_new_vectors=inferred,verified_normalized_target_hashes=4320,verified_physical_anchors=4320,max_unit_norm_error=normerror,teacher_reinference_calls=0,optimizer_updates=0,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),generation_code_commit=subprocess.check_output(['git','log','-1','--format=%H','--','research/wakeword_p4/experiments/cache_external_representation_targets_v1.py'],text=True).strip(),review_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),limits='Original frozen teacher computation authenticated through code/actualexit/log and saved bytes; no independent regeneration of3720teacher outputs or intermediate teacher states. Independent saved-target/source/clock/normalization and600reuse audit only. No student or stream/P4 qualification.')
 with (R/'4090-external-representation-target-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
