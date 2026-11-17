"""Independent archive/receipt and clip identity audit; no inference or tuning."""
import gzip,hashlib,json,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=read(R/'4090-external-representation-continuation-original-v1.json')
assert receipt['status']=='completed-await-independent-review'
assert len(receipt['stages'])==53
training=read(R/'4090-external-representation-training-review-v1.json');assert training['all_passed']
assert receipt['continuation_plan_sha256']==sha(R/'4090-external-representation-continuation-plan-v1.json')
assert all(s['actual_exit_code']==0 for s in receipt['stages'])
for stage in receipt['stages']:
 assert sha(stage['log'])==stage['log_sha256']
assert all(s['actual_exit_code']==0 for s in receipt['actual_training_subprocess_receipts'])
summary={}; members=0
for arm in ['control','feature']:
 summary[arm]={}
 export=read(R/f'4090-external-representation-{arm}-export-original-v1.json')
 assert sha(export['model']['path'])==export['model']['sha256']
 assert export['training_report_sha256']==training['arms'][arm]['training_report_sha256']
 assert export['checkpoint_sha256']==training['arms'][arm]['final_checkpoint_sha256']
 packed=R/f'4090-external-representation-{arm}-model-v1.tflite.gz'
 assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==export['model']['sha256']
 assert sha(R/f'4090-external-representation-{arm}-export-plan-v1.json')==export['plan_sha256']
 for cohort in ['builtin','daily','continuous','primed','jenny','joe','melo','mac']:
  p=R/f'4090-external-representation-{arm}-{cohort}-comparison-v1.json'; d=read(p)
  assert d['external_feature_arm']==arm and d['cohort']==cohort
  assert d['original_result_sha256']==d['archive']['members']['result.json']['sha256']
  replay=read(R/f'4090-external-representation-{arm}-{cohort}-replay-plan-v1.json')
  assert replay['external_feature_arm']==arm and replay['files'][export['model']['path']]==export['model']['sha256']
  a=d['archive']; ap=Path(a['path']); assert ap.stat().st_size==a['bytes'] and sha(ap)==a['sha256']
  with tarfile.open(ap,'r:gz') as t:
   assert len(t.getnames())==len(set(t.getnames())) and set(t.getnames())==set(a['members'])
   for m in t:
    b=t.extractfile(m).read(); spec=a['members'][m.name]
    assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256'];members+=1
  orig=read(R/f'4090-binary40-{cohort}-comparison-v1.json')
  entry={'comparison_sha256':sha(p),'groups':d.get('groups',{}),'reused_background':d.get('reused_background'),'original_binary40_changes':[]}
  assert len(d['records'])==len(orig['records'])
  for row,old in zip(d['records'],orig['records']):
   if cohort in ['primed','jenny','joe','melo']:
    assert row['index']==old['index'] and row['phase']==old['phase'] and row['source']==old['source']
    if row['hits'] is not None:
     assert row['hits']['c161']==old['hits']['c161']
     if row['hits']['binary40']!=old['hits']['binary40']:
      entry['original_binary40_changes'].append({'index':row['index'],'source':row['source'],'before':old['hits']['binary40'],'after':row['hits']['binary40']})
   elif cohort=='continuous':
    assert row['pcm_sha256']==old['pcm_sha256'] and row['gap_samples']==old['gap_samples']
   elif cohort in ['builtin','daily']:
    assert row['index']==old['index'] and row['request']==old['request']
    before=any(e['relative_sample']>0 for e in old['events']);after=any(e['relative_sample']>0 for e in row['events'])
    if before!=after:entry['original_binary40_changes'].append(dict(index=row['index'],request=row['request'],before=before,after=after))
   else:
    assert row['index']==old['index'] and row['source']==old['source']
    before=any(e['relative_sample']>0 for e in old['events']);after=any(e['relative_sample']>0 for e in row['events'])
    if before!=after:entry['original_binary40_changes'].append(dict(index=row['index'],source=row['source'],before=before,after=after))
  if cohort=='continuous':
   entry['containers']=d['records']
   for row,old in zip(d['records'],orig['records']):
    groups=set(row['containers'])|set(old['containers'])
    for group in groups:
     before=set(old['containers'].get(group,[]));after=set(row['containers'].get(group,[]))
     entry['original_binary40_changes'].append(dict(gap_samples=row['gap_samples'],group=group,gained=sorted(after-before),lost=sorted(before-after)))
  summary[arm][cohort]=entry
out={'schema':'external-representation-consumed-independent-review-v1','all_passed':True,'scope':'All 16 archive/member checks and original comparison clip identities; cohort scorers already authenticate raw event replay. No new inference. Not acceptance/P4 evidence.','verified_members':members,'execution_stages':len(receipt['stages']),'receipt_sha256':sha(R/'4090-external-representation-continuation-original-v1.json'),'arms':summary}
path=R/'4090-external-representation-consumed-review-v1.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'all_passed':True,'verified_members':members,'execution_stages':len(receipt['stages'])}))
for arm,cohorts in summary.items():
 print(arm)
 for c,e in cohorts.items():
  print(c,json.dumps(e['groups']),json.dumps(e.get('containers',[]))[:1500])
