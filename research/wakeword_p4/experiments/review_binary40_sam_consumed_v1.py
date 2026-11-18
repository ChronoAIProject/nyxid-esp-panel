"""Independent archive/receipt and clip identity audit; no inference or tuning."""
import hashlib,json,tarfile,sys
import numpy as np
from pathlib import Path
R=Path('research/wakeword_p4/results')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_event_steps(codes, threshold):
 assert threshold==235/256 and codes.dtype==np.uint8
 sums=np.r_[0,np.cumsum(codes,dtype=np.int64)]
 armed=True;last=-1000;events=[]
 for step in range(5,len(codes)+1):
  total=int(sums[step]-sums[step-5])
  if step-last>=67:
   if 2*total<1175:armed=True
   if armed and total>=1175:events.append(step);last=step;armed=False
 return events
# Reuse source/clock authentication and group accounting, replace event decoder
# with the integer implementation above; no producer event_steps implementation.
sys.argv=[__file__,'review','sam']
import score_binary40_sam_v1 as scorer
import replay_binary40_sam_v1 as replay
scorer.event_steps=independent_event_steps
receipt=read(R/'4090-binary40-sam-evaluation-receipts-v1.json')
assert receipt['status']=='completed-await-independent-review'
expected_stages=[]
for arm in ['sam']:
 for cohort in ['builtin','daily','continuous','primed','jenny','joe','melo','mac']:
  expected_stages.extend([arm+'-'+cohort+'-replay',arm+'-'+cohort+'-score'])
 expected_stages.append(arm+'-context')
assert [s['stage'] for s in receipt['stages']]==expected_stages
assert all(s['actual_exit_code']==0 for s in receipt['stages'])
training=read(R/'4090-binary40-sam-training-receipt-v1.json')
assert training['exit_code']==0
training_review=read(R/'4090-binary40-sam-training-review-v1.json')
assert training_review['all_sampling_checks_passed'] and training_review['steps_verified']==20000
summary={}; members=0
for arm in ['sam']:
 summary[arm]={}
 scorer.ARM=arm;replay.ARM=arm
 for cohort in ['builtin','daily','continuous','primed','jenny','joe','melo','mac']:
  p=R/f'4090-binary40-sam-{arm}-{cohort}-comparison-v1.json'; d=read(p)
  assert d['span_arm']==arm and d['cohort']==cohort
  a=d['archive']; ap=Path(a['path']); assert ap.stat().st_size==a['bytes'] and sha(ap)==a['sha256']
  with tarfile.open(ap,'r:gz') as t:
   assert len(t.getnames())==len(set(t.getnames())) and set(t.getnames())==set(a['members'])
   for m in t:
    b=t.extractfile(m).read(); spec=a['members'][m.name]
    assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256'];members+=1
  rescored=scorer.coverage(cohort) if cohort in ['primed','jenny','joe','melo'] else scorer.mac() if cohort=='mac' else scorer.initial(cohort)
  assert all(d[key]==value for key,value in rescored.items()),(arm,cohort,'independent event rescore')
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
   else: assert row['index']==old['index'] and row['source']==old['source']
  if cohort=='continuous':entry['containers']=d['records']
  summary[arm][cohort]=entry
# Independently decode each final context raw trace with cumulative integer sums.
import numpy as np
context_checks={}
source_path=R/'4090-parallel-context-streams-plan-v1.json'
reference_path=R/'4090-parallel-context-streams-original-v1.json'
source=read(source_path);reference=read(reference_path)
expected_names=['builtin','melo-US','melo-BR']
assert [s['name'] for s in source['streams']]==expected_names
assert [s['name'] for s in reference['streams']]==expected_names
assert reference['plan_sha256']==sha(source_path)
for arm in ['sam']:
 plan_path=R/f'4090-binary40-sam-{arm}-context-plan-v1.json'
 export_path=R/f'4090-binary40-sam-{arm}-export-original-v1.json'
 plan=read(plan_path);export=read(export_path)
 assert plan['arm']==export['arm']==arm and plan['model']==export['model']
 for path in [source_path,reference_path,export_path,Path(export['model']['path'])]:
  assert plan['files'][str(path)]==sha(path)
 assert sha(export['model']['path'])==export['model']['sha256']
 d=read(R/f'4090-binary40-sam-{arm}-context-original-v1.json')
 assert d['all_passed'] and d['arm']==arm and d['plan_sha256']==sha(plan_path)
 assert [s['name'] for s in d['streams']]==expected_names
 own=[]
 for row,expected in zip(d['streams'],reference['streams'],strict=True):
  assert row['timeline']==expected['timeline']
  assert sum(t['kind']=='clip' for t in row['timeline'])==48
  assert Path(row['trace']['path']).stat().st_size==row['trace']['bytes']
  assert sha(row['trace']['path'])==row['trace']['sha256']
  with np.load(row['trace']['path'],allow_pickle=False) as z:
   codes=z['uint8'];clock=z['relative_block_return']
   assert codes.dtype==np.uint8 and codes.ndim==clock.ndim==1
   assert np.issubdtype(clock.dtype,np.integer)
   assert len(codes)==len(clock)==expected['timing']['calls']
   prefix=np.r_[0,np.cumsum(codes,dtype=np.int64)];sums=prefix[5:]-prefix[:-5];events=[];armed=True;last=-1000
   for i,total in enumerate(sums,5):
    if i-last>=67:
     if 2*total<1175:armed=True
     if armed and total>=1175:events.append(dict(step=i,relative_sample=int(clock[i-1])));last=i;armed=False
   assert events==row['events']
  groups={}
  for t in row['timeline']:
   if t['kind']=='clip':
    g=groups.setdefault(t['group'],dict(clips=0,hits=0));g['clips']+=1;g['hits']+=any(t['start']<e['relative_sample']<=t['end'] for e in events)
  assert all(g['clips']==row['groups'][key]['clips'] and g['hits']==row['groups'][key]['hits'] for key,g in groups.items());own.append(dict(name=row['name'],groups=groups))
 context_checks[arm]=own
out={'schema':'binary40-sam-consumed-independent-review-v1','all_passed':True,'scope':'All8complete cohorts independently event-rescored plus archive/member checks and original clip identities plus3physical context rawtraces independent integer decoding. No new inference/acceptance/P4 evidence.','verified_members':members,'execution_stages_before_review':len(receipt['stages']),'receipt_before_review_sha256':sha(R/'4090-binary40-sam-evaluation-receipts-v1.json'),'arms':summary,'context_checks':context_checks}
path=R/'4090-binary40-sam-consumed-review-v1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'all_passed':True,'verified_members':members,'context':context_checks}))
