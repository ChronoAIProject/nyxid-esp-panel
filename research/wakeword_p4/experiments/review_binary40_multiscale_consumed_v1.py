"""Independent archive/receipt and clip identity audit; no inference or tuning."""
import hashlib,json,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=read(R/'4090-binary40-multiscale-continuation-original-v1.json')
assert receipt['status']=='completed-await-independent-review'
assert all(s['actual_exit_code']==0 for s in receipt['stages'])
assert all(s['exit_code']==0 for s in receipt['actual_training_subprocess_receipts'])
summary={}; members=0
for arm in ['control','multiscale']:
 summary[arm]={}
 for cohort in ['builtin','daily','continuous','primed','jenny','joe','melo','mac']:
  p=R/f'4090-binary40-multiscale-{arm}-{cohort}-comparison-v1.json'; d=read(p)
  assert d['span_arm']==arm and d['cohort']==cohort
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
   else: assert row['index']==old['index'] and row['source']==old['source']
  if cohort=='continuous':entry['containers']=d['records']
  summary[arm][cohort]=entry
# Independently decode each final context raw trace with cumulative integer sums.
import numpy as np
context_checks={}
for arm in ['control','multiscale']:
 d=read(R/f'4090-binary40-multiscale-{arm}-context-original-v1.json');assert d['all_passed'] and d['arm']==arm
 own=[]
 for row in d['streams']:
  assert sha(row['trace']['path'])==row['trace']['sha256']
  with np.load(row['trace']['path'],allow_pickle=False) as z:
   codes=z['uint8'];clock=z['relative_block_return'];prefix=np.r_[0,np.cumsum(codes,dtype=np.int64)];sums=prefix[5:]-prefix[:-5];events=[];armed=True;last=-1000
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
out={'schema':'binary40-multiscale-consumed-independent-review-v1','all_passed':True,'scope':'All16archive/member checks and original clip identities plus6physical context rawtraces independent integer decoding. No new inference/acceptance/P4 evidence.','verified_members':members,'execution_stages_before_review':len(receipt['stages']),'receipt_before_review_sha256':sha(R/'4090-binary40-multiscale-continuation-original-v1.json'),'arms':summary,'context_checks':context_checks}
path=R/'4090-binary40-multiscale-consumed-review-v1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'all_passed':True,'verified_members':members,'context':context_checks}))
