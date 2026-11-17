"""Independent step-bucket and complete source/container summary audit."""
import hashlib,json,tarfile
from collections import Counter,defaultdict
from pathlib import Path
R=Path('research/wakeword_p4/results');P=R/'4090-parallel-history-expert-scope-plan-v1.json';Q=R/'4090-parallel-history-expert-scope-original-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def target(c,s):return s.get('group')=='nyx' if c=='builtin' else s.get('phrase_id')=='nyx' if c=='melo' else s.get('target',s.get('target_id'))=='hey_nyx'
def main():
 p=json.loads(P.read_text());q=json.loads(Q.read_text());assert q['plan_sha256']==sha(P) and q['total_rows']==1760
 summary={};added=[];lost=[];total=0
 for cohort,block in q['cohorts'].items():
  counts=defaultdict(Counter);continuity=[]
  for row in block['records']:
   total+=1;bucket=defaultdict(list)
   for n,es in row['saved'].items():
    for e in es:bucket[e['step']].append(dict(e,model=n))
   emit=[];suppress=[];last=-1000
   for step in sorted(bucket):
    for e in sorted(bucket[step],key=lambda e:p['priority'].index(e['model'])):
     if step-last>=67:emit.append(e);last=step
     else:suppress.append(e)
   assert emit==row['emitted'] and suppress==row['suppressed']
   events={**row['saved'],'dual':row['dual_emitted'],'triple':emit}
   if cohort=='continuous':
    owners={}
    for n,es in events.items():
     ids=defaultdict(set)
     for e in es:
      window=next(t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end'])
      if window['kind']=='clip':ids[window['group']].add(window['index'])
     owners[n]={k:sorted(v) for k,v in ids.items()};assert owners[n]==row['containers'][n]
    a=set(owners['dual'].get('nyx',[]));b=set(owners['triple'].get('nyx',[]));continuity.append(dict(gap_samples=row['index'],dual=len(a),triple=len(b),gained=sorted(b-a),lost=sorted(a-b)));continue
   s=row['source'];onset=s.get('lead_samples',0) if cohort in ['primed','jenny','joe'] and s.get('target_id')=='hey_nyx' else 0;hits={n:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in es) for n,es in events.items()};assert hits==row['hits']
   role='background' if row['phase']=='background' else 'target' if target(cohort,s) else 'non_target';voice=s['voice_id'].split(':')[-1] if cohort=='melo' and role!='background' else 'all';c=counts[voice+'/'+role];c['clips']+=1
   for n,h in hits.items():c[n+'_hit_clips']+=h;c[n+'_events']+=sum(e['relative_sample']>0 for e in events[n])
   if role=='background':c['reused_seconds']+=s['samples']/16000
   if hits['triple']!=hits['dual']:
    detail=dict(cohort=cohort,index=row['index'],phase=row['phase'],role=role,source=s,hits=hits,events=events)
    (added if hits['triple'] else lost).append(detail)
  summary[cohort]=dict(groups=counts,continuous=continuity)
 assert total==1760
 out=dict(schema='parallel-history-expert-scope-review-v1',all_passed=True,total_rows=total,original_sha256=sha(Q),summary=summary,added_hit_clips=added,lost_hit_clips=lost,review_scope='Independent step-bucket arbitration and every recorded source/container hit/summary identity. Original runner separately authenticates all3raw uint8 clocks/own event decodes and original teacher archive/source bindings. This review does not repeat raw model inference or raw-code decoding.',limits=p['policy'])
 dest=R/'4090-parallel-history-expert-scope-review-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 files=[P,Q,dest,Path(__file__),Path(__file__).with_name('replay_parallel_history_expert_scope_v1.py')];archive=R/'4090-parallel-history-expert-scope-evidence-v1.tar.gz';members=[dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)) for f in files]
 with tarfile.open(archive,'x:gz') as tf:
  for f in files:tf.add(f,arcname=str(f),recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getmembers())==len(members)
  for m,s in zip(tf.getmembers(),members,strict=True):
   b=tf.extractfile(m).read();assert m.name==s['path'] and len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
 receipt=dict(schema='parallel-history-expert-scope-archive-v1',archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),all_member_hashes_verified=True,raw_trace_policy='Existing archived teacher traces and dual runtime evidence retained by reference; no PCM or redundant raw-trace copies here.')
 (R/'4090-parallel-history-expert-scope-archive-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(total_rows=total,added_target=sum(x['role']=='target' for x in added),added_non_target=sum(x['role']=='non_target' for x in added),lost=len(lost),continuous=summary['continuous']['continuous']),indent=2))
 for x in added:
  if x['role']=='non_target':print(json.dumps(dict(cohort=x['cohort'],index=x['index'],text=x['source'].get('text'),source=x['source'])))
if __name__=='__main__':main()
