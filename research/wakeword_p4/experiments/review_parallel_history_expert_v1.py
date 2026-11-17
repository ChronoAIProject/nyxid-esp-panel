"""Independent raw-code decoder, step-bucket arbiter and complete container census."""
import hashlib,json,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 P=R/'4090-parallel-history-expert-preflight-plan-v1.json';Q=R/'4090-parallel-history-expert-preflight-original-v1.json';p=json.loads(P.read_text());q=json.loads(Q.read_text());old=json.loads((R/'4090-parallel-context-streams-original-v1.json').read_text());assert q['plan_sha256']==sha(P) and len(q['streams'])==3
 for path,h in p['files'].items():assert sha(path)==h,path
 rows=[];total_gain=0;total_loss=0;files=[P,Q]
 for row,prior in zip(q['streams'],old['streams'],strict=True):
  assert row['name']==prior['name'] and row['timeline']==prior['timeline'];assert sha(row['trace']['path'])==row['trace']['sha256'];z=np.load(row['trace']['path'],allow_pickle=False);clock=z['relative_block_return'];events={};buckets=defaultdict(list)
  with np.load(prior['trace']['path'],allow_pickle=False) as before:
   for key in ['c161_uint8','binary40_uint8','relative_block_return']:assert np.array_equal(z[key],before[key])
  for name in p['priority']:
   codes=z[name+'_uint8'];assert codes.dtype==np.uint8 and len(codes)==len(clock);events[name]=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)];assert events[name]==row['events'][name]
   for e in events[name]:buckets[e['step']].append(dict(e,model=name))
  emitted=[];suppressed=[];last=-1000
  for step in sorted(buckets):
   for e in sorted(buckets[step],key=lambda e:p['priority'].index(e['model'])):
    if step-last>=67:emitted.append(e);last=step
    else:suppressed.append(e)
  assert emitted==row['emitted'] and suppressed==row['suppressed']
  all_events={**events,'dual':prior['emitted'],'triple':emitted};owned={};outside={}
  for name,es in all_events.items():
   hit=set();extra=[]
   for e in es:
    owners=[t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end']]
    if owners and owners[0]['kind']=='clip':hit.add(owners[0]['text_index'])
    else:extra.append(e)
   owned[name]=hit;outside[name]=extra
  groups=defaultdict(Counter);details=[]
  for t in row['timeline']:
   if t['kind']!='clip':continue
   hits={name:t['text_index'] in ids for name,ids in owned.items()};g=groups[t['group']];g['clips']+=1
   for name,h in hits.items():g[name]+=h
   gain=hits['triple'] and not hits['dual'];loss=hits['dual'] and not hits['triple'];g['gained_vs_dual']+=gain;g['lost_vs_dual']+=loss
   if t['group']=='nyx_connected_command':total_gain+=gain;total_loss+=loss
   details.append(dict(text_index=t['text_index'],group=t['group'],text=t['text'],hits=hits))
  rows.append(dict(name=row['name'],groups=groups,details=details,outside_clip_events=outside));files.append(Path(row['trace']['path']))
 archive=R/'4090-parallel-history-expert-preflight-evidence-v1.tar.gz';members=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 with tarfile.open(archive,'x:gz') as tf:
  for path in files:tf.add(path,arcname=str(path),recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getmembers())==len(members)
  for member,spec in zip(tf.getmembers(),members,strict=True):
   b=tf.extractfile(member).read();assert member.name==spec['path'] and len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
 out=dict(schema='parallel-history-expert-preflight-review-v1',all_passed=True,source_rows=144,streams=rows,target_gains_vs_dual=total_gain,target_losses_vs_dual=total_loss,bounded_gate_passed=total_gain>0 and total_loss==0,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Only fixed third-parent score feasibility on consumed three context streams; not actual interleaved triple execution or independent performance, resource or P4 evidence. No fitting/sweep; all ordinary/name/outside events retained. Gate only authorizes further evidence gathering, no promotion.')
 with (R/'4090-parallel-history-expert-preflight-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(target_gains=total_gain,target_losses=total_loss,bounded_gate_passed=out['bounded_gate_passed'],streams=[dict(name=r['name'],groups=r['groups']) for r in rows]),indent=2))
if __name__=='__main__':main()
