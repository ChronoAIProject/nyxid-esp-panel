"""Independent all-source PCM ledger, native-clock and event/container audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,soundfile as sf
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');P=R/'4090-parallel-context-streams-plan-v1.json';Q=R/'4090-parallel-context-streams-original-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=json.loads(P.read_text());q=json.loads(Q.read_text());assert q['plan_sha256']==sha(P) and len(q['streams'])==len(p['streams'])==3
 for path,h in p['files'].items():assert sha(path)==h,path
 rows=[];files=[P,Q]
 for source,row in zip(p['streams'],q['streams'],strict=True):
  assert source['name']==row['name'];position=0;ledger=[];digest=hashlib.sha256()
  for i,item in enumerate(source['items']):
   pcm,rate=sf.read(item['wav'],dtype='int16');assert rate==16000 and pcm.ndim==1 and hashlib.sha256(pcm.tobytes()).hexdigest()==item['pcm_sha256']
   ledger.append(dict(kind='clip',start=position,end=position+len(pcm),text_index=item['text_index'],group=item['request']['group'],text=item['request']['text']));position+=len(pcm);digest.update(pcm.tobytes())
   if i<47:
    ledger.append(dict(kind='gap',start=position,end=position+8000,after_text_index=item['text_index']));position+=8000;digest.update(bytes(16000))
  assert ledger==row['timeline'] and position==row['pcm_samples'] and digest.hexdigest()==row['pcm_sha256']
  path=row['trace']['path'];assert sha(path)==row['trace']['sha256'];z=np.load(path,allow_pickle=False);clock=z['relative_block_return'];n=len(clock)
  frames=(position+81600+159)//160-2;assert n==(frames+2)//3
  supports=(3*np.arange(1,n+1)+2)*160;expected=np.minimum(((supports+479)//480)*480,position+81600)-81600;assert np.array_equal(clock,expected)
  own={}
  for name in ['c161','binary40']:
   codes=z[name+'_uint8'];assert codes.dtype==np.uint8 and len(codes)==n
   own[name]=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)];assert own[name]==row['events'][name]
  all_events=sorted([dict(e,model=name) for name,es in own.items() for e in es],key=lambda e:(e['step'],e['model']!='c161'));emit=[];suppress=[]
  for e in all_events:
   if not emit or e['step']-emit[-1]['step']>=67:emit.append(e)
   else:suppress.append(e)
  assert emit==row['emitted'] and suppress==row['suppressed']
  counts=defaultdict(Counter);details=[];outside={};hits={}
  for name,events in dict(**own,parallel=emit).items():
   found=set();extra=[]
   for e in events:
    if e['relative_sample']<=0:extra.append(dict(event=e,owner='prime'));continue
    owners=[x for x in ledger if x['start']<e['relative_sample']<=x['end']];assert len(owners)==1;owner=owners[0]
    if owner['kind']=='clip':found.add(owner['text_index'])
    else:extra.append(dict(event=e,owner=owner))
   outside[name]=extra;hits[name]=found
  for item in source['items']:
   index=item['text_index'];group=item['request']['group'];c=counts[group];c['clips']+=1;one={};many={}
   for name in ['c161','binary40','parallel']:
    isolated=item['isolated_emitted'] if name=='parallel' else item['isolated_events'][name];one[name]=any(e['relative_sample']>=0 for e in isolated);many[name]=index in hits[name]
    c['isolated_'+name]+=one[name];c['continuous_'+name]+=many[name]
   details.append(dict(text_index=index,request=item['request'],isolated_hit=one,continuous_container_hit=many))
  rows.append(dict(name=row['name'],groups=counts,details=details,outside_clip_events=outside,all_raw_events_exact=True,native_calls=n))
  files.extend([Path(path),Path(path).with_suffix('.json')])
 archive=R/'4090-parallel-context-streams-evidence-v1.tar.gz';members=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 with tarfile.open(archive,'x:gz') as tf:
  for path in files:tf.add(path,arcname=str(path),recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getmembers())==len(members)
  for m,s in zip(tf.getmembers(),members,strict=True):
   b=tf.extractfile(m).read();assert m.name==s['path'] and len(b)==s['bytes'] and hashlib.sha256(b).hexdigest()==s['sha256']
 out=dict(schema='parallel-context-streams-review-v1',all_passed=True,source_clips=144,streams=rows,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Consumed identical waveforms. Physical history/no-boundary-reset diagnostic, not independent exposure, new speakers or phonetic/first-command/P4 evidence. (start,end] block-return owner; gap events separate. No parameter/order/gap sweep or rescue.')
 with (R/'4090-parallel-context-streams-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps([dict(name=x['name'],groups=x['groups'],outside_events={k:len(v) for k,v in x['outside_clip_events'].items()}) for x in rows],indent=2))
if __name__=='__main__':main()
