"""Independent source/resampling, raw-code event and arbiter checks for all96."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,math,tarfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');DATA=Path('.local/wakeword/parallel-melo-context-language-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=json.loads((R/'4090-parallel-melo-context-replay-plan-v1.json').read_text());report=json.loads((R/'4090-parallel-melo-context-replay-original-v1.json').read_text());source=json.loads((R/'4090-parallel-melo-context-language-plan-v1.json').read_text())
 assert report['completed_rows']==len(plan['rows'])==len(source['requests'])==96
 for path,h in plan['files'].items():assert sha(path)==h,path
 assert report['plan_sha256']==sha(R/'4090-parallel-melo-context-replay-plan-v1.json')
 summaries=defaultdict(Counter);changed=[];non_target=[];members=[];seconds=0
 for i,(gen,r) in enumerate(zip(plan['rows'],report['records'],strict=True)):
  assert gen['index']==r['index']==i and r['request']==gen['request']==source['requests'][i]
  npy,=[x['path'] for x in gen['files'] if x['path'].endswith('.npy')];wav,=[x['path'] for x in gen['files'] if x['path'].endswith('.wav')]
  native=np.load(npy,allow_pickle=False);assert native.dtype==np.float32 and list(native.shape)==gen['native_shape'] and np.isfinite(native).all()
  rate=gen['native_rate'];q=math.gcd(rate,16000);v=resample_poly(native[0],16000//q,rate//q);expected=np.rint(np.clip(v,-1,32767/32768)*32768).astype(np.int16);pcm,rate=sf.read(wav,dtype='int16')
  assert rate==16000 and np.array_equal(expected,pcm) and len(pcm)==r['pcm_samples'] and hashlib.sha256(pcm.tobytes()).hexdigest()==r['pcm_sha256'];seconds+=len(pcm)/16000
  assert sha(r['trace']['path'])==r['trace']['sha256'];z=np.load(r['trace']['path'],allow_pickle=False);clock=z['relative_block_return'];own={}
  for name in ['c161','binary40']:
   codes=z[name+'_uint8'];assert codes.dtype==np.uint8 and len(codes)==len(clock)
   own[name]=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)];assert own[name]==r['events'][name]
  events=sorted([dict(e,model=n) for n,es in own.items() for e in es],key=lambda e:(e['step'],e['model']!='c161'));emit=[];suppress=[]
  for e in events:
   if not emit or e['step']>=emit[-1]['step']+67:emit.append(e)
   else:suppress.append(e)
  assert emit==r['emitted'] and suppress==r['suppressed']
  hits={n:any(e['relative_sample']>=0 for e in es) for n,es in dict(**own,parallel=emit).items()};group=r['request']['group'];c=summaries[r['request']['speaker_condition']+'/'+group];c['clips']+=1
  for n,h in hits.items():c[n+'_hit_clips']+=h;c[n+'_prime_events']+=sum(e['relative_sample']<0 for e in dict(**own,parallel=emit)[n])
  if hits['parallel']!=hits['c161']:changed.append(dict(index=i,request=r['request'],hits=hits))
  if group!='nyx_connected_command' and hits['parallel']:non_target.append(dict(index=i,request=r['request'],hits=hits,events=emit))
  members.extend([Path(r['trace']['path']),DATA/f'{i:03d}.json'])
 members.extend(R/f for f in ['4090-parallel-melo-context-language-plan-v1.json','4090-parallel-melo-context-generation-lock-v1.json','4090-parallel-melo-context-replay-plan-v1.json','4090-parallel-melo-context-replay-original-v1.json'])
 archive=R/'4090-parallel-melo-context-trace-evidence-v1.tar.gz';ledger=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in members]
 with tarfile.open(archive,'x:gz') as tf:
  for p in members:tf.add(p,arcname=str(p),recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getmembers())==len(ledger)
  for x,m in zip(ledger,tf.getmembers(),strict=True):
   b=tf.extractfile(m).read();assert m.name==x['path'] and len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 out=dict(schema='parallel-melo-context-independent-review-v1',all_passed=True,rows=96,whole_audio_seconds=seconds,groups=summaries,changed_vs_c161=changed,non_target_triggers=non_target,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=ledger),limits='Whole synthesized waveforms mechanically validated and retained locally; archive contains score traces and metadata only, no PCM/native audio. One previously consumed builtin voice, targeted contexts, not independent acceptance/human speech/FAR. ASR separately reported without relabeling. No P4 resources or first-command proof.')
 with (R/'4090-parallel-melo-context-review-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k!='archive'},indent=2))
if __name__=='__main__':main()
