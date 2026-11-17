"""Independent integer/window arbitration and complete archive/member review."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/parallel-history-expert-runtime-v1');PRIORITY=['c161','binary40','history_expert']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def decode(codes,clock):
 sums=np.convolve(codes.astype(np.int64),np.ones(5,np.int64),mode='valid');armed=True;previous=-1000;result=[]
 for i,total in enumerate(sums,5):
  if i-previous>=67:
   if 2*total<1175:armed=True
   if armed and total>=1175:result.append(dict(step=i,relative_sample=int(clock[i-1])));previous=i;armed=False
 return result

def main():
 p=read(R/'4090-parallel-history-expert-runtime-plan-v1.json');summary=read(R/'4090-parallel-history-expert-runtime-original-v1.json');assert summary['all_passed'] and summary['plan_sha256']==sha(R/'4090-parallel-history-expert-runtime-plan-v1.json')
 for f,h in p['files'].items():assert sha(f)==h,f
 scope=read(R/'4090-parallel-history-expert-scope-original-v1.json');context=read(R/'4090-parallel-history-expert-preflight-original-v1.json');base=read(R/'4090-c161-binary40-parallel-events-original-v2.json');members={};count=0;calls=0;inference_seconds=0
 def add(path):
  path=Path(path);name=str(path);assert not path.is_absolute() and name not in members;members[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
 for c,spec in summary['cohorts'].items():
  assert sha(spec['path'])==spec['sha256'];report=read(spec['path']);assert report['all_passed'];refs=context['streams'] if c=='context' else scope['cohorts'][c]['records'];assert len(report['records'])==len(refs)==spec['rows'];add(spec['path']);add(Path(spec['path']).with_name('records.jsonl'))
  archive=None
  if c!='context':archive=tarfile.open(refs[0]['teacher_trace']['archive'],'r:gz')
  try:
   for index,(row,ref) in enumerate(zip(report['records'],refs,strict=True)):
    assert row['exact'];t=row['trace'];assert sha(t['path'])==t['sha256'] and Path(t['path']).stat().st_size==t['bytes'];add(t['path'])
    if c=='context':assert row['name']==ref['name'] and row['timeline']==ref['timeline']
    else:assert row['index']==ref['index'] and row['phase']==ref['phase'] and row['source']==ref['source']
    with np.load(t['path'],allow_pickle=False) as z:
     clock=z['relative_block_return'];assert clock.ndim==1 and np.all(np.diff(clock)>=0);own={}
     for n in PRIORITY:
      codes=z[n+'_uint8'];assert codes.dtype==np.uint8 and len(codes)==len(clock);own[n]=decode(codes,clock)
      if c=='context':
       with np.load(ref['trace']['path'],allow_pickle=False) as old:assert np.array_equal(codes,old[n+'_uint8']) and np.array_equal(clock,old['relative_block_return'])
      elif n=='history_expert':
       sp=ref['teacher_trace'];raw=archive.extractfile(sp['member']).read();assert hashlib.sha256(raw).hexdigest()==sp['sha256']
       with np.load(io.BytesIO(raw),allow_pickle=False) as old:
        ck='block_return_sample' if c=='melo' else 'real_block_return_sample' if c in ['primed','jenny','joe'] else 'relative_block_return';assert np.array_equal(codes,old['uint8']) and np.array_equal(clock,old[ck]-(81600 if c=='melo' else 0))
      else:
       sp=base['cohorts'][c]['records'][index]['paths'][n];assert sha(sp['path'])==sp['sha256']
       with np.load(sp['path'],allow_pickle=False) as old:assert np.array_equal(codes,old['uint8']) and np.array_equal(clock,old[sp['clock']]-sp['offset'])
     buckets={}
     for n in PRIORITY:
      for e in own[n]:buckets.setdefault(e['step'],[]).append(dict(e,model=n))
     emitted=[];suppressed=[];last=None
     for step in sorted(buckets):
      for e in sorted(buckets[step],key=lambda e:PRIORITY.index(e['model'])):
       if last is None or step>=last+67:emitted.append(e);last=step
       else:suppressed.append(e)
     assert own==row['events']==ref.get('saved',ref.get('events'));assert emitted==row['emitted']==ref['emitted'] and suppressed==row['suppressed']==ref['suppressed'];calls+=len(clock)
    inference_seconds+=row['timing']['interleaved_inference_and_decoder_seconds'];count+=1
  finally:
   if archive:archive.close()
 assert count==1763;add(R/'4090-parallel-history-expert-runtime-plan-v1.json');add(R/'4090-parallel-history-expert-runtime-original-v1.json');add(Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__));add(Path(__file__).with_name('run_parallel_history_expert_runtime_v1.py').relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__).with_name('run_parallel_history_expert_runtime_v1.py'))
 archive=R/'4090-parallel-history-expert-runtime-evidence-v1.tar.gz'
 with tarfile.open(archive,'x:gz') as tf:
  for name in members:tf.add(name,arcname=name,recursive=False)
 with tarfile.open(archive,'r:gz') as tf:
  assert len(tf.getnames())==len(set(tf.getnames())) and set(tf.getnames())==set(members)
  for name,spec in members.items():
   b=tf.extractfile(name).read();assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
 result=dict(schema='parallel-history-expert-runtime-review-v1',all_passed=True,total_rows=count,total_native_calls=calls,inference_and_decoder_seconds=inference_seconds,models_invoked_per_call=3,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='All1763 full raw traces independently match authenticated references; integer window/step-bucket policy matches all own/emitted/suppressed events. Host execution only; Python timings are not P4 realtime/arena/stability or new generalization/background exposure.')
 with (R/'4090-parallel-history-expert-runtime-review-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='archive'}),flush=True);print(json.dumps(dict(archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members=len(members))),flush=True)
if __name__=='__main__':main()
