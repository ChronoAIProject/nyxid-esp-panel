"""Independent saved dual trace event/cooldown/source audit; no inference."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/c161-binary40-parallel-runtime-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def decode(codes,clock):
 totals=np.convolve(codes.astype(np.int64),np.ones(5,np.int64),'valid');last=-1000;armed=True;out=[]
 for step,total in enumerate(totals,5):
  if total<587.5 and step-last>=67:armed=True
  if armed and total>=1175 and step-last>=67:out.append(dict(step=step,relative_sample=int(clock[step-1])));last=step;armed=False
 return out
def main():
 execution=read(R/'4090-c161-binary40-parallel-runtime-execution-v1.json');assert execution['actual_exit_code']==0 and sha(execution['log'])==execution['log_sha256']
 result=read(R/'4090-c161-binary40-parallel-runtime-original-v1.json');assert result['all_passed'] and result['total_rows']==1760
 plan=R/'4090-c161-binary40-parallel-runtime-plan-v1.json';assert sha(plan)==result['plan_sha256']
 for path,h in read(plan)['files'].items():assert sha(path)==h,path
 reference=read(R/'4090-c161-binary40-parallel-events-original-v2.json');count=0;seen=set();timings={}
 for c,spec in result['cohorts'].items():
  assert sha(spec['path'])==spec['sha256'];d=read(spec['path']);assert d['all_passed'] and len(d['records'])==spec['rows'];calls=0;seconds=0.
  for row,old in zip(d['records'],reference['cohorts'][c]['records'],strict=True):
   assert row['index']==old['index'] and row['source']==old['source'] and row['phase']==old['phase']
   p=Path(row['trace']['path']);assert str(p) not in seen;seen.add(str(p));assert p.stat().st_size==row['trace']['bytes'] and sha(p)==row['trace']['sha256']
   with np.load(p,allow_pickle=False) as z:
    clock=z['relative_block_return'];events={}
    for n,s in old['paths'].items():
     codes=z[n+'_uint8'];assert codes.dtype==np.uint8 and len(codes)==len(clock);assert sha(s['path'])==s['sha256']
     with np.load(s['path'],allow_pickle=False) as saved:assert np.array_equal(codes,saved['uint8']) and np.array_equal(clock,saved[s['clock']]-s['offset'])
     events[n]=decode(codes,clock);assert events[n]==row['events'][n]==old['saved'][n]
   buckets={}
   for n in ['c161','binary40']:
    for e in events[n]:buckets.setdefault(e['step'],[]).append(dict(**e,model=n))
   emitted=[];suppressed=[];eligible=0
   for step,candidates in sorted(buckets.items()):
    if step>=eligible:emitted.append(candidates[0]);suppressed.extend(candidates[1:]);eligible=step+67
    else:suppressed.extend(candidates)
   assert emitted==row['emitted']==old['emitted'] and suppressed==row['suppressed']==old['suppressed']
   assert row['per_model_codes_exact'] and row['online_events_exact'];calls+=row['timing']['calls'];seconds+=row['timing']['interleaved_inference_and_decoder_seconds'];count+=1
  timings[c]=dict(rows=len(d['records']),calls=calls,host_inference_and_decoder_seconds=seconds,host_seconds_per_30ms_call=seconds/calls)
 assert count==len(seen)==1760
 files=sorted(p for p in B.rglob('*') if p.is_file());members={str(p.relative_to(B)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};archive=R/'4090-c161-binary40-parallel-runtime-evidence-v1.tar.gz'
 with tarfile.open(archive,'x:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(B)))
 with tarfile.open(archive,'r:gz') as t:
  names=t.getnames();assert len(names)==len(set(names)) and set(names)==set(members)
  for member in t:
   data=t.extractfile(member).read();spec=members[member.name];assert len(data)==spec['bytes'] and hashlib.sha256(data).hexdigest()==spec['sha256']
 out=dict(schema='c161-binary40-parallel-runtime-independent-review-v1',all_passed=True,unique_traces=count,execution_receipt_sha256=sha(R/'4090-c161-binary40-parallel-runtime-execution-v1.json'),result_sha256=sha(R/'4090-c161-binary40-parallel-runtime-original-v1.json'),timings=timings,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='All raw dual codes/clocks/source identities checked; independent convolution event sums and step-bucket shared cooldown reproduce actual online execution and saved simulation. Host Python/BUILTIN_REF timings not P4 feasibility or first-command evidence. Reused backgrounds nonadditive; no model promotion.')
 with (R/'4090-c161-binary40-parallel-runtime-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,unique_traces=count,verified_archive_members=len(members),timings=timings)))
if __name__=='__main__':main()
