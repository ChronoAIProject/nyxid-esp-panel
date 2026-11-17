"""Read-only preflight of locked pipeline and existing comparison inputs."""
import hashlib,json,shutil,time
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 locks={};checked={}
 for name in ['training','continuation','finalization']:
  p=R/f'4090-external-representation-{name}-plan-v1.json';plan=json.loads(p.read_text())
  for f,h in plan['files'].items():assert sha(f)==h,f;checked[f]=h
  locks[name]=sha(p)
 cohorts=['builtin','daily','continuous','primed','jenny','joe','melo','mac'];comparisons={}
 for prefix in ['4090-binary40','4090-frozen-history-teacher-teacher']:
  for c in cohorts:
   p=R/f'{prefix}-{c}-comparison-v1.json';d=json.loads(p.read_text());assert d['cohort']==c
   assert len(d['records'])==dict(builtin=36,daily=96,continuous=2,primed=372,jenny=408,joe=360,melo=474,mac=12)[c]
   comparisons[str(p)]=sha(p)
 baseline=json.loads((R/'4090-binary40-final-summary-v1.json').read_text())
 for f,h in baseline['source_files'].items():assert sha(f)==h,f
 c161=Path('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite')
 assert c161.stat().st_size==60952 and sha(c161)=='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
 processes={}
 for pid,identity in [(2599129,'12411306'),(2599132,'12411605'),(2601318,'12463346'),(2601847,'12471645')]:
  fields=Path(f'/proc/{pid}/stat').read_text().split();assert fields[21]==identity and fields[2]!='Z'
  processes[str(pid)]=dict(starttime=identity,state=fields[2])
 free=shutil.disk_usage('.').free;assert free>=100*1024**3
 out=dict(schema='external-representation-pipeline-input-preflight-v1',observed_unix=time.time(),all_passed=True,locks=locks,unique_bound_files=len(checked),comparison_inputs=comparisons,c161_sha256=sha(c161),live_processes=processes,free_bytes=free,limits='Read-only hash, saved comparison counts and current process identity audit. Not completed training, export, recognition, final archive review or P4 evidence. Original jobs left unchanged.')
 dest=R/'4090-external-representation-pipeline-input-preflight-v1.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
