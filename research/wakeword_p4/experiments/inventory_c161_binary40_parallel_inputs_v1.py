"""Metadata-only inventory of fixed parallel-parent comparison support."""
import hashlib,json,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-c161-binary40-parallel-input-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 counts={};baseline=json.loads((R/'4090-binary40-final-summary-v1.json').read_text())
 for p,h in baseline['source_files'].items():assert sha(p)==h,p
 for c,n in dict(builtin=36,daily=96,continuous=2,primed=372,jenny=408,joe=360,melo=474,mac=12).items():
  d=json.loads((R/f'4090-binary40-{c}-comparison-v1.json').read_text());assert len(d['records'])==n
  if c in ['primed','jenny','joe','melo']:
   assert all(set(row['events'])=={'c161','binary40'} for row in d['records'])
   assert all(set(row['hits'])=={'c161','binary40'} for row in d['records'] if row['hits'] is not None)
  counts[c]=dict(rows=n,comparison_sha256=sha(R/f'4090-binary40-{c}-comparison-v1.json'))
 parents={}
 for name,p,h,size in [('c161','.local/wakeword/causal-source-balanced-int8-v1/int8.tflite','c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81',60952),('binary40','.local/wakeword/binary40-trained-int8-v1/int8.tflite','454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c',64800)]:
  assert sha(p)==h and Path(p).stat().st_size==size;parents[name]=dict(path=p,sha256=h,bytes=size)
 out=dict(schema='c161-binary40-parallel-input-inventory-v1',all_passed=True,cohorts=counts,total_rows=sum(v['rows'] for v in counts.values()),parents=parents,total_model_file_bytes=sum(p['bytes'] for p in parents.values()),plan_sha256=sha(LOCK),optimizer_updates=0,model_inference_calls=0,limits='Metadata and model-byte identity inventory only. No raw paired trace-clock audit, fusion policy, ceiling scoring, ordinary retention guarantee, resource/P4 or recognition qualification.')
 with (R/'4090-c161-binary40-parallel-input-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
