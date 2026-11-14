"""Authenticate the complete10000batch balanced auxiliary stream, no model calls."""
import gzip,hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-adversarial-schedule-execution-v1.json';RESULT=R/'4090-engine-adversarial-schedule-result-v1.json';OUT=Path('.local/wakeword/engine-adversarial-schedule-v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if sys.argv[1]=='lock':
  parent=R/'4090-engine-adversarial-input-execution-v1.json';d=json.loads(parent.read_text());files=dict(d['files']);report=R/'4090-engine-adversarial-input-result-v1.json';audit=R/'4090-engine-adversarial-input-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_passed'] and a['original_result_sha256']==sha(report)
  for p in [parent,report,audit,Path(__file__),Path(__file__).with_name('engine_adversarial_batches_v1.py'),Path(__file__).with_name('engine_adversarial_full_schedule_v1.md'),Path(__file__).with_name('audit_engine_adversarial_schedule_v1.py')]:files[str(p)]=sha(p)
  for b in json.loads(report.read_text())['batches']:files[b['arrays_path']]=b['arrays_sha256']
  for p,h in files.items():assert sha(p)==h,p
  with LOCK.open('x') as f:json.dump(dict(files=files,batches=10000,original_result=str(report),original_input_lock=str(parent),numpy=np.__version__,model_invocations=0),f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 from engine_adversarial_batches_v1 import AuxiliaryBatches
 OUT.mkdir(exist_ok=False);original=json.loads(Path(d['original_result']).read_text());reader=AuxiliaryBatches();path=OUT/'schedule.jsonl.gz'
 try:
  with gzip.open(path,'xt') as f:
   for i in range(10000):
    (x,ey,ty),receipt=reader.next()
    if i<50:
     old=original['batches'][i];assert receipt['step']==old['step'] and receipt['variant']==old['variant'] and receipt['entries']==old['entries'] and receipt['input_sha256']==old['input_sha256']
     with np.load(old['arrays_path'],allow_pickle=False) as a:assert np.array_equal(x,a['x']) and np.array_equal(ey,a['engine']) and np.array_equal(ty,a['lexical'])
    f.write(json.dumps(receipt)+'\n')
    if (i+1)%1000==0:print(json.dumps(dict(batches=i+1,windows=(i+1)*33)),flush=True)
  assert reader.count==10000
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),path=str(path),bytes=path.stat().st_size,sha256=sha(path),batches=10000,actual_windows=330000,first50_original_actual_tensors_exact=True,engines=reader.engines,texts=reader.texts,all_passed=True,model_invocations=0,limits='Full planned auxiliary source/endpoint/input hashes only. No optimizer/recognition/newdata or implicit formal launch.'),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
