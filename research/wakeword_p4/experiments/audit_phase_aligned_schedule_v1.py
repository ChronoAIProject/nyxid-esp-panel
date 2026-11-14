"""Full actual nominal/phase input schedules; no model imports or updates."""
import hashlib,json,subprocess,sys
from pathlib import Path
from common_carrier_training_batches_v1 import Batches
from phase_aligned_batches_v1 import phase_input,independent_check
from hey_batch_schedule_v1 import batch_digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-phase-aligned-schedule-execution-v1.json';OUT=Path('.local/wakeword/phase-aligned-schedule-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(action):
 if action=='lock':
  parent=R/'4090-hey-common-carrier-raw-formal-execution-v1.json';files=json.loads(parent.read_text())['files']
  paths=[parent,Path(__file__),Path(__file__).with_name('phase_aligned_batches_v1.py'),Path(__file__).with_name('common_carrier_training_batches_v1.py'),Path(__file__).with_name('phase_aligned_raw_training_v1.md'),R/'4090-command-crop-phase-backend-comparison-v1.json']
  for p in paths:files[str(p)]=sha(p)
  for f,h in files.items():assert sha(f)==h
  with LOCK.open('x') as f:json.dump(dict(files=files,policy='All10000 Nyx command_context actual batches, nominal and all-slot224past+zero phase. Independent per-row assignment check. Original full128 labels/weights exact, retain both tensorSHA and complete reader receipt. No model calls, no freshdata.'),f,indent=2)
 else:
  assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
  for f,h in p['files'].items():assert sha(f)==h,f
  OUT.mkdir(exist_ok=False);reader=Batches('hey_nyx','command_context');path=OUT/'schedule.jsonl';changed=0
  with path.open('x') as log:
   for i in range(1,10001):
    (x,y,w),receipt=reader.next();transformed=phase_input(x);independent_check(x,transformed)
    nominal=batch_digest(x,y,w);assert nominal==receipt['tensor_sha256'];phase=batch_digest(transformed,y,w);changed+=phase!=nominal
    row=dict(step=i,nominal_sha256=nominal,phase_sha256=phase,labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),original_receipt=receipt)
    log.write(json.dumps(row)+'\n')
    if i%1000==0:log.flush();print(json.dumps(dict(batches=i,changed=changed)),flush=True)
  reader.finish()
  report=dict(execution_sha256=sha(LOCK),schedule_path=str(path),schedule_sha256=sha(path),batches=10000,changed_batches=changed,all1280000_rows_independently_checked=True,labels_weights_exact=True,model_invocations=0,threshold_selected=False)
  with (R/'4090-phase-aligned-schedule-result-v1.json').open('x') as f:json.dump(report,f,indent=2)
if __name__=='__main__':main(sys.argv[1])
