"""Authenticate existing Nyx phase schedule and construct matched Willow schedule."""
import hashlib,json,subprocess,sys
from pathlib import Path
from common_carrier_training_batches_v1 import Batches
from phase_aligned_batches_v1 import phase_input,independent_check
from hey_batch_schedule_v1 import batch_digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-frozen-encoder-schedule-execution-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if sys.argv[1]=='lock':
  parent=R/'4090-phase-aligned-schedule-execution-v1.json';files=json.loads(parent.read_text())['files']
  for p in [parent,Path(__file__),Path(__file__).with_name('frozen_encoder_hypothesis_v1.md'),R/'4090-phase-aligned-schedule-result-v1.json']:
   files[str(p)]=sha(p)
  for p,h in files.items():assert sha(p)==h,p
  with LOCK.open('x') as f:json.dump({'files':files,'policy':'Reuse original authenticated full Nyx phase schedule; construct all10000 Willow command_context phase inputs, independently check all1280000 transformed rows; no model or optimizer.'},f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 old=json.loads((R/'4090-phase-aligned-schedule-result-v1.json').read_text());assert sha(old['schedule_path'])==old['schedule_sha256']
 out=Path('.local/wakeword/frozen-encoder-schedules-v1');out.mkdir(exist_ok=False)
 reader=Batches('hey_willow','command_context');dest=out/'hey_willow.jsonl'
 with dest.open('x') as f:
  for i in range(1,10001):
   (x,y,w),receipt=reader.next();z=phase_input(x);independent_check(x,z)
   row={'step':i,'phase_sha256':batch_digest(z,y,w),'original_receipt':receipt,'phase_input_sha256':hashlib.sha256(z.tobytes()).hexdigest()}
   f.write(json.dumps(row)+'\n')
   if i%1000==0:f.flush();print('Willow schedule',i,flush=True)
 reader.finish()
 result={'execution_sha256':sha(LOCK),'schedules':{'hey_nyx':{'path':old['schedule_path'],'sha256':old['schedule_sha256'],'reuse_original':True},'hey_willow':{'path':str(dest),'sha256':sha(dest),'rows_independently_checked':1280000}},'batches_per_target':10000,'all_passed':True}
 with (R/'4090-frozen-encoder-schedule-result-v1.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
