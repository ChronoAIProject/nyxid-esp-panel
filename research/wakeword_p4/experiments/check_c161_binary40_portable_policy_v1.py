"""Build complete score-only fixture and verify portable C++ event policy."""
import gzip,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/c161-binary40-portable-policy-v1');LOCK=R/'4090-c161-binary40-portable-policy-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 ROOT.mkdir(exist_ok=False);source=json.loads((R/'4090-c161-binary40-parallel-events-original-v2.json').read_text());fixture=ROOT/'scores-and-events.bin.gz';rows=0;calls=0
 with gzip.open(fixture,'xb') as f:
  f.write(b'NYXPAR01'+struct.pack('<I',1760))
  for cohort in source['cohorts'].values():
   for row in cohort['records']:
    codes={}
    for name,spec in row['paths'].items():
     assert sha(spec['path'])==spec['sha256']
     with np.load(spec['path'],allow_pickle=False) as z:codes[name]=z['uint8'].copy();assert codes[name].dtype==np.uint8
    length=len(codes['c161']);assert len(codes['binary40'])==length
    records=np.zeros((length,5),np.uint8);records[:,0]=codes['c161'];records[:,1]=codes['binary40']
    for name,mask in [('c161',1),('binary40',2)]:
     for event in row['saved'][name]:records[event['step']-1,2]|=mask
    for event in row['emitted']:records[event['step']-1,3]|=1 if event['model']=='c161' else 2
    for event in row['suppressed']:records[event['step']-1,4]|=1 if event['model']=='c161' else 2
    f.write(struct.pack('<I',length));f.write(records.tobytes());rows+=1;calls+=length
 assert rows==1760
 binary=ROOT/'check-policy';compile_args=['g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','research/wakeword_p4/portable/check_c161_binary40_events_v1.cpp','-o',str(binary)]
 compile_run=subprocess.run(compile_args,capture_output=True,text=True);(ROOT/'compile.log').write_text(compile_run.stdout+compile_run.stderr);assert compile_run.returncode==0
 with gzip.open(fixture,'rb') as f:
  check=subprocess.run([str(binary)],input=f.read(),capture_output=True)
 (ROOT/'check.log').write_bytes(check.stdout+check.stderr)
 receipt=dict(schema='c161-binary40-portable-policy-check-v1',plan_sha256=sha(LOCK),compile_args=compile_args,compile_actual_exit_code=compile_run.returncode,check_actual_exit_code=check.returncode,stdout=check.stdout.decode(),stderr=check.stderr.decode(),streams=rows,calls=calls,fixture=dict(path=str(fixture),bytes=fixture.stat().st_size,sha256=sha(fixture)),files={str(p):sha(p) for p in ROOT.iterdir() if p.is_file()},limits='All1760consumed saved-score streams, no PCM or model inference. Portable policy matches independent detector/shared-cooldown event reference; not P4 model numerical/resource/realtime/first-command proof.')
 (R/'4090-c161-binary40-portable-policy-check-v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));assert check.returncode==0
if __name__=='__main__':main()
