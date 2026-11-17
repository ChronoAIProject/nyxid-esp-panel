"""Prelocked complete actual-runtime score fixture and portable policy check."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import gzip,hashlib,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=Path('research/wakeword_p4/portable');PLAN=R/'4090-parallel-history-portable-plan-v1.json';ROOT=Path('.local/wakeword/parallel-history-portable-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lock():
 review=R/'4090-parallel-history-expert-runtime-review-v1.json';assert read(review)['all_passed'];summary=R/'4090-parallel-history-expert-runtime-original-v1.json';files=[Path(__file__),review,summary]+[P/n for n in ['c161_binary40_events_v1.h','c161_binary40_history_events_v1.h','check_c161_binary40_history_events_v1.cpp']]
 for spec in read(summary)['cohorts'].values():
  files.append(Path(spec['path']))
  for row in read(spec['path'])['records']:files.append(Path(row['trace']['path']))
 out=dict(schema='parallel-history-portable-plan-v1',files={str(f):sha(f) for f in files},rows=1763,policy='All authenticated actual triple-runtime traces in original order. Score-only six-byte/call fixture: C161/B40/historyuint8 then own/emitted/suppressed masks(1/2/4). Fixed native235/256/5/67 independent detectors + shared67emitted cooldown, tie priorityC161/B40/history. No PCM, inference, training, threshold/parent/policy sweep, P4/resources or new exposure.')
 with PLAN.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 ROOT.mkdir(exist_ok=False);fixture=R/'4090-parallel-history-portable-scores-v1.bin.gz';rows=0;calls=0
 with fixture.open('xb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as f:
   f.write(b'NYXTRI01'+struct.pack('<I',1763))
   for spec in read(R/'4090-parallel-history-expert-runtime-original-v1.json')['cohorts'].values():
    for row in read(spec['path'])['records']:
     with np.load(row['trace']['path'],allow_pickle=False) as z:
      n=len(z['relative_block_return']);a=np.zeros((n,6),np.uint8)
      for col,name in enumerate(['c161','binary40','history_expert']):a[:,col]=z[name+'_uint8']
     mask=dict(c161=1,binary40=2,history_expert=4)
     for name,events in row['events'].items():
      for e in events:a[e['step']-1,3]|=mask[name]
     for col,key in [(4,'emitted'),(5,'suppressed')]:
      for e in row[key]:a[e['step']-1,col]|=mask[e['model']]
     f.write(struct.pack('<I',n));f.write(a.tobytes());calls+=n;rows+=1
 assert rows==1763 and calls==3961012
 binary=ROOT/'check-policy';cmd=['g++','-std=c++11','-O2','-Wall','-Wextra','-Werror',str(P/'check_c161_binary40_history_events_v1.cpp'),'-o',str(binary)];build=subprocess.run(cmd,capture_output=True);(ROOT/'compile.log').write_bytes(build.stdout+build.stderr);assert build.returncode==0
 with gzip.open(fixture,'rb') as f:check=subprocess.run([str(binary)],input=f.read(),capture_output=True)
 (ROOT/'check.log').write_bytes(check.stdout+check.stderr)
 out=dict(schema='parallel-history-portable-check-v1',plan_sha256=sha(PLAN),all_passed=check.returncode==0,compile_actual_exit_code=build.returncode,check_actual_exit_code=check.returncode,compile_args=cmd,stdout=check.stdout.decode(),stderr=check.stderr.decode(),streams=rows,calls=calls,fixture=dict(path=str(fixture),bytes=fixture.stat().st_size,sha256=sha(fixture)),files={str(f):sha(f) for f in ROOT.iterdir()},limits=p['policy'])
 with (R/'4090-parallel-history-portable-check-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True);assert check.returncode==0
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
