"""Sequential complete consumed cross-engine carrier evaluation, actual subprocess receipts."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments')
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def call(script,args):return ['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py',str(E/script),*args]
def main():
 action=sys.argv[1]
 if action=='plan':
  for c in COHORTS:subprocess.run(call('replay_binary40_cross_engine_carrier_v1.py',['plan',c,'joint']),check=True)
  subprocess.run(call('run_binary40_cross_engine_carrier_context_v1.py',['plan','joint']),check=True)
  return
 assert action=='run'
 destination=R/'4090-binary40-cross-engine-carrier-evaluation-receipts-v1.json'
 assert not destination.exists()
 logs=Path('.local/wakeword/binary40-cross-engine-carrier-evaluation-supervisor-v1');logs.mkdir(exist_ok=False)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
 stages=[]
 for c in COHORTS:
  stages.extend([('joint-'+c+'-replay',call('replay_binary40_cross_engine_carrier_v1.py',['run',c,'joint'])),('joint-'+c+'-score',call('score_binary40_cross_engine_carrier_v1.py',[c,'joint']))])
 stages.append(('joint-context',call('run_binary40_cross_engine_carrier_context_v1.py',['run','joint'])))
 receipt=dict(status='running',supervisor_pid=os.getpid(),stages=[])
 def save():
  temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(destination)
 save()
 for name,cmd in stages:
  start=time.time()
  with (logs/(name+'.log')).open('x') as f:
   proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);code=proc.wait()
  receipt['stages'].append(dict(stage=name,command=cmd,pid=proc.pid,actual_exit_code=code,start=start,end=time.time()))
  if code:receipt['status']='failed';save();raise SystemExit(code)
  save();print(json.dumps(receipt['stages'][-1]),flush=True)
 receipt['status']='completed-await-independent-review';save()
if __name__=='__main__':main()
