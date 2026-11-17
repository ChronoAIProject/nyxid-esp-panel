"""Complete the single predeclared derived-model evaluation, preserving real exits."""
import gzip,hashlib,json,os,subprocess,sys,time,traceback,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-midpoint-continuation-v1')
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
LOCK=R/'4090-binary40-midpoint-continuation-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 paths=[Path(__file__),Path(__file__).with_name('replay_binary40_midpoint_v1.py'),Path(__file__).with_name('score_binary40_midpoint_v1.py'),Path(__file__).with_name('binary40_stream_predict_v1.py'),R/'4090-binary40-midpoint-export-plan-v1.json',Path(__file__).with_name('binary40_midpoint_hypothesis_v1.md')]
 with LOCK.open('x') as f:json.dump(dict(schema='binary40-midpoint-continuation-plan-v1',files={str(p):sha(p) for p in paths},cohorts=COHORTS,policy='All8complete consumed cohorts, sequential bounded replay, one fixed derived model; real subprocess exits and original artifacts. No alpha/threshold/parent selection or automatic promotion.'),f,indent=2);f.write('\n')
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 for p,h in json.loads(LOCK.read_text())['files'].items():assert sha(p)==h,p

def run():
 verify()
 terminal=Path('.local/wakeword/binary40-midpoint-export-execution-v1/terminal.json');export_exit=json.loads(terminal.read_text());assert export_exit['actual_exit_code']==0 and sha(export_exit['log'])==export_exit['log_sha256']
 export=json.loads((R/'4090-binary40-midpoint-export-original-v1.json').read_text());assert export['all_neural_tensors_integer'] and export['fresh_reset_exact'] and export['optimizer_updates']==0
 assert sha(export['model']['path'])==export['model']['sha256']
 B.mkdir(exist_ok=False);state=dict(schema='binary40-midpoint-continuation-original-v1',pid=os.getpid(),status='running',export_actual_exit=export_exit,plan_sha256=sha(LOCK),stages=[])
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def publish(paths,message):
  subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True)
  rc=subprocess.run(['git','diff','--cached','--quiet']).returncode;assert rc in [0,1]
  if rc:subprocess.run(['git','commit','-m',message],check=True)
  subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 def stage(name,args):
  verify();start=time.monotonic();log=B/(name+'.log')
  env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
  with log.open('x') as f:
   child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env);state['active']=dict(name=name,child_pid=child.pid);save();rc=child.wait()
  row=dict(name=name,args=args,actual_exit_code=rc,elapsed_seconds=time.monotonic()-start,log=str(log),log_sha256=sha(log));state['stages'].append(row);state.pop('active',None);save();print(json.dumps(row),flush=True);assert rc==0,row
 py='/usr/bin/python3';base='research/wakeword_p4/experiments/'
 try:
  save()
  model=R/'4090-binary40-midpoint-model-v1.tflite.gz';model.write_bytes(gzip.compress(Path(export['model']['path']).read_bytes(),mtime=0))
  export_receipt=R/'4090-binary40-midpoint-export-execution-v1.json';export_receipt.write_text(terminal.read_text())
  export_log=R/'4090-binary40-midpoint-export-execution-v1.log.gz';export_log.write_bytes(gzip.compress(Path(export_exit['log']).read_bytes(),mtime=0))
  publish([model,export_receipt,export_log,R/'4090-binary40-midpoint-export-original-v1.json'],'research: preserve midpoint final integer export and actual exit')
  for c in COHORTS:stage('plan-'+c,[py,base+'replay_binary40_midpoint_v1.py','plan',c])
  publish([R/f'4090-binary40-midpoint-{c}-replay-plan-v1.json' for c in COHORTS],'research: prelock all eight complete midpoint consumed replay cohorts')
  for c in COHORTS:
   stage('replay-'+c,[py,base+'replay_binary40_midpoint_v1.py','run',c])
   stage('score-'+c,[py,base+'score_binary40_midpoint_v1.py',c])
   publish([R/f'4090-binary40-midpoint-{c}-comparison-v1.json',R/f'4090-binary40-midpoint-{c}-evidence-v1.tar.gz'],'research: preserve complete midpoint '+c+' evidence')
  state['status']='completed-await-independent-review'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(B/'failure.log').write_text(state['error'])
 finally:
  save();receipt=R/'4090-binary40-midpoint-continuation-original-v1.json';receipt.write_text(json.dumps(state,indent=2)+'\n');logs=R/'4090-binary40-midpoint-continuation-logs-v1.tar.gz'
  with tarfile.open(logs,'x:gz') as t:
   for p in sorted(B.iterdir()):
    if p.is_file():t.add(p,arcname=p.name)
  publish([receipt,logs],'research: preserve midpoint continuation actual execution receipts')
 if state['status']!='completed-await-independent-review':raise SystemExit(1)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
