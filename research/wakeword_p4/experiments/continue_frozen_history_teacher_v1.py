"""Wait for the existing paired supervisor; review/export/replay sequentially."""
import gzip,json,os,subprocess,sys,time,traceback
from pathlib import Path
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/frozen-history-teacher-continuation-v1')
ARMS=['control','teacher'];COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']

def main():
 parent=int(sys.argv[1]);identity=sys.argv[2];stat=Path(f'/proc/{parent}/stat')
 assert stat.read_text().split()[21]==identity
 B.mkdir(exist_ok=False);state=dict(schema='frozen-history-teacher-continuation-v1',pid=os.getpid(),training_supervisor_pid=parent,training_supervisor_starttime=identity,status='waiting-existing-paired-training',stages=[])
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def publish(paths,message):
  paths=[str(p) for p in paths];assert all(Path(p).exists() for p in paths)
  subprocess.run(['git','add','--',*paths],check=True)
  code=subprocess.run(['git','diff','--cached','--quiet','--',*paths]).returncode;assert code in [0,1]
  if code:
   subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 def stage(name,args):
  log=B/(name+'.log');start=time.monotonic()
  with log.open('x') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,args=args,actual_exit_code=p.returncode,elapsed_seconds=time.monotonic()-start,log=str(log),log_sha256=sha(log));state['stages'].append(row);save();print(json.dumps(row),flush=True)
  assert p.returncode==0,row
 py='/usr/bin/python3';base='research/wakeword_p4/experiments/';gpu='research/wakeword_p4/training/gpu_python.py'
 try:
  save()
  while stat.exists():
   try:fields=stat.read_text().split()
   except FileNotFoundError:break
   if fields[21]!=identity or fields[2]=='Z':break
   time.sleep(5)
  terminal_path=R/'4090-frozen-history-teacher-training-terminal-v1.json'
  terminal=json.loads(terminal_path.read_text());assert terminal['status']=='completed-await-independent-review'
  receipts=terminal['stages'];assert [r['arm'] for r in receipts]==ARMS and all(r['actual_exit_code']==0 for r in receipts)
  state.update(status='reviewing-final-artifacts',actual_training_subprocess_receipts=receipts);save()
  stage('review-pair',[py,base+'review_frozen_history_teacher_training_v1.py'])
  evidence=[R/'4090-frozen-history-teacher-training-review-v1.json',terminal_path]
  for arm in ARMS:
   evidence.append(R/f'4090-frozen-history-teacher-{arm}-training-original-v1.json')
   dest=R/f'4090-frozen-history-teacher-{arm}-training-execution-v1.log.gz';dest.write_bytes(gzip.compress(Path(next(r['log_path'] for r in receipts if r['arm']==arm)).read_bytes(),mtime=0));evidence.append(dest)
  publish(evidence,'research: preserve reviewed paired frozen teacher training and actual exits')
  for arm in ARMS:stage('plan-export-'+arm,[py,base+'export_frozen_history_teacher_v1.py','plan',arm])
  publish([R/f'4090-frozen-history-teacher-{arm}-export-plan-v1.json' for arm in ARMS],'research: lock both final frozen teacher model exports')
  for arm in ARMS:
   stage('export-'+arm,[py,gpu,base+'export_frozen_history_teacher_v1.py','run',arm])
   path=R/f'4090-frozen-history-teacher-{arm}-export-original-v1.json';r=json.loads(path.read_text());model=Path(r['model']['path']);assert sha(model)==r['model']['sha256']
   dest=R/f'4090-frozen-history-teacher-{arm}-model-v1.tflite.gz';dest.write_bytes(gzip.compress(model.read_bytes(),mtime=0))
   publish([path,dest],'research: preserve final frozen teacher '+arm+' integer export')
  for arm in ARMS:
   for c in COHORTS:stage('plan-'+arm+'-'+c,[py,base+'replay_frozen_history_teacher_v1.py','plan',c,arm])
  publish([R/f'4090-frozen-history-teacher-{arm}-{c}-replay-plan-v1.json' for arm in ARMS for c in COHORTS],'research: lock complete consumed scope for both frozen teacher arms')
  state['status']='running-complete-consumed-replays';save()
  for arm in ARMS:
   for c in COHORTS:
    stage('replay-'+arm+'-'+c,[py,base+'replay_frozen_history_teacher_v1.py','run',c,arm])
    stage('score-'+arm+'-'+c,[py,base+'score_frozen_history_teacher_v1.py',c,arm])
    publish([R/f'4090-frozen-history-teacher-{arm}-{c}-comparison-v1.json',R/f'4090-frozen-history-teacher-{arm}-{c}-evidence-v1.tar.gz'],'research: preserve '+arm+' '+c+' complete consumed evidence')
  state['status']='completed-await-independent-review'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(B/'failure.log').write_text(state['error'])
 finally:
  save();dest=R/'4090-frozen-history-teacher-continuation-original-v1.json';dest.write_text(json.dumps(state,indent=2)+'\n')
  logs=R/'4090-frozen-history-teacher-continuation-logs-v1.tar.gz'
  import tarfile
  with tarfile.open(logs,'x:gz') as t:
   for f in sorted(B.iterdir()):
    if f.is_file():t.add(f,arcname=f.name)
  publish([dest,logs],'research: preserve frozen teacher continuation execution receipts')
 if state['status']!='completed-await-independent-review':raise SystemExit(1)
if __name__=='__main__':main()
