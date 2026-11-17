"""Prelocked sequential paired training, final exports and complete consumed replay."""
import gzip,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');BASE='research/wakeword_p4/experiments/';ROOT=Path('.local/wakeword/binary40-multiscale-continuation-v1');PLAN=R/'4090-binary40-multiscale-training-plan-v1.json';RECEIPT=R/'4090-binary40-multiscale-continuation-original-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();plan=json.loads(PLAN.read_text())
 for path,h in plan['files'].items():assert sha(path)==h,path
 ROOT.mkdir(exist_ok=False);state=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(PLAN),status='running',stages=[],actual_training_subprocess_receipts=[])
 def save():
  raw=json.dumps(state,indent=2)+'\n';(ROOT/'state.json').write_text(raw);RECEIPT.write_text(raw)
 def stage(name,args,arm=None):
  log=ROOT/(name+'.log');started=time.time()
  with log.open('x') as f:
   child=subprocess.Popen(['/usr/bin/python3',*args],stdout=f,stderr=subprocess.STDOUT);state['live']=dict(stage=name,pid=child.pid,started_unix=started);save();code=child.wait()
  row=dict(name=name,pid=child.pid,actual_exit_code=code,started_unix=started,ended_unix=time.time(),log=str(log),log_sha256=sha(log));state['stages'].append(row);state.pop('live',None)
  if arm:
   state['actual_training_subprocess_receipts'].append(dict(arm=arm,pid=child.pid,exit_code=code,log=str(log),log_sha256=sha(log)));(R/'4090-binary40-multiscale-training-receipts-v1.json').write_text(json.dumps(state['actual_training_subprocess_receipts'],indent=2)+'\n')
  save();print(json.dumps(row),flush=True)
  if code:raise RuntimeError(f'{name} actualexit{code}')
 def publish(paths,message):
  paths=[str(p) for p in paths];subprocess.run(['git','add','--',*paths],check=True)
  diff=subprocess.run(['git','diff','--cached','--quiet','--',*paths]).returncode
  if diff:subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 try:
  save()
  for arm in plan['arms']:stage('train-'+arm,['research/wakeword_p4/training/gpu_python.py',BASE+'train_binary40_multiscale_pair_v1.py','run',arm],arm)
  stage('training-review',[BASE+'review_binary40_multiscale_training_v1.py'])
  publish([R/f'4090-binary40-multiscale-{a}-training-original-v1.json' for a in plan['arms']]+[R/'4090-binary40-multiscale-training-review-v1.json',R/'4090-binary40-multiscale-training-receipts-v1.json'],'research: preserve matched multiscale terminal training and full batch audit')
  for arm in plan['arms']:
   stage('export-plan-'+arm,[BASE+'export_binary40_multiscale_final_v1.py','plan',arm]);publish([R/f'4090-binary40-multiscale-{arm}-export-plan-v1.json'],'research: bind final multiscale arm training-state export')
   stage('export-'+arm,['research/wakeword_p4/training/gpu_python.py',BASE+'export_binary40_multiscale_final_v1.py','run',arm]);e=json.loads((R/f'4090-binary40-multiscale-{arm}-export-original-v1.json').read_text());model=R/f'4090-binary40-multiscale-{arm}-model-v1.tflite.gz'
   with model.open('xb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as f:f.write(Path(e['model']['path']).read_bytes())
   publish([R/f'4090-binary40-multiscale-{arm}-export-original-v1.json',model],'research: preserve exact final multiscale arm INT8 bytes')
   for c in plan['consumed_scope']['cohorts']:stage('replay-plan-'+arm+'-'+c,[BASE+'replay_binary40_multiscale_v1.py','plan',c,arm])
   stage('context-plan-'+arm,[BASE+'run_binary40_multiscale_context_v1.py','plan',arm]);publish([R/f'4090-binary40-multiscale-{arm}-{c}-replay-plan-v1.json' for c in plan['consumed_scope']['cohorts']]+[R/f'4090-binary40-multiscale-{arm}-context-plan-v1.json'],'research: lock complete final multiscale arm consumed scope')
   for c in plan['consumed_scope']['cohorts']:
    stage('replay-'+arm+'-'+c,[BASE+'replay_binary40_multiscale_v1.py','run',c,arm]);stage('score-'+arm+'-'+c,[BASE+'score_binary40_multiscale_v1.py',c,arm]);publish([R/f'4090-binary40-multiscale-{arm}-{c}-comparison-v1.json',R/f'4090-binary40-multiscale-{arm}-{c}-evidence-v1.tar.gz'],f'research: preserve multiscale {arm} {c} full raw evidence')
   stage('context-'+arm,[BASE+'run_binary40_multiscale_context_v1.py','run',arm]);publish([R/f'4090-binary40-multiscale-{arm}-context-original-v1.json'],'research: preserve multiscale arm complete physical context streams')
  state['status']='completed-await-independent-review';save();stage('consumed-review',[BASE+'review_binary40_multiscale_consumed_v1.py']);state['status']='completed-reviewed';save();publish([RECEIPT,R/'4090-binary40-multiscale-consumed-review-v1.json'],'research: publish complete multiscale cycle independent evidence audit')
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(ROOT/'failure.log').write_text(state['error']);save();raise
if __name__=='__main__':main()
