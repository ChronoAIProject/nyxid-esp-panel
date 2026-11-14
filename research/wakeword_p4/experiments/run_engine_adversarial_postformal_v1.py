"""One-shot continuation of the original live job. No training launch or retry."""
import datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments')
LOCK=R/'4090-engine-adversarial-postformal-execution-v1.json'
RESULT=R/'4090-engine-adversarial-postformal-result-v1.json'
OUT=Path('.local/wakeword/engine-adversarial-postformal-v1')
FORMAL=Path('.local/wakeword/engine-adversarial-formal-v1')
BRANCH='research-hye-nyx-voice-gpu'
SCRIPTS=['audit_engine_adversarial_formal_v1.py','export_engine_adversarial_v1.py','collect_engine_adversarial_development_v1.py','audit_engine_adversarial_collection_v1.py','lock_engine_adversarial_scoring_v1.py','score_engine_adversarial_development_v1.py','audit_engine_adversarial_scoring_v1.py','report_engine_adversarial_comparison_v1.py','screen_engine_adversarial_v1.py','audit_engine_adversarial_screen_v1.py','run_engine_adversarial_postformal_v1.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def process(pid):
 try:
  tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return dict(state=tail[0],starttime=tail[19])
 except FileNotFoundError:return None

def lock():
 startup=FORMAL/'startup.json';s=json.loads(startup.read_text());p=process(s['pid'])
 assert p and p['state']!='Z' and p['starttime']==str(s['starttime'])
 assert not RESULT.exists() and not OUT.exists()
 assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
 paths=[startup,R/'4090-engine-adversarial-formal-execution-v1.json',E/'engine_adversarial_effect_evaluation_v1.md',E/'engine_adversarial_postformal_v1.md',Path('research/wakeword_p4/training/gpu_python.py')]+[E/n for n in SCRIPTS]
 data=dict(created_utc=utc(),original_startup=s,files={str(p):sha(p) for p in paths},poll_seconds=30,policy='Wait only for original PID/starttime; never launch/restart training. Genuine terminal plus original four-arm result then full audit. Publish stage locks before each once-only action. Actual INT8 consumed usability first, original waveform artifact validation last. Preserve failures; stop on command error, no retries or threshold/candidate selection. Automated git publication includes only named research outputs; no issue messages or model promotion.')
 with LOCK.open('x') as f:f.write(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data))

def run():
 assert subprocess.check_output(['git','show','origin/'+BRANCH+':'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
 OUT.mkdir(exist_ok=False);receipts=[]
 state=dict(started_utc=utc(),pid=os.getpid(),execution_sha256=sha(LOCK),steps=receipts,workflow_completed=False,candidate_promoted=False,threshold_selected=False)
 (OUT/'startup.json').write_text(json.dumps(state,indent=2)+'\n')
 def command(label,args):
  log=OUT/(label+'.log');began=utc()
  with log.open('xb') as f:code=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT).returncode
  receipt=dict(label=label,args=args,started_utc=began,finished_utc=utc(),returncode=code,log_path=str(log),log_sha256=sha(log));receipts.append(receipt)
  (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(receipt),flush=True)
  if code:raise RuntimeError(f'{label} exited {code}; no retry')
 def script(label,name,*args,gpu=False):
  command(label,[sys.executable]+(['research/wakeword_p4/training/gpu_python.py'] if gpu else [])+[str(E/name),*args])
 def publish(label,paths):
  assert not subprocess.check_output(['git','diff','--cached','--name-only']).strip(),'Unrelated staged changes; stopping'
  assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
  paths=[str(Path(x)) for x in paths]
  command(label+'-add',['git','add','--',*paths])
  if not subprocess.check_output(['git','diff','--cached','--name-only']).strip():return
  command(label+'-commit',['git','commit','-m',label])
  command(label+'-push',['git','push','origin','HEAD:'+BRANCH])
 def copy_original(src,dst):
  src=Path(src);dst=Path(dst)
  if dst.exists():assert dst.read_bytes()==src.read_bytes()
  else:
   with dst.open('xb') as f:f.write(src.read_bytes())
  assert sha(src)==sha(dst)
  return dst
 try:
  original=p['original_startup'];last=None
  while True:
   proc=process(original['pid'])
   if proc is None or proc['state']=='Z' or proc['starttime']!=str(original['starttime']):break
   latest=dict(utc=utc(),original_pid=original['pid'],actual_process=proc)
   (OUT/'last-live.json').write_text(json.dumps(latest,indent=2)+'\n')
   # Publish each producer's immutable report once while later arms continue.
   for target in ['hey_nyx','hey_willow']:
    for condition in ['control','adversarial']:
     src=FORMAL/f'{target}-{condition}'/'training.json'
     dst=R/f'4090-engine-adversarial-{target.replace("_","-")}-{condition}-training-original-v1.json'
     if src.exists() and not dst.exists():
      report=json.loads(src.read_text());assert report['target']==target and report['condition']==condition and report['steps']==report['optimizer_iterations']==10000
      publish('Publish original '+target+' '+condition+' final training report',[copy_original(src,dst)])
   if latest['actual_process']!=last:print(json.dumps(latest),flush=True);last=latest['actual_process']
   time.sleep(p['poll_seconds'])
  formal=R/'4090-engine-adversarial-formal-result-v1.json'
  assert formal.exists() and not (FORMAL/'failure.json').exists(),'Original job terminal without successful result; no restart'
  f=json.loads(formal.read_text());assert f['all_passed'] and len(f['cases'])==4
  script('full-training-audit','audit_engine_adversarial_formal_v1.py')
  reports=[formal,R/'4090-engine-adversarial-formal-independent-audit-v1.json']
  for c in f['cases']:
   dst=R/f'4090-engine-adversarial-{c["target"].replace("_","-")}-{c["condition"]}-training-original-v1.json'
   reports.append(copy_original(Path(c['folder'])/'training.json',dst))
  publish('Publish completed original four-arm training and full audit',reports)
  script('export-lock','export_engine_adversarial_v1.py','lock')
  publish('Lock final engine-adversarial FP32 and INT8 export',[R/'4090-engine-adversarial-export-execution-v1.json'])
  script('export-run','export_engine_adversarial_v1.py','run',gpu=True)
  exported=copy_original('.local/wakeword/engine-adversarial-export-v1/result.json',R/'4090-engine-adversarial-export-original-result-v1.json')
  publish('Publish original final engine-adversarial export metadata',[exported])
  script('collection-lock','collect_engine_adversarial_development_v1.py','lock')
  publish('Lock complete consumed INT8 wake-effect collection',[R/'4090-engine-adversarial-development-execution-v1.json'])
  script('collection-run','collect_engine_adversarial_development_v1.py','run')
  script('collection-audit','audit_engine_adversarial_collection_v1.py')
  collected=copy_original('.local/wakeword/engine-adversarial-development-v1/result.json',R/'4090-engine-adversarial-development-original-result-v1.json')
  publish('Publish complete INT8 trace collection and saved audit',[collected,R/'4090-engine-adversarial-collection-independent-audit-v1.json'])
  script('scoring-lock','lock_engine_adversarial_scoring_v1.py','lock')
  publish('Lock original UINT8 complete wake-effect scoring',[R/'4090-engine-adversarial-scoring-execution-v1.json'])
  script('scoring-run','lock_engine_adversarial_scoring_v1.py','run')
  script('scoring-audit','audit_engine_adversarial_scoring_v1.py')
  script('paired-report','report_engine_adversarial_comparison_v1.py')
  summary=copy_original('.local/wakeword/engine-adversarial-development-v1/diagnostic/summary.json',R/'4090-engine-adversarial-int8-diagnostic-original-summary-v1.json')
  publish('Publish actual INT8 wake effects and all paired reference curves',[summary,R/'4090-engine-adversarial-scoring-independent-audit-v1.json',R/'4090-engine-adversarial-comparison-v1.json'])
  script('waveform-lock','screen_engine_adversarial_v1.py','lock')
  publish('Lock original final-export waveform artifact validation',[R/'4090-engine-adversarial-screen-execution-v1.json'])
  script('waveform-run','screen_engine_adversarial_v1.py','run',gpu=True)
  script('waveform-audit','audit_engine_adversarial_screen_v1.py')
  publish('Publish all final INT8 waveform artifact outcomes',[R/'4090-engine-adversarial-screen-result-v1.json',R/'4090-engine-adversarial-screen-independent-audit-v1.json'])
  state.update(workflow_completed=True,finished_utc=utc(),limits='Completed fixed consumed-data diagnostic workflow only; failures in stored numerical/recognition outcomes remain failures. No fresh threshold selection, source qualification, independent acceptance or promotion. Goal not complete.')
 except Exception:
  state.update(finished_utc=utc(),failure=traceback.format_exc(),automatic_retry=False)
  with RESULT.open('x') as f:f.write(json.dumps(state,indent=2)+'\n')
  print(state['failure'],flush=True);raise
 with RESULT.open('x') as f:f.write(json.dumps(state,indent=2)+'\n')
 publish('Publish once-only postformal workflow receipt',[RESULT])

if __name__=='__main__':raise SystemExit({'lock':lock,'run':run}[sys.argv[1]]())
