"""Once-only bound-process continuation, no newdev/promotion or retry."""
import argparse,json,hashlib,subprocess,time,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments');P=R/'4090-hey-broad-consistency-continuation-plan-v1.json';OUT=Path('.local/wakeword/hey-broad-consistency-continuation-v1')
SCRIPTS=['audit_hey_broad_consistency_formal_v1.py','compile_hey_broad_consistency_fixed_grid_v1.py','audit_hey_broad_consistency_fixed_grid_constants_v1.py','screen_hey_broad_consistency_fixed_grid_v1.py','audit_hey_broad_consistency_fixed_grid_screen_v1.py','probe_hey_broad_consistency_fixed_grid_recursive_v1.py','audit_hey_broad_consistency_fixed_grid_recursive_v1.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
 p=Path(f'/proc/{pid}/stat')
 if not p.exists():return None
 fields=p.read_text().rsplit(') ',1)[1].split();return dict(starttime=fields[19],state=fields[0],cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
def plan(pid):
 ident=identity(pid);assert ident is not None and 'run_hey_broad_consistency_v1.py run formal' in ident['cmdline'];formal=R/'4090-hey-broad-consistency-formal-execution-v1.json';lock=json.loads(formal.read_text());files=dict(lock['files'])
 for p in [formal,Path(__file__),*[E/s for s in SCRIPTS]]:files[str(p)]=sha(p)
 with P.open('x') as f:json.dump(dict(pid=pid,identity=ident,files=files,policy='Observe only original PID/starttime; never train/restart. After terminal and complete final aggregate: audit/publish -> publish compileplan -> once compile -> independentconstantsaudit/publish -> publishscreenplan -> publishactualscreenlock -> once full540screen/audit/publish -> publishrecursiveplan -> once recursivefull540/audit/publish. Any missing/failed/hash/git step stops, never retry. Preserve all failures. No freshdev/acceptance/threshold/model selection/promotion.'),f,indent=2);f.write('\n')
def publish(paths,message):
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip(),'Unrelated staged files; stop'
 subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True)
 subprocess.run(['git','commit','-m',message],check=True)
 subprocess.run(['git','push','origin','HEAD:research-hye-nyx-voice-gpu'],check=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 OUT.mkdir(exist_ok=False);phases=[]
 def call(script,*args):
  for p,h in lock['files'].items():assert sha(p)==h,p
  name=script.removesuffix('.py')+('-'+'-'.join(args) if args else '');log=OUT/(name+'.log');assert not log.exists()
  print(json.dumps(dict(phase=name,event='start')),flush=True)
  with log.open('x') as f:subprocess.run([sys.executable,'research/wakeword_p4/training/gpu_python.py',str(E/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
  phases.append(dict(phase=name,log=str(log),log_sha256=sha(log)));print(json.dumps(dict(phase=name,event='complete')),flush=True)
 def result(name):return R/('4090-hey-broad-consistency-'+name+'-v1.json')
 try:
  while True:
   current=identity(lock['pid'])
   if current is None or current['state']=='Z':break
   assert current['starttime']==lock['identity']['starttime'] and current['cmdline']==lock['identity']['cmdline'],'Original PID identity changed'
   time.sleep(5)
  trained=json.loads(result('formal-result').read_text());assert trained['all_passed'] and len(trained['cases'])==4
  call(SCRIPTS[0]);publish([result('formal-result'),result('formal-independent-audit')],'research: publish complete dualbranch fixed10000 training and audit')
  call(SCRIPTS[1],'plan');publish([result('fixed-grid-compile-plan')],'research: publish actual final-weight consistency fixed-grid compile plan')
  call(SCRIPTS[1],'run');call(SCRIPTS[2]);publish([result('fixed-grid-compile-result'),result('fixed-grid-constant-audit')],'research: publish consistency fixed-grid exports and independent constants audit')
  call(SCRIPTS[3],'plan');publish([result('fixed-grid-screen-plan')],'research: publish consistency original full540 numerical plan')
  call(SCRIPTS[3],'lock');publish([result('fixed-grid-screen-execution')],'research: publish actual consistency full540 identity before inference')
  call(SCRIPTS[3],'run');call(SCRIPTS[4]);publish([result('fixed-grid-screen-result'),result('fixed-grid-screen-independent-audit')],'research: publish complete consistency full540 outcomes and audit')
  call(SCRIPTS[5],'plan');publish([result('fixed-grid-recursive-plan')],'research: publish actual consistency full540 recursive reference plan')
  call(SCRIPTS[5],'run');call(SCRIPTS[6]);publish([result('fixed-grid-recursive-result'),result('fixed-grid-recursive-independent-audit')],'research: publish complete consistency recursive reference outcomes and audit')
  record=dict(status='completed_no_admission',plan_sha256=sha(P),phases=phases,limits=lock['policy'])
  with result('continuation-result').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  publish([result('continuation-result')],'research: record single-pass consistency continuation completion')
 except Exception as error:
  record=dict(status='failed_stop_no_retry',plan_sha256=sha(P),error=repr(error),completed_phases=phases,limits=lock['policy'])
  with (OUT/'failure.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);p.add_argument('--pid',type=int);a=p.parse_args();plan(a.pid) if a.action=='plan' else run()
