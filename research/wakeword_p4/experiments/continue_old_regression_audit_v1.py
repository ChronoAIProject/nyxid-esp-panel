"""Observe original collector/reducer shell, then once-only audit/publish/supplement."""
import argparse,json,hashlib,subprocess,time,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');P=R/'4090-old-regression-audit-continuation-plan-v1.json';OUT=Path('.local/wakeword/old-regression-audit-continuation-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(pid):
 try:
  root=Path('/proc')/str(pid);fields=(root/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,starttime=fields[19],state=fields[0],cmdline=(root/'cmdline').read_bytes().decode().split('\0')[:-1])
 except FileNotFoundError:return None

def plan():
 shell=identity(1293911);collector=identity(1293912);assert shell and collector and shell['state']!='Z' and collector['state']!='Z';assert 'collect_old_fixed_grid_regression_v2.py' in ' '.join(shell['cmdline']) and collector['cmdline'][-2:]==['research/wakeword_p4/experiments/collect_old_fixed_grid_regression_v2.py','run']
 paths=[Path(__file__).resolve().relative_to(Path.cwd()),R/'4090-old-fixed-grid-regression-plan-v2.json'];paths += [Path(__file__).with_name(n) for n in ('audit_old_fixed_grid_regression_v2.py','supplement_old_regression_boundaries_v1.py','reduce_old_fixed_grid_regression_v2.py','collect_old_fixed_grid_regression_v2.py','hey_development_metrics_v1.py','run_hey_broad_trained_quantization_v1.py','plan_hey_broad_trained_quantization_v1.py')]
 with P.open('x') as f:json.dump(dict(shell=shell,collector=collector,files={str(p):sha(p) for p in paths},policy='Wait original shellcollector/reducer terminal only, never launch collection/reducer or retry. Require complete1860collection/1242metrics, independent audit once then publish unchanged reports+audit, then once additive fourboundary supplement and publish. Any missing/hash/process/error/gitfailure stops with failure receipt. No recognition-based selection, sourcefiltering, freshdev/acceptance/modelinvoke or promotion.'),f,indent=2);f.write('\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text());OUT.mkdir(exist_ok=False)
 def validate():
  for p,h in lock['files'].items():assert sha(p)==h,p
 def original_live(saved):
  current=identity(saved['pid'])
  if current is None or current['starttime']!=saved['starttime'] or current['state']=='Z':return False
  if current['cmdline']!=saved['cmdline']:
   # Bash may exec its already-authorized final reducer in the same process.
   assert saved['pid']==lock['shell']['pid'] and current['cmdline'] in [['python3','-u','research/wakeword_p4/experiments/reduce_old_fixed_grid_regression_v2.py'],['/usr/bin/python3','-u','research/wakeword_p4/experiments/reduce_old_fixed_grid_regression_v2.py']],current
  return True
 def execute(script):
  with (OUT/(script+'.log')).open('x') as log:subprocess.run(['python3',str(Path(__file__).with_name(script+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
 def publish_files(paths,message):
  subprocess.run(['git','diff','--cached','--exit-code','--quiet'],check=True);subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True);subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','HEAD:research-hye-nyx-voice-gpu'],check=True)
 try:
  validate()
  while original_live(lock['shell']):
   original_live(lock['collector']);time.sleep(15)
  validate();collection=R/'4090-old-fixed-grid-regression-collection-v2.json';metrics=R/'4090-old-fixed-grid-regression-metrics-v2.json';c=json.loads(collection.read_text());m=json.loads(metrics.read_text());assert len(c['records'])==1860 and len(m['points'])==1242 and m['collection_sha256']==sha(collection)
  execute('audit_old_fixed_grid_regression_v2');audit=R/'4090-old-fixed-grid-regression-independent-audit-v2.json';assert json.loads(audit.read_text())['metrics_sha256']==sha(metrics)
  publish_files([collection,metrics,audit],'research: publish complete old regression all fixed points and independent audit')
  validate();execute('supplement_old_regression_boundaries_v1');supplement=R/'4090-old-fixed-grid-regression-boundary-supplement-v1.json';assert len(json.loads(supplement.read_text())['points'])==1242
  publish_files([supplement],'research: publish additive all-four-generated-boundary regression offsets')
  receipt=dict(plan_sha256=sha(P),files={str(p):sha(p) for p in (collection,metrics,audit,supplement)},all_completed=True,model_invocations=0,limits=lock['policy'])
  final=R/'4090-old-regression-audit-continuation-result-v1.json'
  with final.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
  publish_files([final],'research: record original old regression audit continuation completion')
 except Exception as error:
  with (OUT/'failure.json').open('x') as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),plan_sha256=sha(P),retry_authorized=False),f,indent=2);f.write('\n')
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
