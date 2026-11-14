"""Supplemental scoring lock binds completed traces/audit and SciPy FAR runtime."""
import importlib.metadata,json,subprocess,sys
from pathlib import Path
from collect_fp32_reused_development_v1 import R,LOCK,OUT,sha
SCORE=R/'4090-fp32-reused-scoring-execution-v1.json'
def lock():
 p=json.loads(LOCK.read_text());result=OUT/'result.json';audit=R/'4090-fp32-reused-collection-independent-audit-v1.json'
 r=json.loads(result.read_text());a=json.loads(audit.read_text());assert r['all_passed'] and a['all_passed'] and r['execution_sha256']==a['execution_sha256']==sha(LOCK)
 files=dict(p['files'])
 paths=[LOCK,result,audit,Path(__file__),Path('research/wakeword_p4/experiments/audit_fp32_reused_collection_v1.py')]
 for m in r['models']:
  rp=OUT/m['model']/'report.json';assert sha(rp)==m['report_sha256'];paths.append(rp)
  for c in json.loads(rp.read_text())['records']:
   assert sha(c['trace_path'])==c['trace_sha256'];paths.append(Path(c['trace_path']))
 for path in paths:files[str(path)]=sha(path)
 with SCORE.open('x') as f:f.write(json.dumps(dict(files=files,versions={n:importlib.metadata.version(n) for n in ['numpy','scipy']},grid=p['grid'],threshold_selected=False,limits='All completed original1488 traces/audit fixed before scoring. Same original scorer/policy, no selection or freshdata. SciPy chi2 one-sided95%FAR runtime bound explicitly.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(SCORE)])==SCORE.read_bytes();p=json.loads(SCORE.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 subprocess.run([sys.executable,'research/wakeword_p4/experiments/score_fp32_reused_development_v1.py'],check=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
