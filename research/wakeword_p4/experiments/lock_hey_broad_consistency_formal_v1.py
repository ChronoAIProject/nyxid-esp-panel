"""Bind audited actual smoke before formal launch, without mutating smoke runner."""
import json,hashlib,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 smoke=R/'4090-hey-broad-consistency-smoke-execution-v1.json';result=R/'4090-hey-broad-consistency-smoke-result-v1.json';audit=R/'4090-hey-broad-consistency-smoke-independent-audit-v1.json'
 for p in [smoke,result,audit]:assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)])==p.read_bytes()
 r=json.loads(result.read_text());a=json.loads(audit.read_text());assert r['all_passed'] and len(r['cases'])==len(a['cases'])==4 and a['result_sha256']==sha(result)
 lock=json.loads(smoke.read_text());assert r['execution_sha256']==sha(smoke)
 for target in ['hey_willow','hey_nyx']:
  first=[json.loads((Path(lock['output'])/f'{target}-{c}-0/training.json').read_text())['history'][0] for c in lock['conditions']]
  for k in ['supervised_loss','consistency_loss','supervised_gradient_l2','consistency_gradient_l2']:assert first[0][k]==first[1][k]
 for p in [smoke,result,audit,Path(__file__),Path(__file__).with_name('audit_hey_broad_consistency_smoke_v1.py')]:lock['files'][str(p)]=sha(p)
 for f,h in lock['files'].items():assert sha(f)==h,f
 lock.update(mode='formal',steps=10000,repeats=1,output='.local/wakeword/hey-broad-consistency-formal-v1',smoke_cross_condition_initial_components_exact=True)
 dest=R/'4090-hey-broad-consistency-formal-execution-v1.json'
 with dest.open('x') as f:json.dump(lock,f,indent=2);f.write('\n')
 print(str(dest),sha(dest))
if __name__=='__main__':main()
