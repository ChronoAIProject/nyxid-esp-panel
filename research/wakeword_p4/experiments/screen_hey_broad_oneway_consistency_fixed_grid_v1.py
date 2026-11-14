"""Fixed-grid dualbranch final-only full540 evaluator binding; no newdata."""
import argparse,json
from pathlib import Path
import screen_hey_broad_paired_qat_v1 as base
from run_hey_broad_trained_quantization_v1 import published
from plan_hey_broad_trained_quantization_v1 import sha
R=base.ROOT/'results';PLAN=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-plan-v1.json';LOCK=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-execution-v1.json'
def plan():
 original=R/'4090-hey-broad-oneway-consistency-formal-execution-v1.json';old=json.loads(original.read_text());files=dict(old['files'])
 compiled=R/'4090-hey-broad-oneway-consistency-fixed-grid-compile-result-v1.json';formal=R/'4090-hey-broad-oneway-consistency-formal-result-v1.json';constant=R/'4090-hey-broad-oneway-consistency-fixed-grid-constant-audit-v1.json'
 for p in [Path(__file__),Path(base.__file__),original,compiled,formal,constant,R/'4090-hey-broad-oneway-consistency-fixed-grid-compile-plan-v1.json',R/'4090-hey-broad-oneway-consistency-formal-independent-audit-v1.json']:files[str(p)]=sha(p)
 exports=json.loads(compiled.read_text())['arms'];trained=json.loads(formal.read_text());assert len(exports)==len(trained['cases'])==4 and trained['all_passed'];assert json.loads(constant.read_text())['all_passed']
 weights=[]
 for case in trained['cases']:
  assert case['steps']==10000;w=case['final_repeat'];weights.append(dict(target=case['target'],condition=case['condition'],path=w['weights_path']));files[w['weights_path']]=w['weights_sha256']
 for arm in exports:
  assert arm['only_allowed_buffers_changed'] and arm['original_grids_exact'] and len(arm['initializers'])==6 and all(v['all_real_zero'] for v in arm['initializers']);files[arm['path']]=arm['sha256']
 with PLAN.open('x') as f:json.dump(dict(files=files,versions=old['versions'],weights=weights,exports=exports,policy='Allfour final10000 fixedgrid exports x original135 completePCM =540; unchanged wholewave runner/gates/zero-state/repeat/nativefrontend/finalpad/phase/clocks. Save all failures. Runtime/reference separately required. No freshdev/acceptance or model selection.'),f,indent=2);f.write('\n')
def lock():
 published(PLAN);record=json.loads(PLAN.read_text())
 for p,h in record['files'].items():assert sha(p)==h,p
 record['files'][str(PLAN)]=sha(PLAN)
 with LOCK.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
def run():
 base.LOCK=LOCK;base.RESULT=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-result-v1.json';base.OUT=Path('.local/wakeword/hey-broad-oneway-consistency-fixed-grid-screen-v1');base.run()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','lock','run']);a=p.parse_args();{'plan':plan,'lock':lock,'run':run}[a.action]()
