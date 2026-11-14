"""Separate prospective fixed-grid full540 screen; reuse unchanged wholewave evaluator."""
import argparse,json
from pathlib import Path
import screen_hey_broad_paired_qat_v1 as base
from run_hey_broad_trained_quantization_v1 import published
from plan_hey_broad_trained_quantization_v1 import sha
R=base.ROOT/'results'
PLAN=R/'4090-hey-broad-fixed-grid-screen-plan-v1.json'
LOCK=R/'4090-hey-broad-fixed-grid-screen-execution-v1.json'
def plan():
 original=R/'4090-hey-broad-paired-qat-screen-execution-v1.json'
 old=json.loads(original.read_text());files=dict(old['files'])
 for p in [Path(__file__),Path(base.__file__),original,R/'4090-hey-broad-fixed-grid-compile-plan-v1.json',R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json',R/'4090-hey-broad-fixed-grid-constant-audit-v1.json']:
  files[str(p)]=sha(p)
 recovered=json.loads((R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json').read_text())
 assert json.loads((R/'4090-hey-broad-fixed-grid-constant-audit-v1.json').read_text())['all_passed']
 for arm in recovered['arms']:files[arm['path']]=arm['sha256']
 record=dict(files=files,versions=old['versions'],weights=old['weights'],exports=recovered['arms'],policy='Exactly existing full540 numerical evaluator, same fourfinal H5 and original135 wholePCM/arm. Changed input only fixedgrid artifacts; separate output paths. No compiler/recalibration/retry/selection. All original gates/startup/repeat/clocks preserved. Runtime/reference parity is separate and required before admission; this screen alone cannot open freshdev. No freshdev or acceptance.')
 with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def lock():
 published(PLAN);record=json.loads(PLAN.read_text())
 for p,h in record['files'].items():assert sha(p)==h,p
 record['files'][str(PLAN)]=sha(PLAN)
 with LOCK.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def run():
 base.LOCK=LOCK;base.RESULT=R/'4090-hey-broad-fixed-grid-screen-result-v1.json';base.OUT=Path('.local/wakeword/hey-broad-fixed-grid-screen-v1');base.run()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','lock','run']);a=p.parse_args();{'plan':plan,'lock':lock,'run':run}[a.action]()
