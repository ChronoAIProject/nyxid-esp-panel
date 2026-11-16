"""All84 admitted training clips for mixed arm, no repeated baseline replay."""
import json
from pathlib import Path
import sys
import replay_training_carrier_support_v1 as parent
import export_chatterbox_context_pair_v1 as exporter
PLAN=parent.R/'4090-context-mixed-training-replay-plan-v1.json'
OUT=Path('.local/wakeword/context-mixed-training-replay-v1')
def plan():
    original=parent.PLAN;p=json.loads(original.read_text())
    er=exporter.report('mixed');e=json.loads(er.read_text());assert e['all_static_and_runtime_abi_passed']
    model=Path(e['model']['path']);assert parent.sha(model)==e['model']['sha256']
    p['models']=dict(mixed=str(model));paths=[Path(__file__),Path(parent.__file__),original,er,model]
    p['files'].update({str(f):parent.sha(f) for f in paths})
    p['policy']='Single mixed model on all84 original admitted training examples, training-fit diagnosis only. '+p['policy']
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
def run():
    parent.PLAN=PLAN;parent.OUT=OUT;parent.run()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
