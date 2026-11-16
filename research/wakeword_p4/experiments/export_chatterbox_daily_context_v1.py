"""One final daily-negative intervention export, exercised original INT8 ABI."""
import json
from pathlib import Path
import sys
import export_chatterbox_context_pair_v1 as parent
import train_chatterbox_daily_context_v1 as training
CONDITION='daily-context'
def plan():
    parent.training_lock=lambda condition:training.LOCK
    parent.training_output=lambda condition:training.OUT
    parent.training_report=lambda condition:training.R/training.REPORT
    status=parent.plan(CONDITION)
    if status:return status
    path=parent.lock(CONDITION);record=json.loads(path.read_text())
    record['files'][str(Path(__file__))]=parent.exercised.sha(Path(__file__))
    record['policy']='Single final daily-negative intervention export; unchanged previous representative as calibration control, no retry/tuning. C161 baseline. '+record['policy']
    path.write_text(json.dumps(record,indent=2)+'\n')
    return 0
def run():parent.run(CONDITION)
if __name__=='__main__':raise SystemExit({'plan':plan,'run':run}[sys.argv[1]]())
