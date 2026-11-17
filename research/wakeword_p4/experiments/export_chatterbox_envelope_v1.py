"""One final envelope candidate export; same original representative control."""
import json
from pathlib import Path
import sys
import export_chatterbox_context_pair_v1 as parent
import train_c161_envelope_v1 as training
CONDITION='envelope'
def plan():
    parent.training_lock=lambda condition:training.LOCK
    parent.training_output=lambda condition:training.OUT
    parent.training_report=lambda condition:training.R/'4090-c161-envelope-training-original-v1.json'
    status=parent.plan(CONDITION)
    if status:return status
    path=parent.lock(CONDITION);record=json.loads(path.read_text())
    record['files'][str(Path(__file__))]=parent.exercised.sha(Path(__file__))
    record['policy']='Single final envelope candidate, original representative unchanged as control, no retry/calibration rescue. C161 baseline. '+record['policy'].replace('Both arm outcomes reported, C161 retained.','Single candidate outcome reported, C161 retained.')
    path.write_text(json.dumps(record,indent=2)+'\n')
    return 0
def run():parent.run(CONDITION)
if __name__=='__main__':raise SystemExit({'plan':plan,'run':run}[sys.argv[1]]())
