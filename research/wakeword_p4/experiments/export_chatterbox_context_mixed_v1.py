"""Single final mixed-arm export using exercised paired export implementation."""
import json
from pathlib import Path
import sys
import export_chatterbox_context_pair_v1 as parent
import train_chatterbox_context_mixed_v1 as training

def plan():
    parent.training_lock=lambda condition:training.LOCK
    parent.training_output=lambda condition:training.OUT
    parent.training_report=lambda condition:training.R/training.REPORT
    status=parent.plan('mixed')
    if status:return status
    path=parent.lock('mixed');record=json.loads(path.read_text())
    record['files'][str(Path(__file__))]=parent.exercised.sha(Path(__file__))
    record['policy']=('One final mixed-context model export, same unchanged representative and ABI as both prior arms. No calibration or checkpoint retry. C161 remains baseline. '+record['policy'])
    path.write_text(json.dumps(record,indent=2)+'\n')
    return 0

def run():
    parent.run('mixed')

if __name__=='__main__':raise SystemExit({'plan':plan,'run':run}[sys.argv[1]]())
