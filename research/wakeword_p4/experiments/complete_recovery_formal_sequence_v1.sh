#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/experiments/verify_complete_recovery_execution_pins_v1.py
bash research/wakeword_p4/experiments/complete_recovery_control_formal_v1.sh
bash research/wakeword_p4/experiments/complete_recovery_recovered_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_complete_recovery_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/complete_recovery_after_training_v1.py --arm control > .local/wakeword/complete-recovery-control-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/complete_recovery_after_training_v1.py --arm recovered > .local/wakeword/complete-recovery-recovered-formal-v1-postprocess.log 2>&1
