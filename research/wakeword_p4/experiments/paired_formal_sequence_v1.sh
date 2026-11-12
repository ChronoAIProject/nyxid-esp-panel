#!/usr/bin/env bash
# One predeclared run per arm. Fail closed; no automatic restart.
set -euo pipefail
bash research/wakeword_p4/experiments/paired_control_formal_v1.sh
bash research/wakeword_p4/experiments/paired_treatment_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_paired_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/paired_after_training_v1.py --arm control > .local/wakeword/paired-control-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/paired_after_training_v1.py --arm treatment > .local/wakeword/paired-treatment-formal-v1-postprocess.log 2>&1
