#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/experiments/verify_temporal_tail_risk_execution_pins_v1.py
bash research/wakeword_p4/experiments/temporal_tail_risk_control_formal_v1.sh
bash research/wakeword_p4/experiments/temporal_tail_risk_treatment_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_temporal_tail_risk_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/temporal_tail_risk_after_training_v1.py --arm control > .local/wakeword/temporal-tail-risk-control-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/temporal_tail_risk_after_training_v1.py --arm treatment > .local/wakeword/temporal-tail-risk-treatment-formal-v1-postprocess.log 2>&1
