#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/experiments/verify_matched_noise_execution_pins_v1.py
bash research/wakeword_p4/experiments/matched_noise_independent_formal_v1.sh
bash research/wakeword_p4/experiments/matched_noise_common_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_matched_noise_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/matched_noise_after_training_v1.py --arm independent > .local/wakeword/matched-noise-independent-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/matched_noise_after_training_v1.py --arm common > .local/wakeword/matched-noise-common-formal-v1-postprocess.log 2>&1
