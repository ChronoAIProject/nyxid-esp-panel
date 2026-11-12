#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/experiments/verify_phrase_ranking_execution_pins_v1.py
bash research/wakeword_p4/experiments/phrase_ranking_control_formal_v1.sh
bash research/wakeword_p4/experiments/phrase_ranking_treatment_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_phrase_ranking_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/phrase_ranking_after_training_v1.py --arm control > .local/wakeword/phrase-ranking-control-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/phrase_ranking_after_training_v1.py --arm treatment > .local/wakeword/phrase-ranking-treatment-formal-v1-postprocess.log 2>&1
