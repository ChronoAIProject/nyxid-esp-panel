#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/experiments/verify_per_draw_noise_execution_pins_v1.py
bash research/wakeword_p4/experiments/per_draw_noise_fixed_formal_v1.sh
bash research/wakeword_p4/experiments/per_draw_noise_per_draw_formal_v1.sh
python3 research/wakeword_p4/experiments/audit_per_draw_noise_formal_inputs_v1.py
python3 research/wakeword_p4/experiments/per_draw_noise_after_training_v1.py --arm fixed > .local/wakeword/per-draw-noise-fixed-formal-v1-postprocess.log 2>&1
python3 research/wakeword_p4/experiments/per_draw_noise_after_training_v1.py --arm per_draw > .local/wakeword/per-draw-noise-per_draw-formal-v1-postprocess.log 2>&1
