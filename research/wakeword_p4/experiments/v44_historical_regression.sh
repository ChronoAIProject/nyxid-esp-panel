#!/usr/bin/env bash
# Historical development regression only; not new synthetic acceptance/calibration.
set -euo pipefail
model=.local/wakeword/model-gpu-v44-ami-extra
validation=.local/wakeword/eval-v44-ami-extra-validation
test -f "$model/training.json"
test ! -e "$validation/report.json"
python3 research/wakeword_p4/training/evaluate_stream.py \
  --model "$model/hey_nyx.tflite" --features .local/wakeword/features-gpu-v6-qwen \
  --output "$validation" --validation-only --threshold-step 0.00390625 \
  > .local/wakeword/v44-validation.log 2>&1
for corpus in mandarin dipco libri-dev-other; do
  case "$corpus" in
    mandarin) features=.local/wakeword/aishell-human-features-v1 ;;
    dipco) features=.local/wakeword/dipco-human-features-v1 ;;
    libri-dev-other) features=.local/wakeword/libri-dev-other-features-v1 ;;
  esac
  python3 research/wakeword_p4/training/evaluate_frozen_human.py \
    --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
    --features "$features" --corpus-name "$corpus" \
    --output ".local/wakeword/eval-v44-$corpus-validation"
done
python3 research/wakeword_p4/training/calibrate_multicorpus.py \
  --english-validation "$validation" \
  --negative-validation .local/wakeword/eval-v44-mandarin-validation \
  --negative-validation .local/wakeword/eval-v44-dipco-validation \
  --negative-validation .local/wakeword/eval-v44-libri-dev-other-validation \
  --output .local/wakeword/v44-expanded-validation-calibration.json
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
threshold=json.loads(Path('.local/wakeword/eval-v44-ami-extra-validation/report.json').read_text())['threshold']
subprocess.run(['python3','research/wakeword_p4/training/gpu_python.py',
 'research/wakeword_p4/training/compare_streaming.py',
 '--model-dir','.local/wakeword/model-gpu-v44-ami-extra',
 '--features','.local/wakeword/features-gpu-v6-qwen','--threshold',str(threshold),
 '--upstream','.local/wakeword/microWakeWord',
 '--output','.local/wakeword/v44-streaming-probe.json'],check=True)
PY
