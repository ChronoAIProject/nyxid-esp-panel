#!/usr/bin/env bash
# Run only after v26 export and original dense validation finish successfully.
set -euo pipefail
model=.local/wakeword/model-gpu-v26-label-smoothing
validation=.local/wakeword/eval-v26-label-smoothing-validation
python3 research/wakeword_p4/training/evaluate_frozen_human.py \
  --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
  --features .local/wakeword/aishell-human-features-v1 \
  --output .local/wakeword/eval-v26-mandarin-validation --corpus-name Mandarin
python3 research/wakeword_p4/training/evaluate_frozen_human.py \
  --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
  --features .local/wakeword/dipco-human-features-v1 \
  --output .local/wakeword/eval-v26-dipco-validation --corpus-name DiPCo
python3 research/wakeword_p4/training/evaluate_frozen_human.py \
  --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
  --features .local/wakeword/libri-dev-other-features-v1 \
  --output .local/wakeword/eval-v26-libri-dev-other-validation --corpus-name LibriSpeech-dev-other
python3 research/wakeword_p4/training/calibrate_multicorpus.py \
  --english-validation "$validation" \
  --negative-validation .local/wakeword/eval-v26-mandarin-validation \
  --negative-validation .local/wakeword/eval-v26-dipco-validation \
  --negative-validation .local/wakeword/eval-v26-libri-dev-other-validation \
  --output .local/wakeword/v26-expanded-validation-calibration.json
python3 - <<'PY'
import json
from pathlib import Path
import subprocess
validation=Path('.local/wakeword/eval-v26-label-smoothing-validation/report.json')
threshold=json.loads(validation.read_text())['threshold']
subprocess.run(['python3','research/wakeword_p4/training/gpu_python.py',
 'research/wakeword_p4/training/compare_streaming.py',
 '--model-dir','.local/wakeword/model-gpu-v26-label-smoothing',
 '--features','.local/wakeword/features-gpu-v6-qwen','--threshold',str(threshold),
 '--upstream','.local/wakeword/microWakeWord',
 '--output','.local/wakeword/v26-streaming-probe.json'],check=True)
PY
