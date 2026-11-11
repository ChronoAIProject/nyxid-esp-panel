#!/usr/bin/env bash
# v34: training-only four-class auxiliary phrase loss; deploy binary head only.
# Original export calibration retained to avoid confounding this training run.
set -euo pipefail
model=.local/wakeword/model-gpu-v34-ami-extra
validation=.local/wakeword/eval-v34-ami-extra-validation
python3 research/wakeword_p4/training/gpu_python.py \
  research/wakeword_p4/training/train_stream.py --device cuda \
  --upstream .local/wakeword/microWakeWord \
  --features .local/wakeword/features-gpu-ami-extra-v1 \
  --output "$model" --architecture teacher60 --context-frames 150 \
  --batchnorm-policy frozen \
  --initial-weights .local/wakeword/mww-pretrained-reconstruction-v1/reconstructed.weights.h5 \
  --ambient-sampling stratified --prefix-negatives 16 \
  --phrase-auxiliary-manifest .local/wakeword/training-phrase-crop-support-v1.json \
  --phrase-auxiliary-weight 0.1 \
  --steps 120000 --eval-interval 5000 --source-exponent 0 --event-validation \
  > .local/wakeword/v34-training.log 2>&1
python3 research/wakeword_p4/training/evaluate_stream.py \
  --model "$model/hey_nyx.tflite" --features .local/wakeword/features-gpu-v6-qwen \
  --output "$validation" --validation-only --threshold-step 0.00390625 \
  > .local/wakeword/v34-validation.log 2>&1
for corpus in mandarin dipco libri-dev-other; do
  case "$corpus" in
    mandarin) features=.local/wakeword/aishell-human-features-v1 ;;
    dipco) features=.local/wakeword/dipco-human-features-v1 ;;
    libri-dev-other) features=.local/wakeword/libri-dev-other-features-v1 ;;
  esac
  python3 research/wakeword_p4/training/evaluate_frozen_human.py \
    --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
    --features "$features" --corpus-name "$corpus" \
    --output ".local/wakeword/eval-v34-$corpus-validation"
done
python3 research/wakeword_p4/training/calibrate_multicorpus.py \
  --english-validation "$validation" \
  --negative-validation .local/wakeword/eval-v34-mandarin-validation \
  --negative-validation .local/wakeword/eval-v34-dipco-validation \
  --negative-validation .local/wakeword/eval-v34-libri-dev-other-validation \
  --output .local/wakeword/v34-expanded-validation-calibration.json
python3 - <<'PY'
import json
import subprocess
from pathlib import Path
threshold=json.loads(Path('.local/wakeword/eval-v34-ami-extra-validation/report.json').read_text())['threshold']
subprocess.run(['python3','research/wakeword_p4/training/gpu_python.py',
 'research/wakeword_p4/training/compare_streaming.py',
 '--model-dir','.local/wakeword/model-gpu-v34-ami-extra',
 '--features','.local/wakeword/features-gpu-v6-qwen','--threshold',str(threshold),
 '--upstream','.local/wakeword/microWakeWord',
 '--output','.local/wakeword/v34-streaming-probe.json'],check=True)
PY
