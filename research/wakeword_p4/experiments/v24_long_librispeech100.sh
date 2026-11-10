#!/usr/bin/env bash
# Reuse already audited v23 features; preserve all prior experiment artifacts.
set -euo pipefail
python3 research/wakeword_p4/training/gpu_python.py \
  research/wakeword_p4/training/train_stream.py --device cuda \
  --upstream .local/wakeword/microWakeWord \
  --features .local/wakeword/features-gpu-v23-librispeech100 \
  --output .local/wakeword/model-gpu-v24-long-librispeech100 \
  --architecture teacher60 --context-frames 150 --batchnorm-policy frozen \
  --initial-weights .local/wakeword/mww-pretrained-reconstruction-v1/reconstructed.weights.h5 \
  --ambient-sampling stratified --prefix-negatives 16 \
  --steps 120000 --eval-interval 5000 --source-exponent 0 --event-validation \
  > .local/wakeword/v24-training.log 2>&1
python3 research/wakeword_p4/training/evaluate_stream.py \
  --model .local/wakeword/model-gpu-v24-long-librispeech100/hey_nyx.tflite \
  --features .local/wakeword/features-gpu-v6-qwen \
  --output .local/wakeword/eval-v24-long-librispeech100-validation \
  --validation-only --threshold-step 0.00390625 \
  > .local/wakeword/v24-validation.log 2>&1
