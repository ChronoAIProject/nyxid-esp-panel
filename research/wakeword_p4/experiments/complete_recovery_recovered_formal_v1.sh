#!/usr/bin/env bash
set -euo pipefail
python3 research/wakeword_p4/training/gpu_python.py \
 research/wakeword_p4/training/train_stream.py --device cuda --recover-erased-complete-coverage --paired-objective phrase-ranking --paired-rendering-coefficient 1 --paired-draw-noise-condition fixed \
 --upstream .local/wakeword/microWakeWord \
 --features .local/wakeword/features-gpu-ami-extra-v1 \
 --output .local/wakeword/model-complete-recovery-recovered-formal-v1 \
 --architecture teacher60 --context-frames 150 --batchnorm-policy frozen \
 --initial-weights .local/wakeword/mww-pretrained-reconstruction-v1/reconstructed.weights.h5 \
 --ambient-sampling stratified --prefix-negatives 0 --source-exponent 0 \
 --full-source-pack .local/wakeword/british-training-features-v1 \
 --factorized-coverage-pack .local/wakeword/factorized-training-features-v1 \
 --us-temporal-coverage --us-coverage-pack .local/wakeword/us-training-features-v1 \
 --negative-container-mixture --positive-full-timeline --negative-full-timeline --complete-coverage-pack .local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study \
 --steps 120000 --eval-interval 5000 --event-validation \
 > .local/wakeword/complete-recovery-recovered-formal-v1.log 2>&1
