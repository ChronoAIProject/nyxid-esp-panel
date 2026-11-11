#!/usr/bin/env bash
# Supplementary validation only; preserves original exports and final data.
set -euo pipefail
candidate=${1:?Specify v30 or v32}
case "$candidate" in v30|v32) ;; *) exit 2 ;; esac
model=.local/wakeword/model-gpu-$candidate-common-calibration-v1
validation=.local/wakeword/eval-$candidate-common-calibration-v1-validation
test -f "$model/report.json"
test -f "$validation/report.json"
for corpus in mandarin dipco libri-dev-other; do
  case "$corpus" in
    mandarin) features=.local/wakeword/aishell-human-features-v1 ;;
    dipco) features=.local/wakeword/dipco-human-features-v1 ;;
    libri-dev-other) features=.local/wakeword/libri-dev-other-features-v1 ;;
  esac
  python3 research/wakeword_p4/training/evaluate_frozen_human.py \
    --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
    --features "$features" --corpus-name "$corpus" \
    --output ".local/wakeword/eval-$candidate-common-calibration-v1-$corpus-validation"
done
python3 research/wakeword_p4/training/calibrate_multicorpus.py \
  --english-validation "$validation" \
  --negative-validation ".local/wakeword/eval-$candidate-common-calibration-v1-mandarin-validation" \
  --negative-validation ".local/wakeword/eval-$candidate-common-calibration-v1-dipco-validation" \
  --negative-validation ".local/wakeword/eval-$candidate-common-calibration-v1-libri-dev-other-validation" \
  --output ".local/wakeword/$candidate-common-calibration-v1-expanded-validation.json"
