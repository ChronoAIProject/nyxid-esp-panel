#!/usr/bin/env bash
# After both controlled exports and dense validation complete; no final scoring.
set -euo pipefail
for policy in with-extrema representative; do
  model=".local/wakeword/v27-calibration-ablation-v1/$policy"
  validation=".local/wakeword/eval-v27-calibration-$policy-validation"
  for corpus in mandarin dipco libri-dev-other; do
    case "$corpus" in
      mandarin) features=.local/wakeword/aishell-human-features-v1 ;;
      dipco) features=.local/wakeword/dipco-human-features-v1 ;;
      libri-dev-other) features=.local/wakeword/libri-dev-other-features-v1 ;;
    esac
    python3 research/wakeword_p4/training/evaluate_frozen_human.py \
      --model "$model/hey_nyx.tflite" --frozen-report "$validation/report.json" \
      --features "$features" --corpus-name "$corpus" \
      --output ".local/wakeword/eval-v27-calibration-$policy-$corpus-validation"
  done
  python3 research/wakeword_p4/training/calibrate_multicorpus.py \
    --english-validation "$validation" \
    --negative-validation ".local/wakeword/eval-v27-calibration-$policy-mandarin-validation" \
    --negative-validation ".local/wakeword/eval-v27-calibration-$policy-dipco-validation" \
    --negative-validation ".local/wakeword/eval-v27-calibration-$policy-libri-dev-other-validation" \
    --output "$model/expanded-calibration.json"
done
