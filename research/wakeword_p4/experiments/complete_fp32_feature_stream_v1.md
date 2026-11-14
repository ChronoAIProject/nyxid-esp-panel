# Complete FP32 feature traversal

`complete_fp32_feature_stream_v1.py` is a separate typed helper for a future
complete FP32 development collector. The existing UINT8 helper, collector,
selector and their validation remain unchanged. This is not a complete new
collector/selector, candidate admission or production integration.

For one whole original source, it authenticates exact supplied model bytes,
requires the complete uint16 frontend-frame count, pads only the final partial
stride3 group, and normalizes with float32 `/25.6` clipped to `[0,26]`. One
fresh BUILTIN_REF interpreter traverses every frame in order. The helper
requires float32 `[1,3,40]` input and `[1,1]` output, no quantized tensor
parameters, and finite float32 probabilities in `[0,1]`. It returns every
probability, original INT64 clamped clock and explicit tail/input digest. It
does not reset in the middle of the source, quantize probabilities or invoke
an event/threshold policy.

The caller must still bind actual runtime, source/material/model and scoring
identities under a published protocol. A fresh calibration driver must use
the original binary64 five-score FP32 event policy, explicit target metadata,
complete original denominators, independent new-development roles and the
existing selection/FAR requirements. This helper alone grants none of that
admission and must not be substituted into a UINT8 report under its old schema.

The verification plan is frozen in
`../results/mac-complete-fp32-helper-numeric-plan-v1.json` before actual model
execution. It selects the original first variant-zero Piper/Kokoro/Qwen
feature streams for each Willow/Nyx target, as in the earlier P4 witness
subset, and only the already consumed original Nyx raw float-control model.
It authenticates the original packet manifest, every used NPZ/model, the
source bytes, LiteRT installed files and package versions. Six complete streams
run twice with fresh interpreters. Their normalized input hashes, full clocks,
repeat bytes and outputs are compared with the original saved TF/LiteRT
references at the unchanged 1e-4 numeric bound.

```sh
PYTHON research/wakeword_p4/experiments/review_complete_fp32_feature_stream_v1.py \
  freeze --packet ORIGINAL_PACKET --plan NEW_PLAN.json
# Publish the exact plan/source before run.
PYTHON research/wakeword_p4/experiments/review_complete_fp32_feature_stream_v1.py \
  run --packet ORIGINAL_PACKET --plan PUBLISHED_PLAN.json \
  --published-ref ACTUAL_PLAN_COMMIT --output NEW_RESULT.json
```

Constructed checks exercise short, full and partial-stride lengths, fresh
state, final zero padding and malformed model/input/tensor/probability
rejection. They are separate from the real saved-reference numeric witness.
No new weights, fresh development/acceptance, source audio, hardware or
recognition/FAR evaluation is part of this helper verification.
