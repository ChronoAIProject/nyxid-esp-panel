# Complete feature-stream tail policy v1

This opt-in helper is for the next broad225 development scorer. It does not change historical v1 scorers, locks, results or the running official reference job.

The existing `Uint8StreamDetector.predict` processes `floor(N/3)` steps for `N` frontend frames. The historical development collector explicitly asserts that count. The broad225 INT8 probe instead pads the final feature block and processes `ceil(N/3)` steps. The difference is reproducible directly from source: the old loop omits the last one or two real feature frames when `N` is not divisible by three. A new comparison must record which policy it uses; it must not silently combine those output lengths or overwrite old results.

`complete_feature_stream_v1.predict_complete_features` implements the broad probe's complete-tail policy:

1. Keep all original whole-PCM uint16 `[frames,40]` frontend values. The frontend uses only its existing final 160-sample hop padding. Verify `frames == max(0, ceil(original_samples/160)-2)`.
2. Append zero, one or two zero **feature frames** at whole-input end, solely to complete the last stride-three invocation. These quantize to the model's input zero point. No real feature is replaced, trimmed, repeated or reordered; no waveform speed or waveform samples change here.
3. Call the existing detector once for this complete array. The caller resets once at whole-input start. No intermediate reset, re-priming, segmentation or phase dummy is introduced.
4. Require exactly `ceil(frames/3)` raw UINT8 outputs. Preserve one-based raw step indices. Step `s` retains support clock `min((3*s+2)*160, original_samples)`; do not translate clocks for context225's unused diagnostic frame. Events at the final clamped support still count. Background exposure remains the full original PCM duration.
5. Record original samples, original frames, final padding, complete input frames, raw steps and the policy identifier `whole-features-final-zero-pad-stride3-v1` with each result. Threshold arithmetic, smoothing, refractory/rearm behavior and eligibility rules remain separate, explicitly fixed choices.

Use in a new whole-input scorer after its usual PCM/frontend authentication and detector reset:

```python
from complete_feature_stream_v1 import predict_complete_features

detector.reset()
raw_scores, sample_ends, tail = predict_complete_features(detector, raw_features, original_samples)
```

The old custom collector also binds old candidate paths and 54,440-byte models; a broad225 collector needs a new model/execution record rather than editing that historical record. A context change does not change the input stride, one-based raw-step clock or original waveform. The helper does not select a threshold, access media, alter reference configurations or establish recognition parity.

## Constructed verification

`review_complete_feature_stream_v1.py` authenticates and executes the actual `Uint8StreamDetector` class at GPU commit `7d34e9d`, replacing only its interpreter with a stateful input observer. Eighteen frame counts cover 0–8 and remainders around 150, 225 and 600. Every real quantized frame appears in order exactly once; padded frames equal the input zero point. Complete groups and their synthetic output prefix match the historical path. Changing the last real frame in an incomplete group changes only the added final synthetic output. Fresh-reset repeats are exact, source features are unchanged, support clocks match the original clock and five malformed/truncated-input cases are rejected.

This is batching/clock evidence with a fake interpreter, not numerical evidence for a learned model, an actual frontend traversal, P4, recognition, FAR or independent acceptance. The trained model still needs its actual full-input numeric and development checks.

Results: `research/wakeword_p4/results/mac-complete-feature-stream-review-v1.json`.

```sh
python3 research/wakeword_p4/experiments/review_complete_feature_stream_v1.py \
  --gpu-ref 7d34e9d \
  --output NEW_PATH.json
```
