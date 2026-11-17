# Deployment INT8 confirms training history sensitivity

Prelockd4e24ca1 executed once using original Binary40 INT8 SHA256454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c. Actual session90688 exit0. BUILTIN_REF single thread, no optimization. All90 original stratified training source identities/saved feature tensors retained, with four combinations each. All180 physical diagonal suffixes exactly match bytes from complete full replay using the original deployment adapter. Feedback quantization pairs match. This excludes a cache-copy/continuation implementation discrepancy in this diagnostic; artificial off-diagonal state swaps are not deployment reset proposals.

| Training phrase /30 | Silence frontend / silence cache | Silence frontend / speech cache | Speech frontend / silence cache | Speech frontend / speech cache |
|---|---:|---:|---:|---:|
| Bare |30|30|29|25|
| Time command |30|30|30|30|
| Light command |30|30|30|28|

The corresponding float physical speech-history counts are26/30,30/30,30/30. All three additional INT8 crossing losses are in speech/speech: index32 light sum1172; index38 light1137; index45 bare1153, against fixed1175. There are no crossing gains or off-diagonal crossing changes. These scores are descriptive evidence, not grounds to lower threshold, adjust calibration or rescue a checkpoint. Other history losses already exist in float, so quantization alone cannot explain the entire history effect.

Every360 sum/crossing metric was independently recalculated from preserved rawUINT8 arrays and original split/prefix-reference metadata. Source/trace/log/report hashes and actual exit receipt are in `4090-binary40-context-int8-review-v1.json`. Whole source/voice selection,8160sample gap, native-vs-augmented source, automatic endpoint and training-only limitations from the stratified float study remain. These are not event-decoder/first-command retention, background FAR, independent acceptance, human or P4 results.

Historical review found prior same-source rendering consistency, bounded/oneway consistency, QAT and C161 silence/speech context-pair experiments. Therefore neither generic “consistency regularization” nor “add speech history” is a novel proposal by itself. Before any new gradients, compare the exact history pairing, supervision and initialization against those preserved recipes and training-fit failures. A selected intervention must retain quiet standalone support and practical nearword discrimination; no training on consumed evaluation failures or score-picked examples is permitted. C161 remains the baseline, original Binary40 remains an unaccepted candidate, broad goal active.
