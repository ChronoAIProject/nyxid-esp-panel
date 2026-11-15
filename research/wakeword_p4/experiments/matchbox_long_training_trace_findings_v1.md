# Frozen training trace diagnosis: heterogeneous failures

Existing 192 recording traces and both original record-log hashes were verified. No new inference or data. The full pooled5000batch gzip was parsed with exact contiguous steps; original material index mapping verified. All192 real-source smooth crossings match recorded activations: no refractory/rearm suppression accounts for these misses.

At threshold .91796875 the fiveUINT8 sum must reach1175. Pooled misses:

| Engine / conditioning | Exact record training draws | Peak raw code | Peak five-code sum | Peak sum offset from full source+RIR end |
|---|---:|---:|---:|---:|
|Piper428|2|215|1046|9711 samples|
|Piper628|3|194|880|2325 samples|
|Qwen condition_011|92|242|1170|4800 samples|

Piper misses never reach the raw threshold anywhere after source onset; they are not merely a five-step smoothing failure. Their low exact-record exposure motivates checking sampler support distribution, but does not prove extra updates cure them. Qwen has one raw-above-threshold call, insufficient averaged output; despite92draws, its five-code sum stays below1175. Its best peak is later than the training source-end offset envelope256..1024. Within that envelope, Qwen peakraw232 and peaksum1160. This suggests a representation/readout position or sustained-score support question, not evidence of a unique cause. Source/RIR container endpoints are synthetic references, not phonetic endpoints. Exact trained offsets need separate evaluation; cadence envelope is not equivalent.

No threshold lowered or alternative smoothing tested. No FP32 recognition comparison, quantization-distortion conclusion or independent qualification. Original frozen outputs preserved. Blindly extending all-engine training or adding voices cannot be justified as the single fix from these heterogeneous failures. Next protocol should first audit source exposure distribution and complete-phrase endpoint coverage across all labels, then isolate a source-balanced coverage change from a temporally supported endpoint objective; inherited old-encoder temporal failures must be reviewed before proposing any new arm. Commands remain absent from this material. P4 cost remains unresolved. v30 unchanged and broad objective incomplete.
