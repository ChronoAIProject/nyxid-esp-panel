# Training-only long-window INT8 stream diagnosis

Precommitted plan 940a0e5cabc5b7f8efce16c8908e60fb2cf8139c398061af04d8a57668943a10 executed successfully: 96 full recordings per frozen model, 192 streams, 43.71 seconds. Every original trace hash verified; lossless recording receipts committed, raw traces retained locally. Same fixed threshold .91796875, smoothing5, continuous81600zero prime,480cadence, full finish; CUDA MFCC and CPU singlethread actualINT8. No PCM duplicated.

| Four training conditioning IDs per engine | Control target /4 | Pooled target /4 |
|---|---:|---:|
|SpeechT5|4|4|
|Piper|3|2|
|Kokoro|2|4|
|Qwen|0|3|
|Total|9/16|13/16|

Control has one Mix activation among80 non-target recordings; pooled0/80. Both have zero prime/leading-ambient events. Hash-ranked conditions/groups/native records were fixed before scoring; this small training-source sample is neither unseen-speaker evidence nor a population estimate. Qwen positive multiplicity is not exhaustively covered. Conditioning is not physical identity. These are clip events, not phonetic correctness or independent acceptance.

The pooled final model does not trigger on three selected training-source positive recordings at its fixed event policy. Therefore remaining failure cannot simply be described as unseen Melo generalization. This does not distinguish low calibrated score, insufficient consecutive high windows, temporal readout placement, training underfit or INT8 distortion. No threshold or checkpoint rescue is authorized by these results. Commands were absent from this training material, so command fitting remains unmeasured.

Next bounded diagnosis should use these already consumed training traces and sampler endpoint positions to separate event-duration/placement from low peak scores before choosing a new training hypothesis. Do not infer a missing phoneme, generate more voices blindly, compare FP32/CPU-GPU recognition as an admission gate, or open sealed acceptance. v30 unchanged; broad objective incomplete.
