# All48 original bare-training clips: cross-family findings

Prelocked execution 29f9c6a8 completed on RTX4090, CUDA FP32, CPU2/interop1, greedy CTC, actual session61465 exit0. All48 original WAVs and all model-source manifest members verified before inference. Original per-clip details and report are copied byte-identically into results; report verifies details SHA256 89d8c871fcc0fb648b2b25311170619869d2cf12b49f74f85b4b5e8d96a5c564. No synthesis, training, label changes or development/acceptance access.

| Original cohort | Clips | Whisper fixed-alias agreement | CTC fixed-alias agreement | Both |
| --- | ---: | ---: | ---: | ---: |
| All |48|23|2|2|
| Provisionally admitted |23|23|2|2|
| Quarantined |25|0|0|0|

This is not a finding that only two clips are pronounced correctly. CTC emits HEYNIX nine times and HAYNIX nine times, alongside five HEYNEX and four AENIX. Greedy orthographic segmentation/spelling differs from the fixed alias vocabulary; simply counting exact agreement mixes these output conventions with acoustic recognition. The original fixed rules and all transcripts remain unchanged. No retrospective whitespace repair or alias expansion is used to manufacture a better number or filter labels.

Consequently this recognizer's exact-alias count is unsuitable as an independent phonetic admission gate here. It does not adjudicate Whisper, prove target/Next pronunciation confusion, or explain the candidate's remaining training and cross-voice misses. The two recognizers can share errors. Do not train only on the two double-agree clips, admit quarantined clips from this review, or repeat transcription with prompts/checkpoints until desired labels appear.

The actionable result is to reject the proposed use of cross-family transcript agreement as a sufficient pronunciation verification mechanism for this short keyword cohort. The existing provisional labels and model bytes remain unchanged; C161 is still the baseline. A next intervention needs evidence beyond these orthographic transcripts, rather than another ASR filter or tail/mask change. Mac/P4 command-retention/resources/stability evidence remains separate and outstanding.
