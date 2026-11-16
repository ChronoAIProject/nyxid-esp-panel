# C161 pitch augmentation: completed fixed-profile experiment

Original RTX4090 training exited0 after20000updates/1814.4847755s. Single final-checkpoint INT8 export:60952bytes, SHA256 `a679e6e4d0314ac77c6b10e564c824bfe1d23e258fa3314244877ab85b62152d`. No threshold/seed/LR/checkpoint rescue. Original C161 retained unchanged.

The controlled intervention replaced broad training-only bare positive and source-matched negative features with deterministic +/-200cent waveform derivatives at50% probability. Independent RNG left original source schedule and supervision unchanged;12 actual batch comparisons verified this. Cache31644rows/~1.107GiB; no derived WAV corpus retained. Source waveforms include existing environment variants; gains/pitch affect those full waves, not only clean speech. This is one specified augmentation policy, not proof that pitch augmentation in general is ineffective.

| Consumed diagnostic | C161 | Pitch candidate |
| --- | ---: | ---: |
| Legacy target |35/36|36/36|
| Legacy command |26/27|27/27|
| Legacy constructed false clips |4/288|4/288|
| Jenny target |33/36|32/36|
| Jenny command |24/27|23/27|
| Jenny constructed false clips |44/324|54/324|
| Joe target |0/72|0/72|
| Mac Daniel/Samantha target |1/4|1/4|
| Mac other targets |0/8|0/8|

Legacy saved paired events gain1target, retain35; false clips retain3/remove1/add1. Jenny retains32target hits, loses1, gains0; false clips retain42/remove2/add12. Candidate Jenny false labels:Next13/Mix29/Muse12. These constructed phrases remain diagnostic and do not measure daily word frequencies. We do not reject a model merely for failing a zero-nearword goal. Here the cross-voice coverage objective has no consistent gain:Joe/Mac unchanged and Jenny loses a hit.

Both original complete background sets have0events/10.483944826h and0events/9.614419583h, matching C161's zero counts. Reused exposure is nonadditive. Cross-target and priming events0. Jenny8hit clips have events before the generated prefix reference; these synthesis coordinates are descriptive, not human phonetic truth or an automatic failure. Report primary any-after-source hits rather than pretending the generated completion field is a measured acoustic label.

Original continuation finished Legacy372/Jenny408/Joe360 complete streams,1140traces, exit0. Original JSON and UINT8 codes have been preserved losslessly with per-member checks; three evidence manifests record156966+186795+158041bytes. Mac12 output is archived separately. No PCM or full feature corpus duplication in git.

Decision:retain C161 as development baseline; do not ask Mac to replace firmware with this candidate. No pitch magnitude/ratio sweep or retraining follows from these consumed cases. The full goal remains incomplete: independent synthetic voice qualification, everyday-conflict evaluation and C161 P4/full-device feedback remain outstanding. Further work needs a distinct evidence-backed coverage mechanism or practical Mac feedback, not continued tweaking of this failed policy.
