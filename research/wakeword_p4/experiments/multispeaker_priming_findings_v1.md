# Existing zero-PCM priming does not qualify either final model

Read Mac revisions6884555587/6884681206 before this run. Model bytes, all360
consumed synthetic clips and12whole continuous recordings are unchanged.
Sources from70ad0ea307d366163b0253d06335b623c288e411 are saved alongside the
collector. Context309 requests81440samples;170full480blocks feed81600zeros.
One frontend/model lifetime continues through every complete original recording.
The508priming feature frames produce169calls and one pending frame, retained
across the real-input boundary. No cached-feature splice, intermediate finish,
reset, readiness suppression or added wait occurs. Only the complete original
stream end uses the existing complete-input hop/realzero-feature finish.

Raw UINT8 and event state include all priming. App notifications ignore only
prime-stage events; those events still affect arming and67call cooldown. Full
real exposure10.483944826388889hours/model excludes priming, retains every first
phrase and has no internal background resets. Primary event times are480block
returns clamped at complete-input finish. Unclamped feature support clocks and
finish flags are also retained. No threshold is selected from these consumed
inputs, and no fresh development or acceptance is opened.

Original collection91071/PID261618/start821770 exited0,744traces complete.
Scorer62414/PID263495/start834663 exited0. Saved-trace review checks all744hashes,
code/probability identities, original sample counts and clock arrays;169call
prime prefixes are identical across all372inputs per model. Independent reviewer
8440 exited0: cumulative-integer-sum/searchsorted event reconstruction agrees on
all276points, priming/app events, exposure, positive hits, confusion and signed
latencies. These are behavior/integrity checks, no float-model fidelity gate.

| Fixed-grid result | narrow | broad |
| --- | ---: | ---: |
| FAR upper95 <=0.5/h points /138 |30|50|
| Joint common/cross/FAR-qualified points |0|0|
| Complete positives /36 across FAR points |11–29|4–22|
| Standalone positives /9 across FAR points |1–6|0–4|
| Minimum common false clips /288 across all points |15|7|
| Gridmax .99609375 complete /36 |11|4|
| Gridmax standalone /9 |1|0|
| Gridmax common false clips /288 |15|7|
| Gridmax background events |0|0|
| Gridmax FAR upper95 /h |.2857447576425148|.2857447576425148|
| Gridmax pre-source positive events |0|0|

Ranges span all FAR-qualified grid points; no combined operating point or
selected threshold is implied. Gridmax latency medians1436.25/1338ms are
hit-conditioned with25/32misses and cannot demonstrate a latency pass. Generated
prefix/container/RIR references are synthetic boundaries, not human phonetic
truth. This consumed diagnostic is not multi-voice independent acceptance.

Existing priming improves some early/standalone recognition relative to the
readiness-only policy, which had no standalone hits at any FAR point. It still
fails confusion and recognition. Frontend, model and event histories all change
relative to cold start, so this does not isolate a causal mechanism. Original
cold and readiness failures remain unchanged. Neither model qualifies or
replaces frozenv30. No wait/threshold/calibration/checkpoint rescue is attempted.

Original reports, all276detailed point records and all744complete UINT8 sequences
including prime are published losslessly using gzip/NPZ plus SHA manifests
(total about1.63MiB). No waveform/native feature binary is duplicated. Free disk
remains618GiB;100GiBreserve continues. With the existing-runtime diagnosis now
failed, resume the previously declared causal fullsequence/onset coverage
hypothesis: disposable two-step INT8 ABI integration, then one fixed20000update
RTX4090 candidate initialized from preserved original weights, not smoke weights.
No extra float comparison, retrospective qualification or fresh-role reuse.
