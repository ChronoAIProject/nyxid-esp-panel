# Train the wake word, rather than completion of the whole request

The causal candidate improves consumed recall, but its current training target
does not ask for immediate wake-word recognition in positive command examples.
This is a task-definition problem, independent of numerical matching.

At GPU commit745ee1e2, `build_causal_onset_material_v1.py` computes
`end = lead + source.native_samples + full RIR tail`. The native source includes
the whole text, including "what time is it" or "turn on the light".
`causal_sequence_supervision_v1.py` places the positive region30–270ms after
that whole-container end; the wake word and ensuing command are unsupervised.
The native generator explicitly records no phoneme durations. Consequently
this label cannot be described as a wake-word completion label.

The existing actual-primed replay makes this relevant to use: all5 detected
"Hey Nicks, turn on the light" variants fire1.23–1.33s after the generated
wake-prefix support, near the end of the complete request. Four others miss.
All6 false "Hey Muse, what time is it" triggers likewise occur near the end of
the request,1.29–1.59s after its prefix support. The time/timer positive
examples can fire much earlier. Thus32/36 overall detections alone hides a
connected-command latency weakness. The saved events and source observations
are in `../results/mac-causal-training-label-scope-v1.json`.

These generated boundaries are useful source coordinates, not measured phoneme
times. The evidence identifies a concrete target mismatch and actual late
events; it does not prove this is the sole cause of confusion or latency.

The minimal next practical experiment is to use the already available bare
"Hey Nicks." and "Hey Nicks!" positive sources for completion supervision,
keeping the existing negative/ambient material and causal runtime. This avoids
guessing a wake boundary inside an unaligned command and needs no new audio.
Keep the current failed model and its command examples for comparison; do not
crop audio or turn complete requests into keyword-boundary truth. Measure
early wake behavior, misses and false wakes directly after the next fixed run.
This is a different change from adding common-negative command carriers.

If positive commands are subsequently used for temporal supervision, supply
the wake-prefix boundary separately from the whole-container end. An actual
alignment or stored synthesis timing can supply it; word-count proportions or
the complete-request end cannot. Do not require exact CPU/GPU, gradient or
FP32/INT8 equality before testing the resulting usable model.
