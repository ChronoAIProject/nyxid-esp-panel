# Automatic prefix supervision improves command timing but increases near-word errors

The fixed original RTX4090 run23692/PID301084/start1355633 completed naturally,
exit0,20000updates/850.916s. Same original initialization/seed726/freshAdam1e-5
and full causal sequences. Only command-positive temporal references changed:
automatic intended-transcript MFA prefix end + lead + complete RIR tail.
Bare container labels and all positive/negative/ambient pools remained fixed.
No new TTS, waveform crop, checkpoint choice or numerical alignment audit.

Original export46895/PID304903/start1443217 exited0, one builtinINT8 export,
60952bytes, SHA256
`3ea7b067f1000516c8dfe9c6005bccdbe7360a2c26fc3d7790b70e5d26e8d3b7`.
Original actual-primed collector23068/PID305327/start1445904 exited0/all372inputs;
scorer19416 exited0/138points. Same81600continuouszeroPCMprime, all firstphrases,
complete input finishes and10.483944826388889h backgrounds. Previous arms reused
from completed identical-protocol runs, without duplicate baseline inference.

Fixed0.91796875/smoothing5 reference:

| Behavior | Aligned prefix | Bare-only | Whole-command-end | Original v30 |
| --- | ---: | ---: | ---: | ---: |
| Any event after source begins |33/36|28/36|32/36|25/36|
| At/after generated prefix end |28/36|28/36|32/36|25/36|
| Before-generated-end only |5/36|0|0|0|
| Bare any event |9/9|9/9|9/9|8/9|
| Commands |24/27|19/27|23/27|17/27|
| Light |6/9|1/9|5/9|0/9|
| Time |9/9|9/9|9/9|9/9|
| Timer |9/9|9/9|9/9|8/9|
| Common false clips /288 |17|1|9|2|
| Cross-target false /36 |0|0|0|0|
| Background events |0|0|1|0|
| FAR upper95 /h |0.285745|0.285745|0.452489|0.285745|

Aligned-prefix light hits have median132.6875ms after generated prefix,
range102.6875–192.6875ms, versus whole-command-end median1272.6875ms.
All six are within300ms, but three light misses remain. Time/timer medians
153/106.5625ms; all24 detected commands are within300ms. These are descriptive
synthetic-reference/app-return times, not human phonetic latency or a new gate.
The five before-end-only bare events are -32.75 to -2.75ms relative to generated
prefix support. Preserve them explicitly: the generated end is not acoustic
truth, so neither certify complete pronunciation nor automatically relabel
these as false wakes. No source interval or success definition changed.

All138secondary points satisfy the background FAR bound in this consumed
exposure; minimum common errors2 across the grid. Maximum complete31/36,
maximum any-after-source36/36 occur at diagnostic settings. At gridmax0.99609375:
complete28/36, bare8/9, commands20/27, common2, background0. No threshold selected.
Zero consumed confusion is not asserted as mature-model capability or a new
perfect gate. The actual17 reference errors, misses and independent gaps remain.

This supports a practical temporal-supervision effect in this one run, with a
substantial false-wake tradeoff. It does not establish a sole cause, independent
generalization or suitability for replacement. Original v30 remains frozen;
no protected development/acceptance was opened. Independent multi-voice
threshold development/acceptance and officialHiESP same-protocol comparison
are still incomplete. Do not promote or tune this candidate on consumed data.

All original report/138point JSON bytes and372rawcode streams are published
losslessly in762265bytes with SHA receipts. No duplicated audio/features.
See `../results/4090-causal-aligned-prefix-fixed-reference-behavior-v1.json` and
`../results/4090-causal-aligned-prefix-primed-evidence-manifest-v1.json`.
