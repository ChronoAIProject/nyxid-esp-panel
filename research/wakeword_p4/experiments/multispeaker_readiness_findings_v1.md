# Explicit startup readiness is insufficient

The original consumed-only diagnosis session61194/PID242534/start656684
completed with genuineexit0. It reused immutable actual UINT8 outputs, not new
inference or training. The separate saved-behavior reviewer session53504 also
exited0, checking all276 points using the existing independent vector event
oracle, positive/negative counts, original curves, pre-ready codes and exposure.

Readiness was fixed by geometry before scoring: firstvalidcall103, firstallowed
triggercall107/sample51680(3.23s). Smoothing/arming/cooldown start at the first
valid prediction; results are not old event lists with early events deleted.
Every independent original whole stream gets this readiness state once. No
internal background reset or artificial extension of short audio is allowed.

| New readiness behavior | Ready-exposure FARupper95<=.5/h points | Complete positive count at those points | Standalone count at those points | Points also passing common/cross/pre-source requirements |
| --- | ---: | --- | --- | ---: |
| narrow | 30/138 | 9–23/36 | 0/9 | 0 |
| broad | 50/138 | 4–18/36 | 0/9 | 0 |

These are descriptive ranges, not chosen thresholds. The original no-wait curves
remain alongside all new points. Their corresponding FAR-point counts were0/5;
readiness therefore changes system behavior, but does not yield a usable model.
Both readiness variants still have zero joint negative-qualified points and
insufficient complete-positive recognition. Broad remains the only candidate
arm; narrow is not a fallback.

Original background exposure per model is10.483944826388889h. Ready exposure
is10.473178159722222h, with.010766666666667035h (38.76s across12 originals)
inactive. The primary new-system FAR uses ready exposure; minimum upper95 is
.28603850978826895/h. Original-exposure bounds are also retained descriptively,
never used to credit inactive startup silence as detection exposure.

At the already registered maximum threshold .99609375, narrow still triggers
15 common-negative clips and broad7, after readiness. Complete positives are9/36
and4/36, all commands. All9 standalone containers end before readiness and all9
are missed at that point; the records retain these clips and their readiness
availability, rather than deleting the first utterance. Observed hit latencies
at this point are late (median1433ms narrow/1328ms broad relative to generated
prefix reference), with27/32 misses; these hit-conditioned medians are not
latency guarantees or human phonetic timing.

The diagnosis rejects a readiness-only practical fix. It does not retrospectively
pass the old model, identify all errors as startup artifacts or select a wait
length/threshold. No new working point, fresh development, acceptance or P4
handoff follows. Keep frozen v30 and the original failures. Next implement the
distinct causal fullsequence/startup and source-onset coverage hypothesis, using
new whole training presentations and valid temporal label masks. No coefficient,
seed, checkpoint or readiness-duration sweep. No float-equality experiment or
human listening prerequisite. Hardware validation remains separate from the
current simulation research objective.
