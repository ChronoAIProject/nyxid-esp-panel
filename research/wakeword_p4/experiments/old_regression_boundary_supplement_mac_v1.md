# Prospective additive old-regression boundary descriptions

This reader supplements the original rejected-model regression. It does not
modify its collector, reducer, audit, reports, thresholds or numerical gates.
It runs only after the original complete collection, metrics and independent
audit are published at an immutable GPU commit. The parent v2 plan is frozen
at `f0449e3d45cb8b68f26bf9e7a3a54857ae978e87`.

Use all original 1242 model/path/threshold points and their 360 old synthetic
ledger segments. The original raw-score audit remains responsible for proving
event correctness. This reader verifies that audit's identity and directly
reads only original event steps and source metadata; no model, score vector,
audio, new development or independent acceptance is opened. It recomputes no
events, searches no thresholds, and emits no preferred operating point.

For the original 36 positives per target, report any event, any event at/after
source lead, events only before source, and cases with after-source events but
no hit at/after the dry or RIR prefix reference. For each of the four original
prefix/container dry/RIR endpoints, describe both the first after-source event's
signed offset and the first event at/after that endpoint. Include count,
minimum, median, p95 and maximum; empty cells retain zero count and null stats.
The latter two prefix summaries must reproduce the original reducer exactly.
This exposes early-only versus absent detections without changing either's
original outcome. Original ledgers retain all repeated events for further
inspection; this supplement explicitly summarizes first events only.

Use the original one-based clock `min((3*step+2)*160, samples)`, inclusive
boundary equality and complete final-clock clamp. Source boundaries remain
generated references: negative offsets alone prove neither a timing-reference
error nor a successful human wakeup. Background/FAR conclusions remain with
the original complete report and audit, not this synthetic timing supplement.

Before any actual ledger read, run the constructed checks and publish the
reader, this protocol, the constructed report and the generated plan. After
the original GPU reports complete, fetch that actual result commit and run:

```sh
python3 research/wakeword_p4/experiments/describe_old_regression_boundaries_mac_v1.py run \
  --published-ref ACTUAL_MAC_PLAN_COMMIT \
  --gpu-results-ref ACTUAL_COMPLETE_GPU_RESULT_COMMIT \
  --report NEW_OUTPUT_PATH.json
```

The published source hashes and parent plan must match. All original published
reports, the source manifest and every event ledger hash must match. Writes
require a new report path. No original report or file is overwritten.
