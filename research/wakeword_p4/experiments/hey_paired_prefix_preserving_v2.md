# Paired synthetic pilot v2: preserve generated prefix in positive crops

Registered after the input-geometry failure of v1, before any new-pack gradient
updates or synthetic held-out generation/scoring. Original v1 is failed and
retained; this is a separate experiment, not a repaired v1 success.

The two-arm Willow/Nyx question, all voices, exact2400native/7200rendered saved
waveforms, labels, source roles, generated token references,150context,
teacher60/frozenBN, seed726, initialization,20000fixed-final updates, negatives,
optimizer, PTQ, custom event arithmetic and original engineering gates are
inherited unchanged from hey_paired_pilot_protocol_v1.md. No filtering,
resynthesis, waveform repair or new checkpoint selection. Full v1 input material
and its failure remain auditable. No claim that input support predicts recall.

Hypothesis: a geometrically legal positive crop should preserve the complete
allocated wake prefix plus synthetic RIR under every sampled jitter. v1's
fixed3..27 draw violates this on at least two Reza slow Willow RIR rows.

For each training target-positive row, derive
`legal_max = min(27,150-(keyword_frame-lead_frame))` from the same generated
reference. Require `legal_max>=3` on every target-positive row of both arms;
otherwise v2 whole-pack admission fails without filtering or substitutions.
Draw the original `rng.integers(3,28)` once, then use
`effective_jitter=min(original_draw,legal_max)` in the positive endpoint.
Preserve the original draw and effective value in diagnostic metadata.
This deliberately concentrates formerly unsupported draws at the last legal
endpoint; disclose that changed temporal distribution. It does not change RNG
consumption, source order, labels/weights, negative crops or saved waveform.

Only this positive-crop rule changes training. Both arms must use the same
rule on their own positive rows; reciprocal negatives keep original full
negative timeline behavior. New CPU/full20100batch schedules, actual GPU
smokes, source/feature/training locks and candidate freeze are mandatory.
Original trainer and all284 historical pins stay unchanged; implement in
separate v2 adapters/auditor/builder, not an edit to old formal training.

The original token extraction continues to completion. Freeze exact complete
allocation and all-row legal-support results before building v2 packs. Existing
voice/development/acceptance reservations remain unchanged and unopened;
no response to this training-only geometry observation may change held-out
material, threshold grid or gates. Any later failure is retained as null/failure.
