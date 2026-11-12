# v44: fixed US synthesis coverage mixture

Hypothesis: v43 fits inherited training confusion words but actual-float fails
on held-out synthesis identities; fixed new en_US coverage can improve this
failure. Not a quantization-only or temporal-only experiment. No development
failure source, voice or score is used to generate/select training rows.

New training-only pack is exactly324 variants, SHA
4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da,
original source manifest350315d41263ef24d593fa46b1291ccc3ef21c95682840d65ccfcf1f1ea7369f.
All108 generated sources and324 augmented attempts have complete independent
mechanical/hash/conversion/frontend/crop audit; all324fit full container.
No acoustic/physical-person/pretraining independence claim. No retries/mining.

Baseline v43 unchanged. Consume all original main/British/complete coverage
and temporal RNG draws first, then replace batch slots0..7 with8 new positive
crops and32..43 with12 new confusing-negative crops. Fixed independent
numpy default_rng seed167726. Per new draw: for negatives choose Nick/Next/Mix
equally; choose lessac/ryan/ljspeech equally; choose multiplier(.9,1,1.1),
stochastic draw(0,1,2), augmentation variant(0,1,2) uniformly through fixed
sorted eligible index choice within preset/phrase. All cells27variants.
No extra temporal jitter/loss/head. Eight of32 positives and12of48 phrase
negatives become new data; remaining original24/36phrase slots and48ambient
slots unchanged. Negative full-container intervention slots coincide with
v43 first12override slots, but their sources and endpoints change by design.
This is a data-distribution intervention, not isolated source-only causality.

Labels, weights1positive/20negative, batch128, original teacher60/context150,
frozenBN/pretrained initialization/optimizer/120k/every5k checkpoint rule,
full source/complete coverage and all original seeds unchanged. Calibration
uses this same new mixture with original representative-batch logic. No change
to thresholds or138old-stage grid, positive-negative rules, macro floor or FAR
upper95<=.5/h. All remaining old-source quality gaps are explicit: exploratory
training does not certify them or permit automatic promotion.

Before formal launch authenticate pack/report/artifacts; independent sampler
replay50batches, exact chosen labels/new slot arrays and untouched slots;
original main/source/temporal RNG equivalence after draws. Verify seed/grid
uniform procedure (not demand equal finite random counts). Two-step4090smoke,
finite weights/frozenBN/streaming int8[1,3,40]→uint8[1,1] contract required.
Then one120k run. Record selected checkpoint, complete original reports and
all failures. Reused old development is rejection evidence only. No new
SLR83/Vox/final access if fixed old-stage gate fails. Even if it passes, original
quality gates and separately untouched development/final acceptance remain
required before replacing v30. No old-stage failure rescues with new data.
