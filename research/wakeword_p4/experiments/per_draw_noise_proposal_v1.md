# Hypothesis: fresh training-only noise realizations on each paired draw

Evidence: completed matched-noise arms are rejected. Common noise fits all500
first50 training phrase groups, independent499/500; common nevertheless has
more historical false clips at every same-threshold point. Independent also
fails the reserved SLR83 development gate. Training fit is float-window-only,
not an INT8 quality certificate. Capacity, auxiliary phrase heads, consistency,
temporal-tail losses, shared noise and generic SpecAugment have prior failures.
Do not repeat them as this intervention.

The actual paired renderer currently caches by (voice,speed), with phrase-wise
noise seed SHA256(independent-noise-v1:voice:speed:phrase). Across120k updates,
repeated draws of a source therefore reuse the same noise realization; changing
endpoints changes windows, not the underlying acoustic noise. Hypothesis: fresh
per-draw independent noise could reduce memorization of those acoustic templates.
This is a hypothesis, not a demonstrated explanation of the development error.

Before training, audit all120k existing source draws and count reuse without
loading or scoring any new development/final data. Construct a bounded first50
waveform preflight: fixed-control original independent rendering versus fresh
noise seeded SHA256(per-draw-noise-v1:block:voice:speed:phrase). No shared
phrase noise, source substitution, filtering, mining, new losses or coefficient
changes. Preserve original source PCM, complete RIR tails, buffer/endpoints,
noise SNR25/15 and original main/source schedule. Use one jointly derived
headroom scalar across both arms for each source/rendering group, so clipping
prevention does not introduce unmatched gain. Quiet/RIR-only inputs must be
identical across arms. Validate every full PCM and actual frontend input,
independently reconstruct seeds/noise/gain and verify repeatability/source gates.

Do not launch formal training from this proposal. Only after actual runtime
integration, immutable init/first50 no-update probes, fresh2-update matched smoke,
source/main schedules and execution pins are authenticated may a fresh matched
120k pair be considered. Keep current scalar head/BCE/ranking1/BN/optimizer/
checkpoint/PTQ policy; no reuse of old trained controls as a matched fresh run.
Retain old failures. SLR83/dev2/dev3 are now consumed development, never training
or relabeled untouched. A future independent threshold/acceptance stage requires
new sources/voices and a separately frozen protocol before scores. Existing
three final sets remain excluded. v30 stays frozen; no automatic promotion.
