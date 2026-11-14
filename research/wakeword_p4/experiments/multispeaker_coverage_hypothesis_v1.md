# Prospective speaker coverage contrast

Status: source preparation and hypothesis, not a synthesis/training execution lock.

The completed Nyx engine-adversarial contrast did not improve the recognition
tradeoff. Do not repeat it with another coefficient or checkpoint. Test instead
whether broader *within-model* speaker coverage helps lexical discrimination,
holding TTS models, total material count, phrase distribution, environments and
optimizer budget constant. This differs from v44's three-US-voice mixture, which
changed source mass and selected checkpoints, and from adversarial alignment.

Authenticated prospective training sources are ARU medium (12 declared speakers)
and Arctic medium (18), pinned in
`4090-multispeaker-training-source-review-v1.json`. Both explicitly fine-tune
Lessac, already represented in training. Neither is a proposed independent
holdout. Original protected-family policy excludes neither; metadata-only
catalog entries do not by themselves establish an unused source. Complete
original local/tracked role evidence review remains before material reservation;
any protected use rejects that family without substitution under this proposal.

Proposed two arms: narrow uses three speaker IDs per model, chosen by a fixed
salted SHA ranking before inference; broad uses all 12/18. For each model,
phrase and speed, each arm has 36 native requests: narrow 12 draws/speaker,
broad ARU three and Arctic two. Share the first draws where identities match;
save every returned array and failure. This yields equal 50/50 model mass and
equal request counts; total generated duration can differ and must be reported.
No speaker selection from detector scores. No discriminator or additional head.

Freeze exact phrase/token inputs, speaker schedule, runtime, no-trim native
output quality checks, speeds .9/1/1.1 and quiet/25dB/15dB full-RIR rendering
before synthesis. Reuse existing independent native/render auditors. No
endpoint cropping, ASR filtering, retry-to-pass or sound-quality truth claim.
Lock a same-initialization, fixed-final, matched-budget training contrast only
after authentic material/whole-wave frontend audits; do not inherit target
packs whose provenance fails the prospective quality specification.

Jenny remains reserved development; Joe remains consumed diagnosis. Mike and
all prior finals/paired acceptance stay protected. Existing Joe/Mike protocol
explicitly permits distinct voice/model holds while reporting shared rdh parent
and unknown physical/pretraining overlap; this proposal does not invent a new
requirement to certify all upstream people or pretraining. A future execution
must specify what independence is actually documented, audit exact cross-role
overlap, and cannot call related checkpoints parent-independent. Reservation
alone still does not authorize scoring. Multi-voice independent development
and acceptance design, unchanged positive/negative/FAR gates, official Hi ESP
same-PCM replay and frozen-v30 comparison remain required before promotion.

Before generating, estimate complete native/render/feature/checkpoint/trace
storage and apply the disk guard (1.5 times allocation plus 50GiB reserve).
Authenticated models occupy 153520482 bytes total; reuse them. This proposal
does not authorize a GPU job or consuming untouched development/acceptance.
