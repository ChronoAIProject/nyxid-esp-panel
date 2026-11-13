# Broad-source paired study preparation v1

Status: design, not execution/data lock. No synthesis or training authorized by
this document alone. This is a separate experiment; all failed pilots remain.

Hypothesis: a common broad-source recipe with original training conditioning
coverage, command contexts, stratified ambient, prefix negatives and a longer
optimization budget may generalize better than the two-voice fixed20000 pilot.
These are multiple changed factors; this does not isolate voice count or prove
an architecture/word effect. Official same-word development comparison continues.

Reuse-first source baseline:
- Original e95ee537... LibriTTS_r checkpoint/config namespace, exactly old672
  training conditioning IDs; old112/120 roles remain protected. Existing8064
  mechanically passed training-only attempts contain2016 Hey Nicks standalone
  and6048 common confusions. Reuse exact compatible sources after file admission.
- Original pinned Kokoro model/voice bank, exactly20 existing training voices;
  never old validation/testing voices. Existing same-text sources are reuse
  candidates, not automatic quality admission.
- Original Qwen VoiceDesign revision, original24 training conditions/descriptions;
  no assertion these are24 independent physical voices. Preserve existing ASR
  selection history and failures; do not silently inherit postselection filters.

Proposed positive matrix to turn into an exact data lock: each target
(Hey Willow / Hey Nicks), four carriers (standalone, turn on the light,
what time is it, set a timer), Piper672 IDs ×3 native length scales(.9,1,1.1),
Kokoro20 voices ×3 speeds(.9,1,1.1), Qwen24 conditions ×one original-rate draw.
This is8064/240/96 native positives per arm. Qwen rate variation must use a
separately recorded renderer if adopted; do not invent a native speed knob.
Exact punctuation, seeds, sampling settings, runtime source/config hashes and
per-source reuse versus generation must be fixed before inference. Preserve
full native float and PCM, available tokens/durations and complete command tail.

Both arms use the same frontend, full-tail room/noise renderer and positive
source schedule; own targets positive, reciprocal whole other target negative,
common confusions shared. Broad negative material must be enumerated explicitly,
including original ambient and prefix-negative identities. Do not reuse old Nyx
positives as Willow positives. Silence/partial-prefix negatives need separate
labels and legal crop checks. Existing quality failures cannot be normalized,
trimmed or silently removed to satisfy the new matrix.

Training baseline to serialize, audit and smoke-test before execution:
teacher60/context150, original authenticated initialization/head reset,
frozenBN, negative weight20, stratified ambient,16 prefix-negative slots,
120000 updates/arm, original Adam/frontend/export settings. Candidate checkpoint
rule must be frozen before execution with common training-only evidence; no
post-hoc choice using consumed Prudence. This draft does not choose a checkpoint
or claim exact equivalence to original v30 sampler. Need shared ordered tensor
schedule/identity checks and two-reset stream/export admission before formal run.

Fresh development requires an independently reserved source/voice role and
negative background inventory before generation/inference, while preserving
all prior consumed sources and unopened acceptance/Mike/test6/7. No new dev
identity is selected here. Do not substitute held acceptance for development.
Use frozen five-byte arithmetic, complete-phrase generated reference, full-call
FAR and original joint gates; full curves and failures retained. Compare official
Willow and HiESP at a common FAR target, not equal numeric threshold. Any
replacement of v30 still requires independently locked acceptance after passing
development; current v30 stays frozen. No listening/board gate is introduced.

Remaining concrete lock requirements: exact source byte availability and native
quality retention; full per-cell material/reuse table; common negative/prefix
pack; exact optimizer/checkpoint/sampling/export configuration; fresh development
role admission. This file is deliberately not an executable registration.
