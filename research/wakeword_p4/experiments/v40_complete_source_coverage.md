# v40 preregistration: complete-source coverage in the original BCE batch

Freeze before sampler integration, optimization or candidate scoring. v39's
five-score objective failed all138 old-development points (macro max.3645833,
minimum8 negative false clips). Do not infer that another loss weight will fix
it. Test the separate hypothesis that complete-source phrase coverage restricted
to three British training checkpoint groups is insufficient. This is a data
coverage intervention relative to v37, not an ablation isolating v39's loss.

Use original v37 teacher60/context150 reconstructed initialization, seed726,
frozenBN,120000 steps,eval every5000,original Adam/schedule/weighted BCE and
historical checkpoint rule. No five-score loss, temporal jitter, extra head,
mining, prefix negatives, masking, new audio generation or threshold extension.
Original128 batch remains32positive/48phrase-negative/48stratified ambient,
positiveweight1/negativeweight20. Original base sampler/RNG and ambient draws
remain unchanged. FullSourceBatch makes its original80 draws first, retaining
its RNG stream. Replace exactly16positive slots0..15 and24negative slots32..55
with an independent candidate sampler seeded127726. British complete-source
coverage retains the other16positive/24negative slots. No additive objective.

Candidate data is the unchanged Mac training.npz SHA
bb01f3219db10f31e1095af4912c1807ec38b0b3c640c4676186c1b77a620aa1,
from archive26d58635fa21c022245a1ccdadee44299ce29a02e815bf1701a4c7ef160cb396.
Use saved uint16 x150 only; normalize clip(x/25.6,0,26) exactly as original.
2594canonical source WAVs/header/PCM hashes authenticated locally. All7782 raw
arrays and38910 crops/sample bounds checked. Saved Mac frontend arrays are the
explicit training feature environment:30/30 fixed rendered inputPCM exact on
Linux,16/30 full frontend arrays exact,14 differ (max integer delta133). Preserve
that limitation; do not silently regenerate/replace data or assert host parity.
Sequence_x is not used. Original PCM/container and complete RIR are retained.

Sample candidate positives equally across the five engines, then equally across
canonical conditioning identities, then equally across canonical sources, then
uniformly among the three fixed variants. Negative draws first select equally
Nick/Next/Mix, then equally five engines, conditioning identity, canonical source,
variant. All five engines must contain both classes and all three negative
phrases; fail closed if absent. Conditioning identity is engine plus the sorted
JSON array of original speakers (Piper blends remain blend identities, not
independent persons). Canonical WAV aliases contribute once: choose its existing
canonical metadata, retain every alias in the original pack. No source receives
mass proportional to duplicate alias count. Preserve all positive intended-text
labels; restrict candidate negative phrases to exactly 'hey nick','hey next',
'hey mix' to retain the existing confusion task. Excluded Nexus/HiESP rows remain
in the immutable received pack and get explicit sampler accounting, not relabeling.
Neither source disjointness nor physical-voice independence is claimed.

Before optimization: verify all pinned pack/report/source/record hashes; enumerate
all sampler groups and exclusions, validate50 deterministic batches, exact class
slots and untouched British/ambient slots, no duplicates introduced via aliases.
Verify negative phrase/engine group completeness. Two-step4090 smoke must export
valid original streaming int8[1,3,40] to uint8[1,1], finite weights/frozenBN exact.
Representative quantization data uses this same mixed main-batch distribution;
record exact data and sampler hash. Formal training starts once after prechecks.

After authenticated completed120k output, run unchanged historical regression,
Kokoro/VCTK and oldVox full streams and the fixed138-point gate. No eligible point
means rejection with all reports/failures retained. Only an eligible candidate
may enter separately preregistered unusedSLR83/newVox development alongside frozen
v30 and officialHiESP. No final access until a separately frozen final protocol.
Keep v30 until all original acceptance gates pass. Generated presentation bounds
are process references, not phonetic truth. No Mac350 training or threshold use.
