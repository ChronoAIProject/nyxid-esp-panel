# v38 temporal-support hypothesis (training-only preregistration)

v37 fixed complete-source windows fit offline but do not ensure sustained stream
support. Full-history phase1 diagnostic improves agreement but still leaves40/258
training positives without five-score support at the historical threshold. This
does not prove endpoint causality or phonetic correctness.

Before v38 training, derive offsets0..14 feature frames (0..140ms) after each
original crop end from the authenticated reconstructed full raw training streams.
Keep150 frames and require the entire original generated signal plus full RIR
tail to fit within the same effective support convention as v37. Do not trim,
relabel, invent features or use score-based inclusion. Preserve all rejected
offsets. Baseline offset0 must exactly reproduce all1002 v37 crops. All270 prior
unsupported source variants remain excluded. Preserve source/checkpoint/voice,
phrase and base-variant balance: select v37 base example exactly as before, then
uniformly choose its valid temporal offsets. Long sources with fewer offsets
must not lose sampling mass. Positive and confusing-negative slots both jitter.

Use a separate temporal RNG seed107726 so v37 source/base choices and original
ambient RNG are unchanged. Original batch128 (32positive/48phrase-negative/48ambient),
negativeweight20, teacher60/context150/frozenBN, reconstructed initialization,
optimizer,120k/eval5k and historical checkpoint selection remain unchanged.
INT8 representative generator follows this same new training distribution.
No supplementary BCE/head, prefix negatives, hard mining, model-state hacks,
threshold tuning or development-based temporal range choice.

Authenticate full-history array hash and baseline pack; audit all offsets and
sample determinism/ambient preservation, then two-step smoke and export contract
before120k. Run the existing historical+Kokoro/VCTK+oldVox fixed138-point gate.
If no eligible point reject and preserve untouched SLR83/newVox/finals. Only an
eligible candidate proceeds to the existing preregistered unused development
stage. No automatic promotion; frozen v30 remains intact. This tests temporal
coverage of existing sources, not new voice diversity or phonetic truth.
