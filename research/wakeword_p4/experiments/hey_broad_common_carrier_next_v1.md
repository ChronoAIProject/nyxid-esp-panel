# Next recognition hypothesis after original window census

This is an implementation design, not a synthesis/execution lock. No new model,
source generation or fresh split scoring has been launched under this document.

The actual first10000 original batches per word retain320000 positive windows,
320000 ordinary negatives,160000 own-prefix negatives and480000 backgrounds.
Common negatives account for227714 surviving ordinary windows, all bare.
Reciprocal negatives account for92286, including49311 command-carrier windows.
Thus the gap is common-confusion command contexts, not all negative commands.
Counts concern requested synthetic text and actual feature windows, not heard
pronunciation or independent voices. Independent coordinate reconstruction must
pass before using the census as a completed premise.

Do not repeat the failed two-voice pilot under a new name: that pilot explicitly
included four common-negative carriers and its custom development rejected both
words (maximum complete recall0/36 Willow,1/36 Nyx). Its source coverage, model
initialization/training and quantized inference differed from the broad raw-FP32
models. A controlled broad-source continuation may test the missing cells,
without attributing that historical failure to carrier absence.

The proposed minimal intervention is *only* matched Kokoro common-negative
command context, in the existing broad raw-FP32 continuation. Use existing20
training voices and original three speeds; retain their original bare negative
containers as pairing anchors. Do not alter positive, reciprocal, own-prefix,
background or Piper/Qwen ordinary-negative draws. Original source selection,
crop RNG, permutation and original baseline batches stay available and hashed.
Use a separate prelocked RNG to select half of surviving Kokoro common-negative
slots, then one of the three original light/time/timer suffixes uniformly. The
control uses exactly the original slot and window. This isolates a narrow
intervention, not a claim to solve all engines or replace balanced negatives.

Before any generation, fix exact native requests/paired anchors/model+voice bank
hashes/versions/phones/tokens/durations/quality rules and a whole-pack rejection
policy. Retain all original and new failures, native float and complete PCM.
Derive replacement crop geometry using the same original RNG uniform quantile
rather than extra baseline draws; specify coordinate mapping and effective
support before locking it. Never use wake scores to filter synthesized material.
Positive/reciprocal/background tensors and their original draw sequence must
remain exact between arms. The negative context and duration/crop distribution
change together; do not claim a pure lexical-only causal contrast.

Use the authenticated original broad raw-FP32 final10000 initialization for both
arms, fresh identical optimizer, fixed final10000 continuation, no selection or
restarts. Lock exact training source/parameter identities and full ordered
per-arm tensor schedules before training. Both words are paired controls;
primary question is raw-FP32 complete/confusion behavior on the already-consumed
full diagnostic protocol, with the full138-point curve and original FAR rules.
Do not tune a new threshold on reused diagnostics or bypass real-waveform
numerical gates. No new recognition formal is queued while the original
learned-grid session77025 remains live; retain its independent final audit and
full540 continuation. Fresh development/acceptance remain closed until an
actually numerically qualified candidate warrants the original prospective flow.
