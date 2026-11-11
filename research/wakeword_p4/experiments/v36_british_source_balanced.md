# v36: checkpoint-source-balanced synthetic training

Hypothesis: v35's Kokoro-only supplement did not cover the British Piper acoustic
source. Add pinned Alba and Cori training-only checkpoints and sample checkpoint
sources equally before voices, retaining the v35 loss, batch, architecture,
initialization, main data, prefix policy, augmentation and checkpoint selection.
This isolates a source-coverage intervention relative to v35, not evidence that
British accent alone is causal. Alba/Cori may share Cori Samuel provenance;
checkpoint-source balancing does not certify independent physical speakers.

424 original sources: original 400 Kokoro plus 12 Alba and 12 Cori. Same three
augmentation conditions; retain full source and RIR tail, exclude non-fitting
sources without truncation. 1002 included crops, 270 exclusions retained. Never
use VCTK development/reserved-final voices or continuous development/test audio
for training. Equal kokoro/alba/cori groups, then equal voices within phrase;
eight positive and eight confusing-negative appended samples. Keep main128 BCE
plus .1 extra16 BCE; negative weight20, independent RNG87726. No new head.

Run two-step RTX4090 smoke and verify finite export, frozen BN and INT8 contract
before formal120k. Historical checkpoint selection every5000 steps unchanged.
Compare with frozen v30 and rejected v35 on the existing declared development
protocol, including background and quantized streaming. Those already-consumed
development data are exploratory regression evidence, not a new independent
acceptance set. Do not claim eligible replacement from those data. A subsequent
frozen, previously unconsumed acceptance protocol is required for promotion;
three consumed final sets remain sealed and 89 reserved VCTK voices/test shards
remain untouched during this experiment. No threshold-grid expansion or relaxed
FAR95<=.5/h gate to rescue a candidate. Any failure leaves v30 frozen.
