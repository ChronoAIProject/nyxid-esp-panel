# Fixed causal fullsequence/onset training method

One prospective candidate, not a new parameter sweep or isolated causal test.
Reuse the original broad30 ARU/Arctic native training sources; regenerate only
.3/.9/1.8second source-placement presentations, and reference original3.12second
whole-wave/frontend records unchanged. Each of4320 native sources has all4leads
and3original quiet/25dB/15dB-full-RIR conditions:51840 presentations,10368positive
and41472non-target. Nativebytes/text/speaker/speed/draw are unchanged. Source0
through its complete end and full RIR remain, with1second retained tail. Native
TTS provenance/quality reports remain binding, not human phonetic certification.

Freshly reset MicroFrontend once per new whole waveform. Do not cut/concatenate
old frontend features to simulate a new startup. New rows get one shareduint16
binary and offsets; original3.12rows reference their authenticated originalbinary.
Store original/serializedPCM hashes, frontend provenance, complete waveform and
label geometry. Preserve all failures; no retry-to-pass or alternate native draw.

The single fixed formal run is20000updates, seed726, original preserved context309
initialization, all15trainables and16frozen BN variables. Adam1e-5/.9/.999/epsilon1e-7,
FP32 RTX4090 deterministic, TF32disabled, no JIT, masks/mining/head replacement,
weight decay, gradient clipping, LR schedule or early stop. Fixedfinal20000 only.
No fallback model or checkpoint. The material job does not itself start training;
actual model/data/runtime pins and a minimal two-update GPU smoke precede the
separate formal execution lock. No full-batch reconstruction or float score
agreement is a prerequisite to actual INT8 recognition.

Batch128 presentations:32positive,48syntheticnegative,48original training ambient.
For each synthetic class choose the two engines uniformly, then uniform native
rows/lead/condition within engine. Native speaker counts differ intentionally;
within each engine the full declared12/18speaker pool is equally represented.
Same canonical source ordering/seed726, no detector-driven sampling. Ambient uses
original stratified training row probabilities and up to600consecutive feature
frames per episode, with a uniform legal start. These ambient excerpts are
feature-level training negatives with frontend history inherited from the source,
not falsely described as fresh-cold-start waveform extractions. No protected
validation/testing/final arrays enter this pool.

Each sequence starts with zero-real activation histories at every causal layer.
First conv history2, then depthwise histories4/8/12/20, classifier history57 of
60channel activations. Share unchanged deployed model variables; causal firstconv
stride3, pointwise conv/frozenBN/ReLU, rolling Dense equivalent58x60 kernel. No
raw-feature-only left padding substituted for deeper-layer startup. Right-pad
batch features only to the fixed multiple3 maximum from material geometry;
exclude outputs beyond each real sequence's final3frame-aligned call from loss.
Causality ensures padded future input cannot influence supervised earlier outputs.

Sample support at call j is(3*j+2)*160. For positive presentations, support<=source
start is negative; source-onset through complete generated+full-RIR container and
until+480samples is masked. Complete-container positive support is inclusive
[end+480,end+4320], the existing3..27frontend-frame band. Remaining outputs are
masked. Do not label arbitrary command prefixes or later retained wake context
negative. Entire non-target and ambient episodes are negative, including startup.
Source/container boundaries are mechanically known, not wake-prefix phonemes.

For every presentation normalize positive and negative regions separately to sum1
when present. Loss is mean_over_128_presentations(sum(pos_weight*BCE1)+
20*sum(negative_weight*BCE0)). Thus adding different lead lengths does not silently
increase a class/source's loss mass. Positive presentations intentionally add a
source-free negative region alongside the positive region; this is the declared
new temporal objective, not an unreported weight change. Record masks/lengths
and batch hashes/compact source descriptors. Retain initial/final state and
optimizer arrays without storing all full batch tensors.

Export fixedfinal once using unchanged rawsigmoid/BN and sixzero streamingstates,
builtin INT8 representative material fixed from training before scoring; no
conversion/calibration retry or numerical rescue. Verify basic ABI/initialization,
finite outputs/reset and complete actual waveforms, then directly measure actual
INT8 recognition, confusion, continuous-negative FAR and generated-boundary latency.
No startup-output suppression, changed smoothing/refractory or tuned readiness
wait is bundled with this new model. Priorreadiness-only failure stays a separate
system diagnosis. No extra float equality experiment or human listening gate.
Freshdevelopment selects threshold only under a separately frozen role-isolated
protocol; independentacceptance/FARupper95<=.5/h and frozenv30/officialHiESP samePCM
behavior remain required for promotion. All oldfinals/consumed diagnostics keep
roles, v30 unchanged until success, and findings remain synthetic-only.
