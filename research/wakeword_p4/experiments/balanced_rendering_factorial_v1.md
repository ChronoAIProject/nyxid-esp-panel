# Frozen v46 rendering-factor diagnostic

Motivation: all eight balanced-training negative warmed-high clips are Mix variant2;
653 of929 positive container-support misses are variant2. Original variant2 jointly
changes RIR and noise15dB, whereas variant1 uses noise25dB. This cannot isolate factors.
Development residual failures are not explained by this training-only concentration.

Before any new rendering/inference, freeze a diagnostic on ALL8064 original balanced
training sources, not selected failures. Use original serialized PCM and exact existing
frontend/container/gain implementation. Four arms: quiet, noise15dB only, RIR only,
RIR+noise15dB. Quiet and combined must reproduce original variant0/variant2 bytes;
noise-only uses the original variant2 noise RNG seed and noise sequence. RIR-only uses
same RIR [0,160,480,960]/[1,.25,-.12,.06]. Record intermediate arrays, effective gains,
noise bytes, full tails, PCM, features and original source hashes. Combined effective
noise scaling depends on RIR signal RMS; disclose that interaction, do not call additive
waveforms identical between arms. Preserve original files and report every attempt.

Use only original model v46 SHA
5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417,
original balanced source/feature pack. No optimizer, new voices, heldout/final access,
threshold selection or promotion. Fixed .5/.992/.99609375 references; report fulltrace
and warmed-five maxima/events separately, positive complete-container support and
all-source paired arm transitions by phrase/voice/speed. Gains/headroom and RIR tail
are part of intervention, not isolated acoustic truth. No acoustic phonetic claim.

First inspect original implementation and independently verify byte reproduction and
factor construction before scoring any arm. If original quiet/combined bytes fail,
stop and report mismatch; do not relax tolerance or regenerate originals. Source-quality
and inference-contract checks remain required. This diagnostic cannot rescue v46,
which remains rejected; v30 retained. Any future training needs a separate evidence-based
preregistration; no automatic v47 or threshold/exposure rescue.
