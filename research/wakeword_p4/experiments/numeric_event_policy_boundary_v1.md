# UINT8 event-policy boundary diagnosis

The frozen original GPU evaluator uses float64 `numpy.convolve` with five
weights of `1/5`. The current opt-in P4 research runtime sums five dyadic
UINT8/256 scores in double, then divides by five. Its generated threshold
also has an `f` suffix, rounding the threshold to float32. These arithmetic
paths can disagree at an attainable mean or rearming boundary.

One explicit original-v30 counterexample is a score-byte stream of ten zeros,
`[253,253,227,231,211]`, then ten zeros, at threshold `.91796875`. On the
recorded Mac NumPy2.5.4 build, the legacy mean is `.9179687499999999`, below
the threshold. The C++ runtime's mean is exactly `.91796875`. The original
host evaluator has no event; the compiled current C++ core triggers at raw
step15. This is an analytic byte stream, not output from a speech/model run.

The audit constructs28874 analytic windows around the unchanged138-point
grid and half-threshold boundaries. It finds385 differing predicates, then
constructs a full stream for each. Compiling the unchanged event core
extracted from the existing C++ source confirms385 event disagreements:
207 single-trigger and178 rearming-after-refractory streams. It retains all
windows, full streams and both compiled event lists. It compiles the core
on Mac; it does not run an ESP32-P4 or inference engine.

As a proposed arithmetic clarification, taking the integer sum of five
UINT8 scores, converting to binary64 and dividing by1280, and preserving the
configured threshold as binary64, agrees with the compiled double-threshold
core on all385 streams. No epsilon, threshold adjustment or window change is
used. This is a proposal, not an applied change to the current frozen protocol.

The original frozen v30 point was separately replayed on all528 consumed
saved streams (Kristin/Norman126, Joe360, background42). Canonical versus
legacy arithmetic changes **zero event streams** there. Only the original
`.91796875` was compared; no new threshold selection or inference occurred.
This numeric boundary issue does not explain the dominant near-zero float
scores or rehabilitate failed models. Known original results remain valid
within their documented scope.

## Reproduction

Public evidence asset: `numeric-event-policy-boundary-v1.tar.gz` under the
existing research release. It contains the exact audit source, original GPU
evaluator, original C++ source, compiled harness source, complete analytic
cases,528-stream regression event lists and report. It contains no compiler
binary, speech, microphone data, firmware, account or device state.

Extract and run without a git checkout, with NumPy and a C++17 compiler:

```sh
python audit.py --runtime-source original-runtime.cpp \
  --evaluator-source original-gpu-evaluate.py --output FRESH-ANALYTIC-AUDIT
```

NumPy/BLAS/compiler identity matters for the legacy convolution's last bit;
cross-machine disagreement counts are observations, not universal constants.
Use the shipped full cases to inspect the recorded Mac run. The script
authenticates both original source hashes before compiling the event core.

For the complete consumed-v30 regression, also pass `--baseline` pointing to
the original126 baseline and `--gpu-evidence-root` pointing to the three
already authenticated Mac receipt directories described in
`joe_background_saved_score_audit_v1.md`. This uses existing public score
archives; original background waveform bytes are not required or claimed.

The first execution completed numeric checks but failed when serializing a
NumPy boolean. That output directory is retained privately. The corrected
script casts those flags to Python bool; later full runs retain all results.
No failing model or speech input was discarded or replaced.

Before any future candidate scoring, coordinate a versioned arithmetic
contract through Issue13. Keep the original formal training, historical
postprocessing, reports and pins unchanged. A future change needs explicit
host/runtime agreement and P4 verification; this diagnostic is not candidate
acceptance, recognition improvement or Hi ESP parity.
