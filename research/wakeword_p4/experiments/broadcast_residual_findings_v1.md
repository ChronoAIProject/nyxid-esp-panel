# Frozen broadcast residual trial: retain C161

The fixed broadcast residual adaptation fails as a C161 replacement. Its smaller
43000-byte export is not a recognition improvement. Keep the original C161 and
fixed threshold235/256, smoothing5; do not flash this candidate for acceptance.

This is a BC-ResNet-inspired adaptation, not a paper reproduction. It jointly
changes width, frequency routing, and normalization relative to Binary40, so the
result cannot isolate frequency locality or invalidate published BC-ResNet.
There is no BN/SSN/Swish in this adaptation. The specific causal mechanism of
failure is not established by these results.

Training used fresh seed726, original128-row Sampler726 and region-normalized
positive+20negative BCE, Adam1e-3, final20000 only. All20000 batches and2560000
descriptors/features/masks match the reconstructed original sampling ledger.
Training, exhaustive ledger review, final export, all17 evaluation stages,
independent1787-member consumed review, training-source audits and float context
review actually exited0. Optimizer arithmetic was not replayed. Successful
execution is distinct from successful recognition.

| Normal target cohort | C161 | Frozen candidate |
|---|---:|---:|
| Legacy primed |35/36|0/36|
| Jenny |33/36|0/36|
| Joe |0/72|0/72|
| Melo US |33/36|0/36|
| Melo BR |21/36|0/36|
| Mac |1/4|0/4|

The initial builtin development target diagnostic has1/12 candidate hits; this
is not evidence of broad usefulness. Daily96 clips have no candidate events.
Reused background streams of10.483944826h,9.614419583h,7.902862222h have zero
candidate and C161 events. These are reused consumed streams, not new exposure;
low false triggers accompanying loss of target response do not justify promotion.
Hey Willow distractor clips are not a deployed Hey Willow recognition benchmark.

All three complete connected-command contexts have0/12 candidate hits versus
C1615/12 builtin,2/12 US,0/12 BR. All ordinary conversation/name/common-word/
household clips in these streams have zero candidate events. Saved native-clock
CUDA float and INT8 decisions agree on all144 clips: all zero. Thus the context
misses already exist in float; this does not prove identical isolated-cohort
float behavior or identify a single training/architecture cause. Independent
float event/4-member archive review passed, with zero optimizer updates.

The31644-row original bare training-source audit also reveals poor float fit.
Original Binary40 to candidate valid positive event rows: Piper5934→2318/6048,
Kokoro180→66/180, Qwen216→76/216. Piper positive-mask crossings5918→2231;
independent source transitions retain2216, gain15, lose3702, neither115. These
are original Binary40 float comparisons, not C161 float comparisons. Piper Mix
negative valid events rise431→845/6048, despite lower Nick5→1,Next53→3,Muse8→2.
Consequently this is not merely an unseen-voice or INT8-conversion issue.

Final original INT8 SHA256:
`c7b51fcec6cc03f7872a7b7fc61d1618d26a79250fb5302a23414b763662c559`.
The gzip preserved at232196f7 was anonymously downloaded/decompressed and verified.
Actual operators: ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC,
MEAN, PAD, RESHAPE, STRIDED_SLICE. Host graph inspection records3312bytes explicit
cache inputs, largest tensor1312bytes, summed tensor bytes31786. These are not
peak arena or P4 kernel evidence. Mac reported missing MEAN/PAD registration;
no kernel, arena, actual recognition or first-command retention qualification
has been performed for this failed candidate. Environment versions and precise
operator counts are in4090-broadcast-residual-final-host-metadata-v1.json.

Authoritative evidence is under research/wakeword_p4/results: the
4090-broadcast-residual training/review/export reports, eight joint comparison
reports and archives, consumed-review, joint-context-original, broad-fit and
bare-transitions-review, and float-context original/review/evidence. Original
reports remain unchanged. Do not rescue this trial with threshold, checkpoint,
seed, calibration, or extra training; any subsequent hypothesis requires its own
justification and protocol. C161 Mac/P4 usability requirements remain open.

## Fixed two-training-batch internal profile

Prelocked dc3d42a9, zero-update CUDA4090 profile of the first two original128-row
Sampler726 batches completed actualexit0. Initial and final checkpoint weights
remain bitwise unchanged after backward. All module gradients are finite.
Final original BCE is0.10412/0.08902 versus initial2.46579/1.98879; mean
positive-mask probabilities rise0.09523/0.10749→0.83882/0.83032, while
negative-mask means fall0.06638/0.05441→0.00119/0.00100. This is not an
all-constant classifier on these training batches, and means are not event hits.

Some ReLU inputs have channels nonpositive over each entire sampled batch:
final spectral.4 has6/32, temporal.3 and temporal.4 each8/32; initial temporal.0
and temporal.4 each9/32. Stem/head retain nonzero gradients; final temporal.1
has no zero gradient entries. These observations do not support a claim that
all branches died during training, and do not prove normalization would fix the
failure. Raw branch RMS and gradient norms are preserved in
4090-broadcast-residual-fit-profile-original-v1.json. The two batches cannot
establish global channel inactivity or causal attribution. No tuning or new
training follows automatically from this profile.
