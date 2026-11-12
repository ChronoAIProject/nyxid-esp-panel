# Proposed matched experiment: absolute temporal tail risk

Status: preregistered proposal before gradients, not permission to skip preflight.
Frozen v30 and every prior rejection remain. No new development/final access.

## Evidence and hypothesis

The phrase-ranking pair failed all original138 points. On the original50 training
batches ranking improved target top1 to500/500 while reducing high absolute target
coverage. The bounded first2 training-batch full-float diagnostic, after the existing
phase1 alignment correction, shows persistent temporal variation: among18 fully
supported targets, ranking any16/last15/mean11 at.99609375, versus INT8 16/15/10.
This is descriptive reused-training evidence, not population inference or a unique
cause. The older v39 arithmetic-probability-mean BCE experiment also failed.

Test an absolute label-directed loss that emphasizes the worse of five outputs,
while retaining same-source phrase discrimination. For each paired training record,
let b_t=softplus((1-2y)*z_t), t=0..4. Define

    temporal = mean_records(weight * (logsumexp_t(b_t) - log(5)))

with original positive/negative weights1/20. No temperature, margin, clipping,
minimum-probability filter, mining, or chosen frame. This smooth risk lies between
mean frame BCE and maximum frame BCE; its gradient weights difficult frames more.
It anchors absolute logits, unlike shift-invariant phrase ranking. It does not
mathematically guarantee a detection threshold or generalization. Unlike v39 this
is not BCE of the arithmetic probability mean and it retains the later matched
four-phrase ranking objective and factorized training-only source pack.

## Fixed intended design

Fresh matched control0/treatment1 for this temporal term; ranking coefficient1 in
both. Main128 sampler, optimizer, seed726 initialization, teacher60 encoder/head
reset, context150, frozenBN,120k updates, original5k checkpoint selection and PTQ
remain unchanged. Last-view paired40 BCE and10-group ranking stay in both arms.
Both arms explicitly calculate temporal backward and perform one Adam update.

Both use a new explicitly identified paired temporal schedule, independent RNG
seed237726, two distinct training conditioning IDs, speed uniform .9/1/1.1, four
phrases and five renderings40 records, five endpoints e-12,e-9,e-6,e-3,e, all e%3=0.
Common e is uniform among legal values max(feature_crop_end)+12 <= e <=
min(350,raw_length-1) across all40. All five windows must satisfy the original
conservative full-container rule. This changes paired endpoint support relative
to the prior experiment in BOTH fresh arms; the old completed arm cannot serve
as this control. Enumerate every possible ID-pair/speed feasibility before launch;
if any group has no legal endpoint, stop and revise the proposal before gradients,
never silently discard/resample sources. No waveform trimming/padding or old
validation failures enter training. Phase convention remains training end0 mod3;
full-stream comparisons use the established one-frame window offset explicitly.

## Required bounded preflight

1. Independent float64 value/analytic-gradient and finite-difference checks,
   equal/extreme logits, temporal permutation, mean/max bounds, absolute-logit
   sensitivity and correct gradient signs for both labels; coefficients never
   estimated from probe values.
2. Full training-only metadata feasibility enumeration, source/raw identity,
   complete-container support and deterministic schedule replay, every grouping
   from metadata rather than positional labels. Stop on any infeasible group.
3. Certified initialization actual-network zero-update gradient probe: finite,
   connected encoder/head gradients, saved logits/arrays, unchanged BN/variables.
4. Fresh matched2-update smoke: actual initialization and draws exact; observe
   all backward terms, one update, finite weights/frozenBN/INT8 deployment shape.
5. Source/execution pins and commands delivered in Issue13 before expensive runs.
   No formal run until these concrete checks pass; no optional-feedback wait.

Then original historical138-point gates unchanged. Both failures reject both;
passing that stage still requires untouched-development selection and independent
reserved synthetic/official Hi ESP protocol, not promotion from old diagnostics.
No coefficient sweep/restart/new-data rescue or relaxed gates.
