# v39: direct five-score support, independent of v38 temporal jitter

Preregister before any v39 optimization. v38 failed all138 fixed grid points,
minimum15 confusing-negative clips. Same-checkpoint float diagnostics and actual
shared-sequence mapping support testing the decision statistic directly, but do
not establish a unique cause or expected generalization benefit.

Baseline is v37, not v38: unchanged128 single-window BCE examples, full-source
phrase replacement, same original48 stratified ambient slots/weights and RNG,
teacher60/context150/frozenBN, original reconstruction/seed/Adam/schedule120k,
validation every5k and same historical checkpoint selection. No temporal jitter,
auxiliary classifier, prefix negatives, mining or threshold extension.

Add independent16-example training-only sequence batch:8positive,8 confusing
negative; equal kokoro/alba/cori checkpoint groups, equal voice within group,
equal Nick/Next/Mix negative phrase, uniform eligible base examples, seed117726.
Use original reconstructed full raw training features only. Each162-frame segment
begins at original crop start and extends12frames after its end; shared sequence
graph yields five scores at starts0,3,6,9,12, each using149effective frames.
Require entire generated signal/fullRIR support in all five windows; use original
v37 conservative150-frame support rule for both offset0 and12. Unsupported bases
remain present in main v37 BCE, excluded only from sequence supplement. Preserve
all exclusions; no padding/trimming/relabeling/score-based inclusion. No new audio.

Loss = mean(original128 weighted BCE) + 1.0*mean(sequence16 weighted BCE of
five-score arithmetic probability mean), positiveweight1/negativeweight20.
No max pooling or stopped gradients; backprop through all five probabilities.
Compute this exact probability-mean BCE in log space with
logsumexp(logsigmoid), not clipping aggregated probabilities. Mac e32f0d6
flagged saturated-probability zero-gradient risk before v39 optimization;
verify logits+/-20 and direct overlapping-window gradients before training.
Use original model optimizer once per step over shared original variables only;
BN frozen. This changes objective mass and is not directly comparable to weak
v35/v36 supplements. Export a separate original window copy, never mutate the
shared training graph through upstream conversion. Inference graph unchanged.
Representative INT8 data keeps baseline v37 training distribution (same windows);
sequence-derived weights receive original production export, no new state policy.

Before120k: audit input hashes/source/labels/full-support exclusions, exact five
outputs, variable sharing, direct-five-window gradients versus sequence gradient,
finite updates/BN unchanged and two-step4090 streaming INT8 export. If group is
missing, stop and report rather than reweighting to another source. Then run
original historical/Kokoro/VCTK/oldVox138-point gate; reject if no eligible point.
Only eligible old-stage candidate may consume preregistered unusedSLR83/newVox;
finals remain separate/frozen. Retain v30 until all original acceptance gates pass.
Synthetic generated boundaries are not final-phone truth. No Mac350 training.
