# Frozen-final training-support gradient diagnostic v1

Hypothesis: rare final float/exact-FC discrepancies on original training support
may retain output loss seeds yet have ineffective parameter gradients because
of encoder/FC STE masks, cancellation between float and quantized consistency
paths, or opposition to supervised BCE. No one mechanism is assumed true.

Inputs are exactly the existing authenticated10 whole batches per target from
4090-bounded-training-support-forward-plan-v1 (1,1001,...9001), original labels
and1/20weights. All4 original final10000H5 models, original compile grids and
frozen BN; original nofold/noarithmetic deterministic4090 execution. Both original
nominal225 and the already-defined shift-left224+zero contexts. Keep every batch,
role and example; no selection by discrepancy. No new PCM or heldout material.

Before running, publish source and an actual hash lock covering source/dependencies,
versions, input NPZ, prior plan/result/audit, H5/TFLite and compile templates.
One run only; errors stop with original evidence retained. No updates, optimizer,
checkpoint selection, export, runtime activation invoke or threshold selection.

For each case/mode/batch, compute the original dual raw-BCE supervised S and
bounded-logit Huber C at lambda1. Use persistent autodiff to observe full15variable
S and C gradients, plus float-only and exactFC-only C paths via output seeds
and vector-Jacobian products. Record None gradients explicitly (do not silently
substitute before reporting). Save actual seeds, scalar-FC accumulators, logits,
FC outputs, losses and every variable-gradient array with stable variable inventory.
Compute exact scalar-FC declared surrogate activity from original configuration.
Report full norms, dot/cosine, branch sum residuals and supervised/C merge norms;
zero norms have null cosine. Residuals descriptive, not relaxed/reinterpreted old
additivity gates. No parameter update is allowed even if loss/gradient is finite.

Repeat each full diagnostic twice on the same frozen model and require saved
outputs/gradients exact, plus model/BN byte inventory unchanged. Forward values
must match the prior saved nominal/phase outputs exactly; if not, stop and retain
failure instead of changing execution policy. Independent noTF auditor recomputes
saved seed formulas, loss reductions, branch sums/norms/cosines and all file
hashes. Autodiff correctness and live unchanged inventory remain producer evidence,
not inferred from mathematical saved-array checks.

This is a static training-support diagnostic, not wholewave numerical admission,
recognition/FAR evidence, a causal intervention, or permission to train on scored
diagnostic records. Existing bounded rejection/freshdev closure/v30 freeze remain.
Do not change lambda, surrogate, sampler or labels without a separate evidence-led
prospective experiment. Mac original native replay and official QEMU stay separate.
