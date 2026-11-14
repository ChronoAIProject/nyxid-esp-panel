# Fresh matched one-way smoke v1

Two targets x two conditions x two independently created models/Adam optimizers.
Exactly4 updates each, original225context/128batch/first4SHA/15trainables/frozenBN,
originalequalized initializer and original Adam settings. No old final/smoke
weight continuation, no seed-based replacement of admitted initializer.

Conditions twoway_consistency and oneway_consistency compute and explicitly
observe both gradients; only chosen C gradient differs in S+C explicit merge.
Same scalarC/lambda1/normalization8/Huber1/dualrawBCE1:20. Float-target stop is NOT
fixed teacher: shared parameters continue to receive quantized/S updates.
Nofold/noarithmetic deterministic4090 policy unchanged. Both C scalar values exact
each update; first-step all scalar/gradient diagnostics must match across arms
because both diagnostics are executed on the same initial variables/batch.

Save initial/final H5, numerical variable inventory, complete4batch loss/gradient
trace, finalAdam NPZ and receipt. Require finite values,16BN variable bytes unchanged,
optimizeriteration4, original initial numerical inventory and originalbatchSHA.
Two fresh model/optimizer final numerical identities exact within condition.
Auditor independently checks savedH5/Adam/init/schedule/firststep diagnostics.

This explicitly retains the entire original training sampler including training-only
records previously examined diagnostically: no rows are selected/excluded/reweighted
using diagnostic scores; no consumed evaluation/witness input enters training.
This prospective fresh recipe supersedes any ambiguous phrase forbidding ordinary
training records merely because they were included in training-only diagnostics.
Those diagnostics remain training-role data, never heldout evidence. No historical
failed candidate is rescued; this smoke is disposable and never formalinit.

Publish actual source/dependency/input/protocol hashlock before the single run.
No retries, threshold selection, export, numerical admission, freshdev/acceptance
or candidate promotion. Completion is smoke wiring only, not quality improvement.
Formal training requires a separate prospective budget/execution/validation lock.
