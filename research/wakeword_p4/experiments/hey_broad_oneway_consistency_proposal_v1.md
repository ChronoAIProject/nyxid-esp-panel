# One-way bounded consistency: prospective controlled proposal

Evidence is the full saved final-snapshot gradient diagnostic8d2c98a3, not a
selected failing example. F/Q consistency VJPs oppose in the bounded-family's
21/21 batches with both nonzero (Willow nominal2/phase1, Nyx nominal9/phase9).
Opposition is partly expected for a difference loss; it does not establish that
cancellation caused failures. Nyx supervised/Q opposition remains, so a stopped
float target need not help. Larger remaining gradients are not a quality metric.

Hypothesis: stop only the float side of bounded C, preserving the original
supervised S on both raw logits, may improve quantized alignment without allowing
the consistency term itself to move the float target. The float graph shares
weights with the quant graph, so it still moves after parameter updates; this is
NOT a fixed pretrained teacher, EMA teacher, independent teacher or distillation
quality guarantee. No new architecture/head/data/threshold change.

Helper hey_broad_oneway_consistency_loss_v1 delegates unchanged S/C forward to
original bounded helper, passing stop_gradient(float_bounded) only. Lambda stays1,
normalization8/Huber1, labels/weights and quant branch unchanged. Control is original
two-way C. No hyperparameter search or loss chosen by development outcome.

First prerequisite is an actual no-update gradient wiring verification at ALL
existing fixed final-model batches and both context modes: forward losses and S
gradients unchanged, one-way C gradient equals original saved Q VJP within the
original diagnostic's explicit saved residual scope, full outputs/repeats/model
inventory saved. Publish source+actual hashes before execution. If numerical
comparison needs an acceptance tolerance, predeclare its basis before run; do not
retrofit a passing tolerance. No optimizer and no training on selected tail cases.
This checks intervention semantics, not training efficacy.

Only after wiring verification, define a separate matched fresh initializer
smoke with2 fresh replicas/arm, original supervised recipe and prospective fixed
schedule. Both arms calculate/observe the same diagnostics before explicit
component merge and one optimizer step. No reuse of final/smoke trained weights.
A formal budget/source/execution lock requires a new protocol, complete source
quality/ancestry guards and validation plan before any formal training. Existing
failed controls cannot be relabeled successful or used to rescue this treatment.
A new entire-population training protocol must explicitly handle previously scored
training-only records; no diagnostic tail mining, evaluation input reuse or
training-only diagnostics falsely described as heldout evidence.

All original numerical .0001/.02/.1/.25 wholewave gates stay. Complete training,
compile/constants, whole540screen/recursive and independent audit precede any
fresh development. One-way treatment must independently qualify; two-way results
cannot rescue it. No newdev/acceptance/threshold/model-promotion authorization
comes from this proposal. V30 immutable. Mac originalnative and official QEMU are
not duplicated or restarted.
