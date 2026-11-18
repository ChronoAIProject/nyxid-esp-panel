# Fixed fresh channel-BN training hypothesis

Evaluate whether ordinary training-time channel normalization changes the poor
fit of the fixed broadcast adaptation. Source rationale b087860b separates
training statistics from deployed causality. This is not a BC-ResNet/SSN/Swish
reproduction or a claim BN fixes the observed failure. Prior affine-only frozen
BN and pretrained routes remain distinct historical failures.

Use CausalBroadcastChannelBN with11BN layers after stem, spectral and temporal
convolutions, before unchanged ReLU, eps1e-5 and momentum.1. Other geometry stays
32channels/20frequencybins/dilations1,2,4,8,10,306frameRF. Trainable parameters8161;
frozen folding returns7457 and the same runtime topology. Preflight738696ef
proves fresh original convolution initialization, two original batch finite
backward, no optimizer updates, deployed full/chunk/cache exactness, no future
dependence, and fold output/cache errors below1e-6. Engineeringfe1b2e80 gives
43000byte untrained builtinINT8,128originaltraining representatives,64Torch/Keras
parity and256feedback/freshreset checks. This is not a trained handoff or P4 proof.

Fresh seed726, no failed final checkpoint reuse. Original128-row Sampler726,
features/masks/positive+20negative region-normalized BCE, Adam1e-3,20000updates,
final checkpoint only. BN moving moments update on each full training batch and
are saved in initial/final checkpoints; use final20000 moments, freeze/fold at
export, no posttraining moment recalibration. Training-mode BN intentionally uses
whole-batch time statistics and is not asserted causal/chunk invariant. Deployed
frozen inference must pass causality/chunk/folding/export tests independently.
No new sources, appended pairs, heldout, threshold/seed/epoch/calibration rescues.

Historical noBN run used identical original sampling and convolution seed but
is not simultaneous. Compare final data support and full8consumedcohort/3context
results, retaining original C161 as practical baseline. Entire ledger must match
historical independently reconstructed original batches by descriptors and
feature/mask hashes; do not unnecessarily read hundreds of GB reconstructing
identical arrays again when full original reconstruction already provides the
hash reference. Recheck terminal artifacts, moving-moment counts20000 and fresh
initialization. Preserve actual exit receipts and originals even on failure.

Use RTX4090, CPU2/interop1, one training job, disk reserve100GiB. Finite training
is not recognition; do not tune from consumed evaluation or automatically promote
from loss/size/rare-nearword rejection. Actual P4 kernels/arena, normal synthetic
acoustic recognition, first-command retention and whole-system stability remain
unproven; Mac continues C161. No candidate delivery until fixed final evidence.
