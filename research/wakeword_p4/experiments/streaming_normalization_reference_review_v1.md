# Separate training statistics from deployed temporal causality

Pinned microWakeWord source4665173cd35f1cff9a61e06fc427f124766c488e was fetched
and hashed in4090-streaming-normalization-reference-review-v1.json. README
lines29–33 describe non-streaming spectrogram training followed by streaming
conversion, warn that training-window ambient estimates are imperfect, and note
model-dependent quantization effects. mixednet.py lines338–356 explicitly uses
BatchNormalization after pointwise convolutions and in residual projections.
These are source observations, not executed reproduction or accuracy evidence.

A decision in the recent broadcast-residual adaptation deserves reconsideration:
requiring training-mode zero future-frame gradients excludes ordinary BN training
statistics. That requirement is stronger than causal deployed inference and was
an implementation choice, not a user requirement. BN in training may depend on
other training frames; at inference frozen moving moments are constant affine
transforms. The latter does not read future runtime audio and can often be
folded into a preceding convolution. Training outputs and deployed outputs are
different functions and require explicitly different tests. This does not make
any use of development or heldout audio for normalization legitimate.

The failed broadcast model omitted BN/SSN and changed width/routing/activation;
it is not a negative test of normalized BC-ResNet or microWakeWord. Its zero-update
profile shows partial inactive channels but does not prove normalization is a
remedy. Existing C161 used pretrained initialization/frozen BN; prior BN-affine
adaptation and pretrained Matchbox experiments also exist. Consequently neither
pretraining nor BN alone is a new, untried solution and no success is predicted.

A defensible next normalization study must first specify the exact reference
placement, temporal geometry, train/eval moment semantics, padding and original
training-only moment provenance. Use a matched fresh initialization and fixed
original training sampling; do not continue or rescue the frozen failed model.
Keep all training-only running statistics and optimizer states. Prove deployed
eval full/chunk outputs and caches agree and future perturbations cannot affect
prior outputs. Separately test frozen-normalization folding and export parity,
INT8 cache quantization/reset, actual operators and runtime costs. Training-mode
chunk equality need not be asserted for a method whose declared batch statistics
change with the training chunk; that difference must be disclosed, not hidden.

Before committing a full training recipe, establish these mechanical and resource
facts with zero optimizer updates and record the limitations of the small-batch
statistics. This review does not authorize declaring a candidate valid, selecting
a normalization placement from consumed evaluations, repeating seed/threshold/
checkpoint searches, or changing C161. A future fixed comparison still requires
normal target support, full daily/common-word/connected-command results, original
artifact preservation and Mac/P4 usability evidence. No new training was run here.
