# Prospective folded-operator QAT feasibility, no training launch

Equalized exports remain rejected. Preserve all failed wholeinputs and original
1e-4/.02/.1/.25 numeric gates; short transition errors are not exempt.

Investigate an explicit TensorFlow fake-quant training graph sharing original
compensated teacher60 variables, with a separate ordinary export graph. This
avoids requiring unsupported Keras3 wrappers in the streaming converter.
No optimizer updates or final coefficient/step budget selected at this stage.

Assert fixed topology: initial Conv5/stride3, four depthwise5/9/13/21 (multiplier1),
pointwise60+frozenBN+ReLU, flatten30x60, Dense1. Fold BN algebra into each
pointwise kernel/bias for quantization: a=gamma/sqrt(var+epsilon),
Wfold=W*a and bfold=beta-mean*a. Keep original BN statistics frozen.

First implement the exact float functional operator graph and compare it against
original compensated offline model and gradients on actual unchanged training
batches from original CPU sampler, not invented labels or development. Pin
CPU2threads and max probability tolerance1e-4 before execution. Retain any
failure; no post-hoc tolerance change or immediate formal training.

Then add explicit per-output-channel symmetric weight fake quantization and
fixed activation fake quantization from the actual equalized INT8 tensor scales
and zeros. Include original input quantizer, depthwise and fused pointwise sites,
flatten and bias scale handling. Zero-channel guards, quantizer zero alignment,
idempotence and boundary rounding need independent numerical tests.

Training supervision remains weighted BCE from pre-sigmoid logits; deployment
bounded head is an output mapping, not permission to zero confident-wrong BCE
gradients. Precisely specify head weight/bias mapping and straight-through
gradients before using any fake-quant loss. Do not call a graph complete QAT
unless every inference-relevant quantizer/state representation is accounted for.

Preflight must record actual input/label/weight hashes, variable identities,
frozen-statistics checks, finite loss/gradients, and independent finite-difference
or analytical STE tests where applicable. Floating finite differences do not
validate a straight-through gradient across discontinuous rounding.
No newdev/acceptance, candidate export or formal optimizer run authorized here.
A later controlled training protocol must freeze initialization, paired control,
data/sampling, steps/checkpoint policy, all quantizer settings and export gates
before any model optimization; do not select them by saving the one Nyx failure.
