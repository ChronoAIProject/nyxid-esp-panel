# Prospective exact positive channel equalization

Use authenticated original7502-step representative channel peaks only.
Per-channel scale=2**(-ceil(log2(max_abs))); zero peaks use1. No factor clipping,
percentile, search or fitting. All factors finite positive float32, stop overflow.
This activation equalization differs from weight-norm CLE; no INT8 guarantee.

Assert original no-residual context225 teacher60 topology and shapes. Let s0
be firstactivation scale; ti the depthwise-i scale; si the following postBN/ReLU
scale, i=1..4. Simultaneously transform original weights:

- First convolution output kernels multiply s0 (original has no bias).
- Depthwise-i kernel multiplies ti/s(i-1), bias multiplies ti.
- Pointwise-i kernel divides input channel by ti.
- BN-i gamma/beta multiply si; moving mean/variance/epsilon unchanged.
- Dense input kernel divides by tile(s4,30), bias unchanged.

Positive ReLU homogeneity cancels these scales at consumers in real arithmetic,
not a floating-point error guarantee. Six state shapes unchanged: input history
unscaled, five following activation histories scaled s0..s4. Fresh zero states
remain zero. Old nonzero state bytes must not be reused in new representation.

Keep bounded[-8,8] head as shared reference. No optimizer, BN re-estimation,
source/representative changes. This estimates equalization versus boundedmodel.

Before export publish immutable factors/transformed-weight receipt; verify
original weights untouched, finiteparameters, sixstates and all270original
complete training witnesses CPU2threads/twicefreshreset. Compare transformed
float versus saved boundedfloat at original1e-4 and actualofflinephase224+zero
at1e-4. Preserve failures; no rescue factor or tolerance change. Add matched15site
originalrepresentative control for exact scale relation, with implementation
tolerance fixed before execution.

Only passingfloat/state control may publishactualexportlock. OneINT8export/arm,
same100x75representatives/zero/full26/no-source-reset/variablequantization.
Then unchanged270wholeinput numericalgates and independentimmutable receipt.
No development/acceptance authorized. Rejection stops without sweep/retraining;
v30 retained.
