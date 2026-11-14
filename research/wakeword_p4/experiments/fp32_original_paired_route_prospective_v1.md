# Original paired models: prospective FP32 route v1

Preparation only: no training, export, model invocation, audio/features, scoring,
threshold selection or development access is authorized by this document.
Mac request: Issue13 comment6873929681. P4 resource feasibility is pending.
Current synthetic INT8 objective and all historical rejections remain unchanged.

## Fixed inclusion and identity

Include ALL FOUR final-repeat0 H5 from original paired-QAT formal result:
Willow/Nyx × float_control/integer_qat, final10000 only. Float controls are the
primary same-recipe two-word comparison; both integer-trained arms are retained
as explicit diagnostic controls, never fallback rescue models. No new bounded
or one-way candidate, checkpoint, pretrained initializer or best-of-results
selection is substituted. Manifest binds exact original H5 bytes and source
receipts. Earlier consumed recognition evidence cannot become fresh evidence.

## Output and state contract

Reconstruct original mixednet context225 architecture from pinned source and
load H5 without transforming any parameter. Original Dense(1,sigmoid) returns
sigmoid(raw Dense logit). DO NOT install bounded_head_relu6, clip logits,
equalize channels, fold/re-estimate BN or insert QAT/integer surrogate operators.
BN is inference-only with original gamma/beta/moving mean/variance unchanged;
all H5 arrays must be numerically identical before/after reconstruction/export.
Training variable.trainable flags do not redefine inference semantics.

FP32 external-state stream has input[1,3,40], six explicitly supplied all-real-zero
initial states, float32 probability output and float32 state outputs. Preserve
original state shapes/order and stride3. Every full input starts fresh; carry
all six states in order within input. No startup priming/removal, speed changes,
frame shifts, truncated contexts or state resets within a waveform. Frontend,
final padding to multiple3 and unclamped sample clocks are the original runner.
Offline nominal225 and historical phase224+zero are DIFFERENT references:
phase comparison uses precisely the established streaming phase at index74;
never shift actual stream to make nominal align. First74 outputs stay in every
TF/LiteRT/MicroInterpreter full-stream comparison and recognition ledger.

## Required later execution lock before export/inference

This proposal does not bind actual environment, exporter or test media yet.
A separate immutable execution must bind actual exporter/runner source, TF,
Keras, LiteRT and TFLM versions, original frontend, model/BN inventories,
all six state identities, complete original135 requests per target and their
source-role/hash chain. Run all four models × their original135 requests (540)
without score-based filtering. No new development/acceptance audio at this stage.
Export FP32 only with no optimization/quantization/representative calibration;
reject hidden INT8 kernels or output-head changes. Record full op/tensor metadata.

Prospective FP32 numerical gates: all arrays finite, all full probability outputs
and each full carried-state tensor TF-vs-LiteRT and TF-vs-real-MicroInterpreter
max_abs<=1e-4; two fresh interpreter resets repeat byte-exact per backend.
Original offline-vs-stream phase probability max_abs<=1e-4 after index74.
No tolerance fitted from future outcomes; no cross-backend byte equality claim.
The 1e-4 probability gate matches existing float phase tolerance; applying it to
all state tensors is a conservative new FP32 engineering requirement, not an
inherited proven guarantee. Failure remains failure; preserve all originals.
Independent saved-array auditor must authenticate every request, input/output/
state hash, repeats, clock, phase and per-tensor maximum. Actual host TFLM
is required; LiteRT alone does not establish MicroInterpreter compatibility.

## Separate deployment and recognition decisions

Mac owns P4 app-only resources/timing; host arenas/linking are not P4 evidence.
P4 must retain cold calls, validate output<=1e-4 and every invoke<=30ms hop,
and separately establish whole-panel incremental resource/audio-path headroom.
No board is prerequisite for continuing the current synthetic INT8 research.
Only after numerical and runtime feasibility may a NEW registered FP32
recognition protocol name fresh unconsumed development sources and selection
rules. It must compare same-PCM official Willow/HiESP and frozenv30, freeze
workpoints, then use independent synthetic acceptance and FAR upper95<=.5/h.
No current H5 is promoted by this protocol, historical results or FP32 feasibility.
