# Prospective learned activation-grid feasibility (no formal training authorization)

Problem: original fixed-grid/STE/equalized-source paired QAT and consistency
experiments failed full-stream numerical gates; pooled/flat matched exports also
failed. Four fixed complete internal witnesses now reconstruct every temporal
block and show new integer-output residual at the first depthwise, later propagated
through high-gain pointwise layers. This does not prove a particular grid is causal.

New hypothesis: fixed activation grids constrain adaptation; jointly learned
positive per-tensor activation steps, exported without a second calibration, could
reduce complete-stream discretization error. This differs from changing output
thresholds, repeating PTQ calibration, or rerunning existing fixed-grid consistency.
It is a proposal, not an established remedy. Must first inspect all historical
implementations for equivalent scale learning before authorizing any formal run.

Deployment stays signed INT8 activations/weights, int32 bias and uint8 probability,
original temporal state/clock and ABI. No float fallback, int16 substitution,
startup removal, extra context, human data or new development/acceptance access.
The frozen v30 is unchanged. Existing matched models remain rejected.

Feasibility requirements before a training protocol:

1. Choose one existing matched Nyx flat final model as a mechanical witness only;
   preserve its H5/export, initialize all steps to original actual tensor grids.
   No optimizer updates or candidate selection in feasibility.
2. Define learned log-step parameterization and explicit positive finite bounds,
   saturation rule, rounding and step-gradient estimator in source. Independently
   check analytic/finite-difference behavior away from discontinuities and document
   surrogate disagreement at rounding boundaries. No claims of exact derivatives
   through discrete rounding or anti-collapse guarantees.
3. Include every shared state/input/concat constraint: a state tensor and its
   producer/consumer cannot silently acquire incompatible quantization metadata.
   Share steps where operator ABI requires equality; keep real-zero initialization.
4. Export actual final steps/zero points, quantized weights, bias scales and integer
   requantization constants directly. Never let a converter silently recalibrate
   the final network. Check all per-channel weight/bias and operator constraints;
   static acceptance is insufficient for arithmetic acceptance.
5. At unchanged initial steps require byte-exact original integer outputs on all
   four fixed full startup captures. For controlled constructed step changes,
   require training-graph forward vs actual exported builtin-reference and existing
   integer oracle parity, two fresh repeats, finite values, state reset, padding
   and phase. Any mismatch is preserved as a failed feasibility result; do not
   launch training or relax tolerances to rescue the proposal.
6. If feasible, register a separate matched control/treatment protocol before any
   updates: same original initialization/batches/optimizer/budget; control fixed
   grids, treatment learned steps; include all learned parameters and Adam state
   in repeat/audit accounting. Fix loss and scale bounds before training. Full540
   numerical qualification remains mandatory before recognition admission.

This document freezes scope, not numerical hyperparameters or an execution lock.
No model calls, exports, training or scale choices have been performed for it.
