# Consumed Joe v30 float/input-rounding/INT8 diagnostic v1

Hypothesis to distinguish: the zero72positive activations at frozen v30's
operating point may persist in the original float model, or be introduced
mainly by quantization. Use all360 already-consumed Joe presentations; no
new audio, regeneration, source selection, threshold grid or model change.
This is distinct from Mac's consumed126diagnostic, which Mac continues.

Use retained original selected70000SavedModel authenticated to dated original
hashes and exact31parameter equality to retained HDF5. Compare three arms:
original float with clip(raw/25.6,0,26); identical float graph fed exact
INT8-input-quantize/dequantize values using original TFLite input scale/zero;
original saved INT8 smoothed traces from the completed fixed-baseline run.
Never rebuild/re-export/requantize or update weights. No training allowed.

Original full source PCM/frontend/padding and same stream steps throughout.
Zero all saved float state variables before each stream/arm; require first
stream reset-repeat equality in each arm. Save all raw float outputs plus
smoothed traces and original INT8 comparisons; finite/range/count checks.
Threshold0.91796875, float64 mean5 and original event-latch/refractory stay
unchanged for all arms, descriptive for float rather than float optimization.
Report complete per-row events, post-source activation, pre-source events,
common/cross-target negatives and score error/peaks. No late-window censoring.

Input-rounding contrast isolates the effect of feature-input rounding within
the float graph. Remaining float-rounded versus INT8 difference also includes
weight/activation/state quantization and runtime/operator numerics; do not call
it solely weight quantization or use additive error as causal decomposition.
Neither a fixed-point activation difference nor small errors prove phonetic
truth, new-corpus generalization, population FAR or hardware parity.

Run CPU-only with2threads, original TF version, after pins are committed and
pushed. Any failed/missing row prevents a complete diagnostic summary; retain
all partial artifacts, no retries under this plan. Sole recovered candidate,
original training/checkpoint policy, development gates and Mike/acceptance
reservation remain unchanged. Findings do not authorize a different candidate
or threshold and are not a prerequisite for current matched formal jobs.
