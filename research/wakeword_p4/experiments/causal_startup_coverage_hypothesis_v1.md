# Prospective causal startup and onset coverage

This is a new hypothesis, not an execution lock or a claim of identified cause.
The original speaker-coverage candidate remains rejected, and v30 is unchanged.

Actual schedule review (`4090-multispeaker-startup-coverage-review-v1.json`)
finds all5120000 input windows end at or after frame309; synthetic source lead
is always frame310 (49920 PCM samples before frontend framing). Narrow/broad
have9401/9480 full-context source-not-started synthetic negative windows, so
"there were no pre-source negatives" would be false. Positive windows begin no
earlier than frame358. The original training graph processes complete windows,
not partially filled per-layer streaming histories. At the fixed maximum grid
threshold, saved consumed diagnostic events occur at .44s or .74–.86s before
source placement. This motivates a coverage question without proving causation.

Bounded historical distinction: the reviewed phase-aligned experiment shifts
full-window supervision; raw/bounded consistency and QAT focus on score/grid
agreement; zero-state export changes initializers. Their actual failures stay
failures. Do not rerun them with another coefficient, checkpoint or seed. The
review is not an exhaustive claim about every historical source file.

Proposed next single candidate jointly addresses actual startup state semantics
and source-position coverage. Reuse authenticated ARU/Arctic broad30 training-only
native outputs, preserving every original artifact. Prospectively create new
whole waveform presentations with source starts .3/.9/1.8/3.12 seconds and the
same quiet/noise/full-RIR conditions. Never crop the generated phrase or trim
RIR; extract the unchanged frontend freshly from each complete new waveform.
Do not crop a middle slice of existing frontend features and describe it as a
fresh-start waveform: frontend histories/AGC would differ.

Train a causal full-sequence graph with the same deployable parameterization and
six zero-real activation histories at stream start. Vectorized causal layers
must pad each layer's activation history, not simply prepend zeros to raw
features and assume internal states match. Produce one output per3frontend
frames throughout the presentation, including startup. Reuse the original
preserved initialization, never the failed narrow/broad finals. The new method
changes data placement and temporal supervision together; comparison to the
failed baseline cannot isolate which change causes any improvement.

Use only mechanically defensible temporal supervision. Entire non-target and
ambient streams are negative. For positive presentations, source-free leading
frames are negative; frames from source onset through the complete generated
container/full-RIR boundary are unlabelled, never arbitrary prefix negatives.
Complete-container positive outputs use the existing3..27frame completion band;
later retained outputs stay masked unless a separately justified label is
available. Whole-command boundaries are not wake-prefix phoneme boundaries.
Specify presentation/region normalization before execution so sequence lengths
do not silently change class/source mass. Preserve true early detections and
all misses in evaluation; do not suppress startup outputs in deployed event
logic or add a warmup delay to manufacture success.

Before any synthesis/training execution, freeze native references, exact waveform
recipe/placement schedule, source roles, label masks, optimizer/batch/update
budget, shared initialization and one fixed-final export policy in a pushed
lock. Check disk budget and reuse shared immutable storage. Verify basic shapes,
label geometry, finite updates and stream ABI with one minimal disposable GPU
smoke. No full-batch audit or float/runtime score equality prerequisite. No
additional precision-comparison experiment. Then directly evaluate actual INT8
recognition, confusion, complete continuous negatives and generated-boundary
latency on consumed diagnostics; use role-isolated fresh development/acceptance
and frozen-v30/officialHiESP same-PCM behavioral gates only if justified by
actual behavior. Broad30 is a training source pool, not a fresh holdout.
Jenny/Mike/old finals keep their roles. No human listening requirement.

This document does not launch a new GPU run. The next work is to turn this
specific method into source-pinned waveform/label/causal-training implementation,
review any additional matching historical implementations found, and lock one
original experiment. It is not permission to tune failures until they pass.
