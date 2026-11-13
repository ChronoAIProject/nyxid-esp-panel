# Broad context225 prospective scoring contract

This governs the new broad experiment only. Historical pilot scores, failures,
source locks and live official9672-stream task remain immutable. This document
authorizes no detector inference or source generation by itself.

## Whole-input and tail behavior

Process each complete original16k mono PCM once, resetting frontend and detector
only at source start. Preserve PCM speed, amplitude and gaps. Frontend performs
its existing final-only160-sample hop padding. After the whole frontend finishes,
append0/1/2 zero feature frames to complete the last3-frame model input. These
are feature-domain final padding, not fabricated real waveform or phone data.
No real frontend frame is discarded. Number of calls is ceil(real_frames/3).
Retain real-frame count, padding count and full ordered raw UINT8 outputs.

Use fresh reference interpreter twice per complete source and require exact raw
traces. Do not reset within a call or derive chunks from VAD. Complete3-frame
prefix inputs and outputs must match the old implementation exactly; an
incomplete final group causes exactly one additional invoke. Mac's separately
published tail helper and source/constructed tests require identity admission
before integration; do not modify the historical detector to backfill scores.

Raw output step s is one-based, support clock=(3*s+2)*160samples. Preserve raw
step and unclamped support; reported final support=min(clock,original_samples).
The225 offline unused-frame diagnostic does not shift audio or event time.
No waveform resampling, dummy waveform insertion or searched timing offset.

## Models and comparisons

Bind each completed formal120000 snapshot/export by actual bytes/hash and
shape/dtype, not a hardcoded54440-byte size or historical candidate directory.
Require verified120100 actual training/PTQ batches and fixed-final receipt.
Frozen v30 is the original54440-byte model SHA
04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c,
with fixed .91796875/smoothing5 point reported separately. Replay v30 under the
new tail rule in a separately named result; never overwrite its old results.

Main comparison is official Hey Willow, same-recipe trained Hey Willow, and
same-recipe Hey Nyx. Hi ESP retains its own target and explicit denominators.
Models select independent numeric thresholds at common FAR target. Muse is
optional and cannot silently replace either official reference. Old official
curves remain descriptive on their original consumed PCM; independent new
comparison requires the same newly locked PCM through all relevant models.

## Selection and promotion

Inherit exact custom138-point threshold grid, integer5-byte sum/1280 smoothing,
67-step refractory and half-threshold rearm from the existing paired scoring
contract. No epsilon or implicit threshold rounding. Before inference bind
exact grid bytes/code and model/event/metric implementations in execution pins.

Preserve complete-phrase recall at/after generated prefix end and no later than
saved container end; report full-RIR-boundary recall and any-after-source-start
recall separately. Report all pre-source/during-prefix events and signed offsets
to dry-prefix, full-RIR-prefix and command-container ends. These are generated
references, not human final-phone truth or certified listening quality.

Retain gates macro recall>=.8, standalone>=.75, command>=.75, no pre-source
positive events, zero common/cross-negative events, and nominal one-sided95%
Poisson FAR upper<=.5/h over full continuous original background exposure.
FAR upper=.5*chi2.ppf(.95,2*(events+1))/hours. Report clustering limitation.
Macro equal weighting across conditions; any extra voice weighting, source
counts and complete carriers/speeds/conditions must be frozen in the material
manifest before synthesis/scoring, never selected from model response.
Select eligible point by macro, lower FAR upper, then higher threshold; null
if none. Save all points and failed gates, not just selected/max recall.

Current user goal requires unconsumed development for threshold selection.
Explicitly reused old development may be a labeled regression but cannot
satisfy that requirement. Authenticate declared voice/source roles, ancestry
limits, exact encoded/PCM history and background source manifest before opening
new development in any detector. Metadata absence alone is not admission.
Mike/HFCmale/test6/7 and all three consumed historical finals keep their roles;
no acceptance source rescues training or threshold selection.

Development passing alone does not replace v30. Separate locked independent
acceptance, synthetic quality and trained full-waveform INT8 gates are required.
Keep v30 on any failure; report scope only as synthetic-protocol performance.
