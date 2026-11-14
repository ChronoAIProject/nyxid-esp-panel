# Next milestone: recognition generalization before candidate promotion

The active overall goal compares official WakeNet9 Hey Willow, same-recipe
trained Hey Willow and same-recipe Hey Nyx. Hi/Hey Muse supplements word-length
coverage; Hi ESP remains an engineering reference. Compare matched sources,
environment and false-activation requirements, not equal numeric thresholds.

The next milestone is to explain and reduce the observed cross-source and
near-word errors without concealing them through boundary or threshold changes.

1. Independently check the completed Mac training-witness saved-event result;
   do not duplicate model inference or the already-completed 135-stream run.
2. Census the active feature pack and effective sampler inputs by word, source,
   carrier and label. Distinguish generation intentions, corpus counts and
   surviving sampled windows. Check historical experiments before proposing
   another intervention.
3. Inspect the possible near-word/carrier coverage gap: the inspected generation
   plans contain command-bearing positives but only bare common-negative
   phrases. Reciprocal positives may still supply command-bearing negatives;
   generation plans alone do not prove effective coverage or causality.
4. Freeze one minimal controlled intervention, its original continuation
   control, inputs, crop/RNG/augmentation rules, final checkpoint and evaluation
   before new generation or training. Prioritize raw-FP32 recognition; retain
   the separate INT8 implementation results and their failures.
5. Report every fixed development point with per-source, near-word and carrier
   results and the common FAR requirement. Promote only a qualifying frozen
   candidate to independent acceptance, followed by P4 resource, first-utterance,
   long-call and reconnect checks. Current candidates are not qualified.

Keep fresh development and independent acceptance closed during this diagnostic
phase. Preserve source PCM speed and the single I2S → AEC → AGC → ordered FIFO
path. Do not use Mac speaker playback, recording or new Mac TTS, and do not
operate the dining-room production board. GPU/Mac coordination remains in
[Issue #13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13).

## Current evidence

The saved-event independent check is now reported complete by GPU `460bcecc`;
Mac source/receipt review matches every original NPZ identity. The original
metadata census is complete on both machines, with byte-identical output.
Common-negative command carriers are absent from the actual training metadata;
11 of 13 persistent Nyx float false clips use command carriers, while two bare
false clips remain. GPU has now supplied its first-10,000-batch surviving-window
census and independent coordinate-to-tensor reconstruction. Mac authenticated
the archive and every payload, but has not yet independently reduced the trace.
The reported 227,714 surviving common-negative slots per word are all bare.
This is a coverage observation, not proof of the cause of the recognition gap.

GPU subsequently locked 1,620 Kokoro common-negative command-context requests
and reported starting native CPU synthesis (46d7e75d / d335edac). These are
reported partial results, not a complete admitted pack or new trained candidate.
The derived Kokoro graph has explicit random seeds; its new waveforms do not
share the original bare anchors' random realization. The proposed intervention
therefore changes command context, duration/crop and waveform realization; it
must not be described as a text-only causal experiment.

## Current stage objective

Deliver one interpretable paired recognition result: original raw-FP32
continuation versus the predeclared Kokoro common-negative command-context
substitution, for both Hey Willow and Hey Nyx. Use the same original final
initialization, optimizer, fixed-final training budget and unchanged positive,
reciprocal, own-prefix and ambient inputs. Freeze the complete actual tensor
schedules before training. Preserve the already-running learned-grid job and
its original handles; do not restart or interrupt it for this stage.

Mac first independently checks saved window coordinates and grouped metadata
counts. GPU completes the native pack audit and renderer/features/schedule lock,
then executes the paired recognition continuation according to the existing
sequencing plan. A synthesis timeout is not permission to restart or silently
replace requests. Preserve failures and do not filter by wake scores.

The next result must report all fixed, already-consumed diagnostic points and
separate target recall, bare versus command-carrier near-word errors,
per-source behavior, background FAR and generated-boundary timing. This phase
does not open fresh development or independent acceptance. It tests whether the
specific intervention helps; it does not establish mature-model parity. After
a promising result, separately freeze the next evaluation protocol and work
points before opening the corresponding development or acceptance material.

The overall completion criterion remains matched-source/environment/FAR
comparison against official Hey Willow, then independent acceptance and P4
resource, first-utterance and long-call/reconnect validation. Hi/Hey Muse is a
supplementary word-length reference; Hi ESP is an engineering reference. No
candidate is qualified or promoted at this stage.

Original trained-model P4 numeric/resource fixtures have separately been
prepared and cross-linked: four unchanged models, six complete training streams,
two fresh repeats and 8,336 prepared calls. No P4 execution has occurred; this
does not establish timing, resource headroom, recognition or acceptance.
