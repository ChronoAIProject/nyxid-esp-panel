# Actual INT8 recognition is the execution priority

Mac communicated the user's practical-recognition adjustment in Issue13 comments
6884218322 and 6884346145. Read these actual comments on 2026-11-14 before
changing further workflow. This supplements historical records; it does not
rewrite the original training, completed audit, export locks or model bytes.

The full paired batch audit had already genuinely completed before these
comments were read. The original numerical screen and consumed recognition
collector were launched independently; recognition does not wait for numerical
screen completion or numerical score agreement. Let the existing screen finish
and preserve its results. Do not add further floating-point comparison jobs.

For actual INT8 recognition, require the original fixed-final model identity,
completed original execution, correct source/data roles, whole input PCM and
working streaming ABI/initialization, finite quantized outputs and complete
continuous replay. Full-batch reconstruction and offline/stream/TF/FP32 score
agreement are diagnostic evidence, not prerequisites to collecting or evaluating
actual INT8 recognition or preparing fresh development. Do not restore their
prerequisite role in future export workflows; use original completion/exit and
fixed-final identity checks instead.

Use the actual INT8 recognition, confusion, continuous-negative FAR and generated
boundary latency results to decide the next step. Fresh development and independent
acceptance remain distinct, with previously consumed/final material excluded.
Retain the preregistered behavioral quality, positive/negative and FAR upper95
<=0.5/h criteria, threshold-selection policy and frozen-v30/official-HiESP
same-PCM comparisons. No human phonetic or real-user generalization claim and
no promotion from consumed diagnosis alone. Broad remains the sole candidate;
narrow remains a diagnostic control. Keep frozen v30 unless full behavioral
acceptance passes.
