# Multiscale float context target identities

All four BR target losses already occur in float, far below the fixed0.91796875 threshold. Netfloat6→3 hid four lost targets and one gained target, rather than only three losses.

| BR text index | Control float five-mean peak | Multiscale float peak |
|---|---:|---:|
|2|0.958117|0.022489|
|8|0.986161|0.088086|
|9|0.996680|0.359552|
|10|0.964054|0.042732|

Control hits text0/2/6/8/9/10 in both precisions. Treatment float hits0/1/6; INT8 hits0/6. Therefore the additional netINT8miss is the newly gained text1, not recovery of one of the four original lost commands. The observed four severe score collapses cannot be remedied by decoder cooldown adjustments or blamed solely on INT8 rounding.

The full audit covers all144physical-context containers perarm in both precisions, with source text, savedtrace SHA256, original events, five-mean maxima and crossing counts. Code audit_multiscale_float_context_support_v1.py; result4090-multiscale-float-context-support-v1.json. No new inference/training/exposure or threshold sweep. Saved scores cannot isolate learned neural history from frontend interaction; no phonetic causation claim.

Next intervention must address learned continuous-context support without repeating closed context/teacher/consistency/span/calibration recipes or selecting head subsets on these consumed samples. This diagnostic does not justify a new training sweep. Keep C161 and original Mac handoffs, with actual P4 integration/resources/realtime/first-command/stability evidence outstanding.
