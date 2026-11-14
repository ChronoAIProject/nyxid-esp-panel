# Consumed-only explicit startup readiness diagnosis

Mac Issue13 comment6884525325 proposes a deployment behavior hypothesis:
triggering before model history exists. Test it on saved UINT8 output only before
new training/rendering. This is distinct from changing weights or proving a
causal explanation for all existing errors. The proposed causal training work
is deferred until this cheaper actual-system behavior question is answered.
The previous no-wait model/system remains rejected and all its curves remain.

Geometry is fixed without looking at scores: context309/stride3 gives103 model
calls until a complete context. The first valid prediction is call103 (raw index102).
Initialize smoothing/arming/cooldown at that prediction, accumulate five valid
predictions (calls103–107), and permit detection at call107. With frontend support
clock (3*call+2)*160, earliest readiness is sample51680 /3.23s. Report the geometry
and actual sample count per stream. A shorter stream never becomes ready. Do not
prime histories with fictitious audio or extend input to manufacture readiness.

`wake_readiness_policy_v1.py` defines the new explicit streaming state machine;
the bulk diagnosis restarts the original event arithmetic on raw[102:] and maps
indices back to the full timeline. It does not filter old event lists. Original
half-threshold rearming/67call cooldown and integer five-score means stay intact.
Each original independent whole stream starts once; no internal reset, splitting
of backgrounds or altered model output. A later physical restart would restart
this readiness state, an operational behavior that Mac must eventually implement
and verify if a viable new system candidate emerges.

Lock all existing744 traces, source reports, original276point curve/audit,
policy/scorer/version/geometry before recomputing all138 original thresholds/model.
Preserve all pre-ready UINT8 outputs, original pre-ready event counts at every
threshold, both new ready and old no-wait summaries, complete signed latency,
first-phrase misses/late detections and every original36positive clip. No dropping
short/early utterances. No chosen threshold, wait-duration sweep or new model/audio
inference. Source placement and generated container end are synthetic references,
not phoneme truth.

For negative exposure, report full original hours, inactive/not-ready hours and
active-ready hours. Count all events including final padding as before. The
primary new-system FAR upper95 uses active-ready original audio exposure only;
also report the same count over original hours descriptively. Do not pool old
exposure or credit inactive startup silence to reduce the bound. Preserve all
near-word/cross-target/pre-source events and all276point results.

This is a newly declared readiness behavior on consumed diagnostics, not a
retrospective pass for the old models. No fresh development/acceptance is opened
by the diagnosis itself, and no threshold is selected from it. V30 remains
frozen. If behavior is still poor, use that evidence to continue the distinct
causal-startup/onset training hypothesis. If it is promising, it needs separately
locked role-isolated fresh development/acceptance and actual runtime/restart
integration; consumed success alone is insufficient. Do not add float-equality
experiments or human listening requirements.
