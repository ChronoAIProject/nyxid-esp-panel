# Fixed parameter midpoint: continuity gains do not preserve ordinary coverage

The prospectively fixed alpha=.5 midpoint of original Binary40 and its direct frozen-history-teacher continuation has completed construction, final INT8 export and all eight complete consumed replay cohorts. Construction has zero optimizer updates; the checkpoint step20000 describes the completed parents, not a new training run. Exact parent hashes, float32/float64 once-rounded arithmetic, strict saved reload and calibration batch identities passed. Export actual exit0, full/chunk maximum logit error0, Keras probability error9.095e-13/cache3.052e-5, integer-only graph and feedback/reset gates passed. Final INT8 is64800bytes, SHA25698d8ffa7b6ed5b72cb67d3039e8354155f33ac7711488c28fb3b9e364c055700.

All24 sequential plan/replay/score stages completed with actualexit0. The continuation and finalization handles both returned actualexit0. Independent review verified all8archives/1787members, complete comparison source identities and saved event rescoring; reports preserve all1760input traces. Results and publication receipts are pushed1fdf711d/2ce3f795. See 4090-binary40-midpoint-final-summary-v1.json and final-table-v1.md for complete four-model totals and exact gains/losses.

| Ordinary target scope | C161 | Original Binary40 | Prior teacher | Fixed midpoint |
|---|---:|---:|---:|---:|
| Builtin |5/12|8/12|10/12|9/12|
| Continuous8000gap |0/12|7/12|11/12|11/12|
| Continuous32000gap |2/12|8/12|10/12|8/12|
| Legacy commands |26/27|27/27|25/27|26/27|
| Jenny commands |24/27|24/27|21/27|22/27|
| Joe whole-utterance unclassified |0/72|17/72|17/72|16/72|
| MeloUS bare |9/9|8/9|8/9|7/9|
| MeloUS commands |24/27|25/27|24/27|23/27|
| MeloBR bare |7/9|6/9|3/9|3/9|
| MeloBR commands |14/27|14/27|10/27|12/27|

The midpoint retains teacher short-gap hits but fails the intended preservation of ordinary crossvoice coverage: relative to original it loses two Jenny commands, three British bare clips and three US targets overall. Relative to C161 it improves builtin/continuous/Joe while losing Jenny and both Melo target totals. These are ordinary positive losses, not a rare-nearword zero-error requirement. No model promotion is supported; keep C161 as baseline. This rejects this exact fixed derivation, not all parameter interpolation. Do not search other alpha/parents based on these scores.

Daily conflicts remain1/96 versus C1610/96: original index75, "Hey Nick, what time is it?" The common-name conflict is not relabeled as a positive. Legacy10.483944826h, Jenny9.614419583h and Melo7.902862222h reused backgrounds each have zero midpoint events; these are separate reused sets, not newly accrued or additive independent exposure. Joe has no background-hours denominator. The continuous32000gap event at sample312000 falls320samples into the gap after clip2; record its timing without calling it phonetic endpoint truth, first-command retention, or a new unrelated speech false alarm.

Mac host target2/4 is unchanged from both parents and is not P4 execution. No acceptance material was accessed, no alpha/threshold/calibration rescue occurred, and no full midpoint float-attribution replay was performed. Thus these INT8 results alone do not identify whether each individual loss is already present in midpoint float. Separate Mac/P4 numerical, command retention, resources, realtime and stability evidence remains required.

Next work should inventory distinct acoustic/source-supervision approaches against the recorded failures and original literature before selecting another optimizer intervention. Preserve this completed cycle; do not restart it or automatically run another generic retention/context/regularization recipe. The broad useful-model/handoff objective remains active. CPU replay was one-thread BUILTIN_REF; supervisors were bounded to two CPUs, disk remains above100GiB reserve.
