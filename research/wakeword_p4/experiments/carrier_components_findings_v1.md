# Carrier component findings

Neither component arm supports replacing C161. Both were fresh fixed-seed Binary40 trainings; original128 rows were preserved and only four existing Nicks command positives or four Nick command negatives appended. Existing control128 and joint136 are reference cells; supervision and compute differ from component132. No heldout tuning, threshold/checkpoint rescue or new TTS.

All20000 batches independently reconstructed per arm (2640000 descriptor rows each); training, exports, all34 evaluation stages and independent consumed audit exited0. Audit verified3574 archive members and6 complete context traces. Artifacts: aee62e50.

| Target cohort | Original Binary40 | Positive only | Negative only |
|---|---:|---:|---:|
| Builtin | 8/12 | 7/12 | 7/12 |
| Legacy | 36/36 | 35/36 | 36/36 |
| Jenny | 33/36 | 32/36 | 33/36 |
| Joe | 17/72 | 17/72 | 10/72 |
| US | 33/36 | 36/36 | 34/36 |
| BR | 20/36 | 19/36 | 19/36 |
| Mac | 2/4 | 2/4 | 2/4 |

| Complete context | Control | Positive only | Negative only |
|---|---:|---:|---:|
| Builtin | 8/12 | 7/12 | 9/12 |
| US | 11/12 | 9/12 | 11/12 |
| BR | 6/12 | 1/12 | 1/12 |

Both components independently reproduce severe BR approximate “Hey Nicks” context regression. Therefore the earlier joint regression cannot be assigned solely to Nick negative supervision. Positive-only also loses US tolerance; negative-only retains the US total but introduces an ordinary-name conflict: “Hey Max, can you carry the other bag upstairs?”. Counts alone do not establish identical hit sets; complete per-clip control transitions are preserved in 4090-carrier-components-context-transitions-v1.json. These are INT8 findings; component-specific float causality has not been tested.

Daily name_address hits: control1/24, positive0/24, negative1/24; all other daily groups0. Positive-only all ordinary full-context groups0; negative-only has the one US Max hit, other ordinary full-context groups0. Rare constructed nearwords remain diagnostic, not a zero-FP gate.

Reused historical backgrounds remain separate: Legacy10.483944826h, Jenny9.614419583h, Melo7.902862222h. Both components have0 events on each; these are previously consumed backgrounds, not added exposure or a fresh FAR estimate.

Keep C161 and the existing Mac handoff. No candidate promotion. The result rules out treating Nicks command-positive addition alone as a reliable tolerance remedy, and rules out blaming only Nick negatives for joint BR regression. Any follow-up must distinguish learned float behavior from component-specific quantization before proposing calibration/QAT; prior joint float evidence already showed pre-quantization regression. Synthetic host replay does not prove first-command usability, human generalization or P4 readiness.

Text identity correction: the original6444 bare positive rows are all “Hey Nicks.”; the appended540 Kokoro positives are180 each of “Hey Nicks, turn on the light.”, “Hey Nicks, what time is it?”, “Hey Nicks, set a timer.”. Thus “canonical” previously described the fixed training command pool, not a different literal pronunciation. A canonical-versus-approximate spelling mechanism is unsupported. Metadata cannot establish realized acoustics. The bare pool spans716 conditioning IDs across3 engines; appended commands cover20 Kokoro IDs only. This difference is confounded with context and masks, so it is not a proven cause. See4090-carrier-components-positive-text-census-v1.json.
