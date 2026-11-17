# Paired capacity continuation: complete consumed replay

Neither continuation is an overall replacement. Retain C161 as baseline and original Binary40 as a research candidate. Wide128 improves Melo coverage, but loses short-gap continuous and Joe coverage relative to original Binary40. Lower everyday/background events do not justify ignoring these losses. This experiment does not prove capacity is useless: it tests one fixed widening/initialization/continuation recipe.

Training started from original Binary40 final weights, with matched original sampler/masks, fresh Adam, and 20,000 additional updates per arm. Control64 and wide128 data order was authenticated earlier. All training/export/replay stages finished; the continuation receipt contains 53 successful stages and both successful training subprocess receipts. Independent review verified all 16 archives and 3,574 member size/hash pairs, and source identities against original Binary40 comparisons. Raw-code event rescoring remains in the cohort scoring reports. No new model inference was required for this review.

| Consumed synthetic endpoint | C161 | Original Binary40 | Continued64 | Wide128 |
|---|---:|---:|---:|---:|
| Builtin /12 |5|8|6|5|
| Continuous 0.5s /12 |0|7|2|2|
| Continuous 2s /12 |2|8|7|6|
| Daily negative events /96 |0|1|0|0|
| Legacy target /36 |35|36|34|36|
| Legacy command /27 |26|27|25|27|
| Jenny target /36 |33|33|32|33|
| Jenny command /27 |24|24|23|24|
| Joe unclassified target /72 |0|17|9|10|
| Melo US bare /9 |9|8|8|9|
| Melo US command /27 |24|25|22|27|
| Melo BR bare /9 |7|6|7|7|
| Melo BR command /27 |14|14|16|20|
| Mac synthetic host target /4 |1|2|2|2|

Continuous container events are not phonetic timing or first-command retention evidence. Joe carriers remain unclassified. Mac host results are not ESP32-P4 execution. Hey Willow clips here are distractors, not a deployed Hey Willow detector benchmark.

Identical totals can hide changed failures. Against original Binary40, wide128 Jenny gains index66 and loses68. Wide128 Joe gains6 but loses12,22,26,28,29,44,49,50. Its Melo targets gain77,83,146,224,225,226,227,289,293,298,299 and lose364. Continued64 loses Legacy46,47; Jenny51; Joe7,12,13,22,28,29,50,67. Full gain/loss records, including negative changes and original source identities, are in `4090-binary40-capacity-consumed-review-v1.json`.

Reused background is separate, never additive new exposure: Legacy10.483944826h and Jenny9.614419583h have zero events in all four models. Melo7.902862222h has C1610/original Binary401/continued640/wide1280. Daily Nick-address improvement and removal of the reused Melo event are useful observations, not proof of a general false-alarm rate. Constructed nearword counts remain diagnostic.

Both continuations substantially reduce broad-training Mix negative events, while broad positive fit deteriorates (see the complete broad-fit findings). This and the consumed tradeoffs do not support another width/seed/negative-pressure sweep. Next investigation should examine the fixed training objective's temporal supervision and its relationship to ordinary short-gap/connected-command failures, using original training data and existing traces before deciding whether a distinct training intervention is justified. Do not tune against consumed evaluation clip identities or independent acceptance material. No threshold/calibration/checkpoint rescue or automatic promotion is selected.

Artifacts: original comparisons and archive manifests are preserved per cohort; `4090-binary40-capacity-continuation-original-v1.json` binds actual execution receipts; `review_binary40_capacity_consumed_v1.py` reproduces this archive/source review. Wide128 final INT8 is155360bytes, SHA2568f585dc94e9a4e5e53fc28144e666970f9c80a30ed3cdadf752f40b860133b46; continued64 is64800bytes, SHA256ab72bf961ce60f009ef00896a546e2d59ccc5b6f218a45677d155e73a9801acf. C161 baseline identity and deployment event policy remain fixed. No human/P4/acceptance claim follows from these consumed diagnostics.
