# Binary40 final float versus original INT8

C161 remains baseline; Binary40 remains a research candidate. The verified final-weight float replay indicates that normal voice regressions and the observed everyday background error are already present before quantization. Quantization contributes some misses, but these results do not justify treating QAT or calibration changes as the main remedy.

| Consumed diagnostic | Final float | Original INT8 |
|---|---:|---:|
| Builtin target | 9/12 | 8/12 |
| Continuous 0.5 s | 7/12 | 7/12 |
| Continuous 2 s | 9/12 | 8/12 |
| Daily negative events | 1/96 | 1/96 |
| Legacy target | 36/36 | 36/36 |
| Jenny target | 33/36 | 33/36 |
| Joe target | 20/72 | 17/72 |
| Melo target | 53/72 | 53/72 |
| Mac host Nyx | 2/4 | 2/4 |

Jenny and Melo target clip identities agree, including normal-hit losses relative to C161. The Melo background event occurs at the same step 4311 / 124.26 s in float and INT8. Reused background events are Legacy 0/0, Jenny 0/0, Melo 1/1 (float/INT8); no new independent exposure is claimed. Host synthetic results do not establish P4 or human performance.

The first float replay (v1) omitted loading checkpoint weights: all its model metrics are invalid. Its original report and failure logs are retained. The v2 replay strictly loaded and checked every final-weight tensor; a separate native-step/Keras check passed. Duplicate phase indices overwrote 24 synthetic trace files (12 Legacy, 12 Jenny), not the initially estimated 66. The recovery plan was pushed before recovery; only those 24 were recomputed. Every recovered file SHA256 and event list matches the original v2 record exactly. Unique phase/index filenames prevent the collision in v3. All 1760 float and saved INT8 trace hashes were verified, and all 1768 archive members reopened and verified. No background or INT8 reinference, training, threshold or calibration change occurred during recovery.

See results/4090-binary40-float-replay-original-v3.json, 4090-binary40-physical-float-check-original-v2.json and 4090-binary40-float-trace-repair-review-v1.json for authoritative records. Evidence archive SHA256: c1aabc84a40643669acbc4d3270980c3f84aff9138a22615f52c31e9d16d537c.

Next: select a distinct training intervention against the documented history, addressing float-level crossvoice/connected-command losses and ordinary-speech confusion without optimizing on independent acceptance material. No candidate promotion or new training recipe is approved by this diagnostic alone.
