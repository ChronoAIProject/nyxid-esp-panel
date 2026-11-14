# Original raw FP32 recognition diagnosis

The full four-model native numerical test passed; actual recognition on the old reused development set still has separate failures. All552registered raw-FP32 points were scored and independently audited. No workpoint was chosen.

The matched material contains36positive clips/word from one reused Piper voice (en_GB-semaine-medium), three speeds and three render variants with standalone/command carriers. Background exposure is10.483944826388887h, all12whole streams. These are consumed diagnostics, not new voice-independent development or acceptance. Generated allocation boundaries are not human final phonemes.

Across each entire registered region satisfying FAR upper95<=.5/h:

| Model | Points in region | Any after source /36 | Complete prefix /36 | Common false clips /288 | Background events |
| --- | ---: | --- | --- | --- | --- |
| Official Hey Willow | 10/13 | 0–5 | 0–5 | 0 | 0 |
| Same-recipe Willow float | 0/138 | No conforming points | No conforming points | — | — |
| Same-recipe Willow integer-trained raw FP32 | 138/138 | 0 | 0 | 0 | 0 |
| Same-recipe Nyx float | 138/138 | 33–35 | 30–34 | 13–25 | 0 |
| Same-recipe Nyx integer-trained raw FP32 | 0/138 | No conforming points | No conforming points | — | — |
| Official Hi ESP engineering reference | 10/13 | 0–36 | 0–17 | 0 | 0–1 |

These are complete-region ranges, not selected operating points or equal numeric thresholds. Official native event policy differs from the explicit FP32 event policy; this limitation stays visible. Zero background events gives upper95 .285744758/h; one gives .452488505/h. The silent Willow integer-trained model is not a successful detector. Original full curves contain all missing and negative outcomes.

The Nyx float route has actual negative-word confusion even when ambient FAR passes. Willow float's failure is not solely early-versus-complete accounting:28–29/36clips have no after-source event across its full grid, and no point passes ambient FAR. Weak official Willow coverage on this same single synthetic voice also limits claims about mature-model quality. Do not infer a unique root cause or fix either issue by changing scoring boundaries.

The current matched pool/flat experiment is separately registered and has completed four fixed10000-step runs from equivalent readout functions. Its original540waveform numerical screen is ongoing. It tests readout parameterization/representation sensitivity, including altered Adam dynamics; it is not a replacement for this same-recipe comparison and is not evidence that data coverage improved. Resolve its original numerical gates before any new training intervention. New source/voice coverage must retain phoneme/runtime/native-float provenance and role isolation, and must not turn this consumed diagnostic set into fresh development. Original INT8 failures, unopened independent data and frozenv30 stay unchanged.
