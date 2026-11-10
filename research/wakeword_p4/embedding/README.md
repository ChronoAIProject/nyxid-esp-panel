# Pretrained speech embedding experiment

This is an alternative to the rejected small MicroWakeWord classifiers. The
production/default wake word is still Hi ESP. The compact **Hey Nyx** head has
not established recognition acceptance. Standalone inference timing is not
whole-application acceptance.

## Frozen source and numerical reconstruction

Use the v0.5.1 openWakeWord release assets listed with SHA256 identities in
[`../results/embedding-feasibility-v3.json`](../results/embedding-feasibility-v3.json).
The Google embedding has 328,224 parameters. Its float graph is reconstructed
from the original weights and each operator's padding, strides, pooling and
activation options. Compare to the original interpreter **before** quantizing.
The full-window reconstruction agrees within 5.35e-5 on 28 training/control
sequences, but full-window INT8 costs about 606 ms on the P4 and is rejected.

Temporal-convolution caching retains the same weights and temporal context.
Eight new mel frames arrive per 80 ms call; four initial frames align the first
76-frame complete window. Compare only fully warmed outputs to the original
window graph: 105 checked outputs agree within 5.92e-5. Do not reset resource
variables using `reset_all_variables()` between recordings: it fails to clear
the first nine steps' resource history. Construct a new interpreter per clip.

Quiet calibration adds explicit -8/-4/0/2 histories to training-only PTQ
representatives. Streaming silence maps to -8; a whole loud clip's dB floor
would obscure this. The resulting frozen INT8 backbone has SHA256
`af0e12dd0cd6f368b0c2c24b2b5f4ff0df85af8175185d8e6bb6ef07e69e233a`.
Training-only embedding cosine mean/min are 0.9860/0.9676. This measures
embedding fidelity, not wake recall.

Use LiteRT `BUILTIN_REF` for host extraction and numeric goldens. XNNPACK and
`BUILTIN_WITHOUT_DEFAULT_DELEGATES` differ from the P4's reference-compatible
outputs by up to 36 and 9 byte counts in the audited fixture. Their speed is
not a reason to substitute them in this experiment.

## Exact integer activation folding

The float graph's 19 `LeakyRelu`/`Maximum(-0.4)` pairs are not ReLUs. The
`fuse_activations.py` helper runs each original fixed INT8 pair on all 256
possible bytes and writes a `NYX_INT8_LUT` custom node with the resulting map.
No learned weights or quantization scales change.

```sh
python research/wakeword_p4/embedding/fuse_activations.py \
  --model /private-run/embedding-stream-int8-quiet.tflite \
  --expect-pairs 19 --output /private-run/activation-lut
```

The P4 registration accepts exactly 256 opaque option bytes, one INT8 input
and one INT8 output with equal element counts. Keep the model buffer alive
for the interpreter lifetime. Invocation is:

```cpp
output[i] = table[int(input[i]) + 128];
```

The public helper reproduces the measured model bytes exactly, SHA256
`78816bed1f61f2ef5bfb2b83b056dd481849bc3fd11c9c8a29ae47281811435a`.
Exhaustive pair checks alone do not prove the complete model: the development
P4 also matches all 114 x 96 expected output bytes, including cold state.
Cached inference falls from about 79 ms to 70 ms per call with the pinned
ESP-NN 1.1.2 kernels. A tested newer ESP-NN commit is numerically exact but
slower at about 154 ms, so it is rejected.

Model-placement probes also match all114x96 bytes per case. Flash model with
internal arena costs70.41ms median; PSRAM model with internal arena69.85ms;
PSRAM model with PSRAM arena70.36ms. All use a96KiB tensor arena
(76,604bytes used) and a separate16KiB internal variable arena, on360MHz
P4/200MHz PSRAM. The native FFT scratch is allocated through the same
PSRAM-preferring heap policy. This allows an eventual whole-app test to
measure resource margin using the actual model expansion placement.

## Native frontend

`NativeMel.block()` in `native_mel.py` consumes exactly 480 preceding plus
1280 new mono samples at 16 kHz. Use the source graph's 512-point window,
257x32 mel matrix and 160-sample hop. The equations are windowed FFT,
magnitude squared, source mel weights, power log10, an 80 dB floor **per
streaming block**, then the source `/10+2` transform. Whole-recording flooring
is a different feature distribution and is not used for head training.

The P4 diagnostic implements a 512-point radix-2 FFT and the 229 nonzero
mel coefficients. Coefficients retain original float32 bits. On 33 fixed
training/control blocks it matches all 8,448 quantized source-graph inputs;
maximum float discrepancy is 6.80e-5. Median/max latency is 2.813/2.815 ms
with 6,148 bytes of scratch. The host helper also reproduces zero quantized
differences. Always regenerate goldens for the selected backbone's own
quantization scales.

## Classifier experiment and remaining acceptance

Freeze the backbone before training the wake head. Use existing source/voice
partitions, training and validation only. Each TTS clip has an original and
room/gain/noise version at its original speed, a 2.4-second lead and a
one-second tail. Human/environment features keep their original source
partitions. Heads sample only complete warmed histories, starting at frame
25 or later, so the first nine cold embeddings never enter their 16-frame
input histories.

The augmented variant has 24 reflection taps at delays80..2399samples, a
unit direct tap, Gaussian reflection weights with .2 standard deviation and
exponential decay500..1400samples. Normalize speech RMS to .04.. .20 of
full scale using a655-source-RMS floor. Add AR(1) noise with pole.5.. .96
at8..30dB SNR. Round/clip to signed16bit before the block frontend. Keep
per-source variant seeds, checksums and keyword-end metadata in private indexes.

The prespecified head is Conv2D24(3x96), ReLU, temporal MaxPool14, Dense16,
ReLU, Dense1 sigmoid. Fit 10k steps with seed726 and checkpoints
2,500/5,000/7,500/10,000; batch128 contains 32 positive, 48 TTS-negative and
48 continuous-negative histories. Sample TTS engines uniformly and continuous
negatives by duration. Negative loss weight is5. Learning rate is .001,
multiplied by .3 after5k and .1 after8k. Positive window endpoints are
.03.. .32 seconds after the existing keyword endpoint metadata; those
endpoints retain synthesis/alignment uncertainty.

Export fixed batch-one INT8 heads using 512 training histories and the
physical byte-range endpoints. Verify that input scale/zero are exactly
1/128 and0 before copying embedding bytes directly. The untrained interface
diagnostic is10,544bytes, with five operators and a mean float/INT8 output
error below.001. It is not an accuracy result.

Validation freezes a three-step score average, 80 ms cadence, a 25-step
refractory and rearming below half threshold. Use the prespecified
19-threshold grid. Prefer operating points with <=.5 continuous events/hour
and <=1% false clips in each engine/corpus subgroup, then maximize macro
recall. Freeze the checkpoint/threshold before existing paired/long
regressions. Newly reserved 6.455h English and3.126h Mandarin remain unread.
The exact grid, recipe and checkpoint tie order are recorded before fitting
in [`head-plan-v1.json`](head-plan-v1.json). `requirements.lock` records the
host package versions used for this experiment.

The helpers implement this frozen v1 topology/data schema. To prepare TTS
histories from the existing `features.json` plus `dataset/` directory:

```sh
python research/wakeword_p4/embedding/prepare_head_features.py \
  --source-features /private-run/source-features \
  --model /private-run/embedding-stream-int8-quiet.tflite \
  --expected-model-sha256 af0e12dd0cd6f368b0c2c24b2b5f4ff0df85af8175185d8e6bb6ef07e69e233a \
  --window /private-run/native-fft-window.npy \
  --mel-filter /private-run/native-mel-filter.npy \
  --output /private-run/head-features-tts-v1 --workers 3
```

For continuous negatives, substitute `--jobs /private-run/negative-jobs.json`
for `--source-features` and name the output `head-features-negatives-v1`.
Each job records `id`, `source`, `source_sha256`, `split`, `kind`, `label`,
`variant`, `engine` and optional `keyword_end`. Explicit testing jobs are
rejected. Use the unfused quiet backbone for host extraction; firmware LUT
equality is established separately on the P4. The extractor writes feature
hashes, input identities, an index and a completion marker into a fresh
private directory.

```sh
python research/wakeword_p4/embedding/train_head.py \
  --data-root /private-run \
  --plan research/wakeword_p4/embedding/head-plan-v1.json \
  --output /private-run/head-conv24-v1
python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root /private-run \
  --plan research/wakeword_p4/embedding/head-plan-v1.json \
  --training /private-run/head-conv24-v1
```

Training requires both complete feature directories and verifies source split
disjointness and array hashes. Calibration binds the head, backbone, event
policy and exact preparation/index identities. Public-helper replay reproduces
both TTS variants and all6,230 frames of a continuous training-only fixture
byte-for-byte. Export replay matches all64 head output bytes and the complete
untrained TFLite artifact. These are interface/reproducibility checks; trained
recognition results are separate.

### DNN and hard-negative follow-ups

The 32/64-wide DNN alternatives flatten the same16x96 input and use two
Dense/LayerNorm(epsilon1e-5)/ReLU stages followed by a sigmoid. The source
reference and fixed training recipes are in `head-dnn{32,64}-plan-v1.json`.
Strict validation selects low-recall points:32-wide32.94% and64-wide16.56%.
These cannot replace Hi ESP.

Hi ESP itself responds to many of its own near phrases, so the arbitrary1%
near-phrase gate is not a demonstrated equivalence criterion. Keep those
failures and separately freeze continuous-FAR diagnostic points before
development regression. Set `selection_policy` to `continuous` explicitly in
a new plan to maximize validation macro recall at <=.5 continuous events/hour;
omitting it preserves strict-first selection. This is a selection policy,
not permission to declare recognition acceptance.

The selected64-wide secondary diagnostic scores41/50 paired wake clips versus
Hi ESP48/50,2/100 ordinary-sentence false clips versus0, and3 English plus3
Mandarin continuous events versus0/0. Those are existing development corpora;
the newly reserved testing corpora remain unread.

`mine_negatives.py` scans training negatives only, verifying feature and teacher
identities. It takes up to two separated TTS peaks per variant, and up to256
one-second continuous peaks per source, at raw scores>=.05. For example:

```sh
python research/wakeword_p4/embedding/mine_negatives.py \
  --data-root /private-run --teacher-plan /private-run/frozen-teacher.json \
  --backbone-sha256 af0e12dd0cd6f368b0c2c24b2b5f4ff0df85af8175185d8e6bb6ef07e69e233a \
  --output /private-run/mined-training.json
```

A warm-start plan must bind `initial_weights_sha256`; pass that exact `.keras`
file with `--initial-weights`. A mining plan must also bind
`mined_negatives_sha256`, passed with `--mined-negatives`. The trainer verifies
the mining index/backbone, training-only labels, feature hashes and warmed
window boundaries. Separate RNG streams consume identical ordinary draws
before replacing half of each negative category. Positives, the other random
halves and all512 PTQ representatives stay identical to control.
The optional recipe field `hard_negative_stride` can replace a smaller share:
`4` replaces one quarter of each negative category; the historical default
remains`2`. The same original draws are consumed before replacement. Run
`python -m unittest test_head_sampler.py` from this directory to check that
labels, positives and the unmodified draws remain comparable, invalid strides
and held-out mining are rejected, and the default half-mining path is unchanged.

The5,238-window matched experiment fails: control41/50 versus mined35/50
paired recall; English false events6 versus1, Mandarin2 versus2. The smaller
false count does not compensate for lost wake recall. Full aggregates, model
and plan hashes are in `../results/embedding-head-dnn-v3.json`.

The subsequent prespecified research comparison uses the upstream default128-wide
head,40k steps at initial LR.0001, and full versus confidence-filtered training
loss. The optional recipe field `loss_confidence_filter` drops already-easy
training negatives below.001 and positives at/above.999, matching the upstream
confidence bounds. It averages the remaining weighted loss and returns zero
loss/gradient for an empty active batch. All evaluation still scores every
window. This adapts upstream training; it does not claim the same optimizer
schedule or accumulation implementation. Both runs fail development regression:
full loss36/50 and confidence loss39/50, versus Hi ESP48/50. Continuous English
false events are1/1 and Mandarin3/2, versus reference0/0. See
`../results/embedding-head-dnn128-v3.json`. No classifier from these runs has
been installed in the complete application.

### Additional environmental training views

`prepare_environment_views.py` keeps the original v1 feature directories and
adds training-only views. Variant2 mixes a verified interval from one of the45
existing training environmental recordings. Variant3 also applies synthetic
second-order shoebox reflections and a second-order60..180Hz to2,500..7,500Hz
bandpass. Rooms are synthetic, not measured impulse responses. Direct arrival
stays at time0. Neither variant changes sample rate, duration or speaking speed.
Both use8..25dB SNR and the previous.04...20 speech-RMS range.

The frozen extraction plan binds source/background indexes and jobs, backbone,
window/filter arrays and extraction/frontend code. Held-out audio hashes are
excluded from both source and background pools. Worker checks verify actual
WAV hashes and mono16kHz format; mixed PCM and embedding hashes are recorded.
Silent background intervals are retried and never replaced by synthetic noise.

```sh
python research/wakeword_p4/embedding/prepare_environment_views.py \
  --data-root /private-run --plan /private-run/environment-views-plan-v2.json \
  --output /private-run/head-features-environment-v2 --workers 4
```

After all extractions complete, bind the new directory's `index_sha256` and
`complete_sha256` under `extra_training_features` in a new training plan.
Each entry has `directory`, `index_sha256` and `complete_sha256`; the directory
must be a unique child of the data root. Extra directories reject validation
rows. Calibration still loads the original validation directories, and all512
PTQ representatives still come from original training views in both arms.
The24-view six-engine replay and source-timebase/shape checks pass, as do
modified-index/held-out-row/path rejection and unchanged representative checks.
All 20,140 added views pass checksums, shapes, timebase and source/background
split checks. The matched control reproduces all four prior model artifacts
exactly. Added environments improve paired recall from41 to43/50, ordinary
false clips from3 to1/100, English false events from6 to1/5.549h and Mandarin
from2 to0/1.294h. This is still below Hi ESP48/50 with zero shared ordinary
false events. See `../results/embedding-head-environment-v3.json`.

The same64-wide float and INT8 heads on frozen paired features score42/50
and41/50 respectively, with26 negative events each. The maximum smoothed
score difference is.0800. Head quantization loses one wake clip here but does
not explain the whole recognition gap. This says nothing about the separate
backbone quantization effect. See `../results/embedding-head-quantization-v3.json`.

The next training-only extension fills the remaining64 entries of the existing
local Qwen age/gender/accent/texture grid. In `training/mlx_samples.py`,
`--profile-start 32 --profiles 96 --batch 8` skips the already generated first32
conditions and creates960 new normal-speed source clips. These descriptions
are not verified independent human identities. Initial waveform checks pass,
but32-clip independent phoneme/ASR diagnostics show pronunciation uncertainty;
source labels and recordings remain intact. A frozen three-arm recipe compares
control, all new views and a consistency-screened view on unchanged validation.
No reserved scoring or classifier acceptance follows from synthesis alone.

All960 source recordings finish with waveform checks and no source-hash
overlap with existing head data. The same four acoustic views produce3,840
new histories. Complete phoneme/ASR diagnostics support88 of192 new positives
under the frozen union rule;104 uncertain positives remain outside the separate
screened view. This is not a proof that those104 are mispronounced. All768
new negatives remain, yielding3,424 screened histories. Original labels,
recordings, feature arrays and held-out rows are preserved.

`screen_tts_features.py` requires complete hash/label-matched audits and
training-only source/features. It keeps a positive when the first12 greedy
phones contain optional-h/eɪ-or-eː-or-e-or-ɛ/n/ɪ-or-i-or-iː/k/s, **or** ASR
normalizes to `hey nicks`, `hey nix`, `hey nyx` or `hey knicks`. It excludes a
negative if either recognizer predicts a canonical wake. The new view refers
to the exact original arrays. Supply matching repeated `--features`/`--name`
arguments, plus `--source`, `--phonemes`, `--asr`, `--plan`, `--output-root`
and a new `--report` path. Invalid audit identities or held-out sources fail
before output. The32-source pilot retains92 feature rows byte-for-byte.

The subsequent control/full/screen10k-step comparison shares the environmental
candidate's initial float weights, LR.0001, seed726, original validation and
all512 original PTQ histories. Final feature-index/completion hashes are bound
before fitting. Data evidence is in `../results/embedding-qwen-extension-data-v3.json`;
all three calibrations and nine development regressions now complete. Control/full/
screen score39/43/41 of50 paired wakes, versus Hi ESP48/50. Each has two
English and one Mandarin false event, versus zero for Hi ESP. None exceeds
the preceding environmental candidate (43/50, English1, Mandarin0). Reject
all three follow-ups; more synthesized conditions did not establish improvement.
See `../results/embedding-head-qwen-extension-v3.json`. Reserved scores remain
unread. A standalone P4 classifier-head numerical/timing diagnostic follows;
it is not recognition or whole-application acceptance.

Remaining gates include recognition, human/acoustic comparison to Hi ESP,
actual whole-app flash/heap/CPU fit, immediate speech continuity, rearming,
quiet periods, long-call echo and reconnection. The 73 ms sum of standalone
frontend/backbone timings leaves little scheduling margin and must not be
reported as measured whole-app timing. The current raw and compressed model
sizes also require an actual unchanged-slot application build.

All device diagnostics verify the development board MAC and running slot,
flash the app only, then restore and verify the original app, selected service
and Hi ESP listener. Never run these tests on the dining-room production
board. Credentials, generated model binaries, PCM and private captures belong
outside git. No upstream model binaries are bundled here; release-asset
provenance still needs resolution before distributing a replacement.

### Head compatibility and temporal-order follow-up

`head_probe/` and `prepare_head_probe.py` build a private standalone classifier
probe. `run_head_probe.py` verifies the development app/USB identity, tests all
three placements, rejects partial or mismatched results and restores the original
app/service/Hi ESP. See `head_probe/README.md` for fixed dependencies, numerical
fixtures, per-operator traces and the local upstream-FC compatibility shim.

A separately frozen two-arm20k-step experiment uses two32-channel temporal
convolutions, each spanning five embeddings, followed by positional max pooling.
Its nine-embedding receptive field retains time order before pooling, unlike
the earlier three-embedding Conv24 pool. Both new arms share original512 PTQ
histories, training draws, data, initial inference weights, BatchNorm and schedule;
one adds training-only dropout.15 after pooling. BatchNorm folds into the export.
This changes neither audio speed nor PCM/Opus configuration. The20,769-parameter
models finish fitting/calibration. Frozen validation selects plain5k at.3 and
dropout20k at.3 (macro recall90.46%/95.26%,2/3 events in7.304h). Existing
development paired/English/Mandarin regressions are running; validation and
topology rationale are not recognition acceptance evidence.
No reserved scores or whole-application replacement are authorized by synthesis
or a numerical probe alone.

The temporal plain/dropout development regressions finish42/44 of50 with
English1/6 and Mandarin0/1 false events. Neither replaces Hi ESP. The fixed
equal mean then selects.4 on original validation, before development scores:
46/50 paired, ordinary1/100, English1/5.549h, Mandarin0/1.294h. This improves
on the previous43/50 research baseline but remains below Hi ESP48/50 and
zero shared false counts. `evaluate_equal_mean.py` checks frozen member/model/
score identities and equal timebases; an exact1/1536 numerator matches sums
of six raw sigmoid bytes in22 streamed arithmetic checks. Invalid shapes,
NaN, off-grid and out-of-range scores reject. It fits no ensemble weights.
See `../results/embedding-head-equal-mean-v3.json`. This is the next research
candidate, pending actual unchanged-slot whole-app feasibility and acoustic/
reserved evaluation, not a release.

The nontracing classifier-head P4 probe passes299cases in all three placements
(897exact outputs). Medians are2.867/2.893/2.977ms and tensor arena use7532bytes.
The original app, slot, service and Hi ESP listener restore after each of the
five hardware probes, including failures. Full intermediate/rounding evidence
and limitations are in `../results/embedding-head-p4-v3.json`.

The temporal head needs a separate resource correction. Its initial full-width
convolution makes pinned ESP-NN request about248KiB scratch and fails the64KiB
probe arena. `reshape_temporal_head.py` moves the96feature values from spatial
width to channels, inserts a reshape at the input, and keeps every constant byte,
quantization parameter and external input/output shape. Bounded int32 sums make
the reordering exact. Supply the frozen model and299-case input file with their
expected SHA256 arguments; `--output` must name a fresh private directory.

The transformed head matches all299 host cases and all897 P4 outputs in the
same three placements. It uses18,956 tensor bytes; median invocation times are
892/827/827microseconds (flash/internal, PSRAM/internal, PSRAM/PSRAM). The
original app/service/Hi ESP restores afterward. This removes the scratch-memory
blocker without retraining or changing any recognition score. See
`../results/embedding-temporal-layout-p4-v3.json`. Complete-app fit/scheduling
and physical recognition remain separate measurements.


The first complete-app equal-mean builds fit the existing OTA slot at
6,445,840bytes, but fail real-time processing: approximately118ms per80ms
input step. Moving the backbone arena into internal memory and pinning the
listener to core0 did not fix it. A whole-app stage/operator profile attributes
about106ms to the backbone, mostly convolution. See
`../results/embedding-full-app-resources-v3.json`; these are failed resource
experiments, not wake-recognition acceptance.

`explicit_same_padding.py` rewrites the nine unit-stride horizontal SAME
convolutions as explicit zero-point PAD followed by VALID convolution.
ESP-NN1.1.2 dispatches implicit padding to its slower generic path; the explicit
graph permits the existing P4 vector implementation. It preserves every
existing constant buffer, quantization, output shape, sample rate and cadence,
and checks int32 accumulator bounds. Before LUT folding, both source and
rewritten graphs must reproduce the114frozen streaming fixtures exactly.
The folded graph must then pass P4 numerical checks independently. This is
a dispatch optimization and does not improve the classifier's46/50 paired
recall or change its frozen threshold.

### Controlled follow-ups that did not improve recognition

The temporal Qwen control/screen continuations use identical initial weights,
seed727, original validation and PTQ representatives. The screened equal mean
still detects46/50 paired wakes, with6 English and2 Mandarin false events.
Reject it; adding screened synthetic conditions did not improve the frozen
46/50 candidate. See `../results/embedding-temporal-qwen-v3.json`.

Quarter-hard temporal continuation replaces only one quarter of negative
training draws with5268 training-only mined windows. Its PTQ representatives
remain byte-identical to control. The equal mean falls to45/50, with ordinary
1/100, English1/5.549h and Mandarin0/1.294h false events. Reject it; reducing
training mining intensity did not recover the recognition gap. See
`../results/embedding-temporal-hard-quarter-v3.json`.

Replacing positional maximum with an average preserves all original learned
tensor shapes and initialization. Original validation freezes step7500 and
threshold.5; the fixed Dense/temporal equal mean freezes.4 before development
scoring. It remains46/50, ordinary1/100, English1/5.549h, Mandarin0/1.294h.
There is no measured improvement over the prior equal mean. See
`../results/embedding-temporal-mean-pool-v3.json`. Reserved scores stay unread.

The next prespecified positional-pooling comparison uses a bias-free8x1
depthwise filter over the32 temporal channels. Both arms start with exactly
uniform1/8 weights and identical original convolution/BatchNorm/Dense tensors.
One freezes the filter; the other lets it learn feature positions. This tests
whether pooling away position loses phrase-order information. It adds neither
a microphone stream nor an audio transformation. It is a research hypothesis,
not established improvement or MCU compatibility.

`prepare_positional_pool.py --source <verified-temporal.keras>
--source-sha256 <sha> --representatives <training-only.npy>
--representatives-sha256 <sha> --output <fresh-private-directory>` prepares
the two initial models and a hash/equivalence report. All512 training fixtures
must have identical arm predictions and agree with average pooling within
3e-6 on both pre-Dense features and output probabilities. The current fixtures
are exactly equal. Bind the resulting model hashes in the two frozen arm
plans before running the existing `train_head.py` and `evaluate_head.py`.
The common schedule, source partitions and selection rule are in
`head-positional-pool-comparison-plan-v1.json`; no reserved data may be used
to fit or select. A promising result still needs frozen development regression,
separate P4 depthwise numeric/layout checks and the whole-app/voice gates.

The positional-pool experiment now finishes: validation macro recall is
92.370% fixed and93.113% learned, below the preceding average-pool93.522%
and original temporal95.261%. It does not advance to device/reserved scoring.
Initial arm features/predictions and all512 PTQ representatives are exactly
equal. See `../results/embedding-positional-pool-v3.json`.

`audit_phase_sensitivity.py` freezes a48-source training-only pilot across
six engines, balanced labels, and offsets0/16/160/320/640/960/1264samples.
Move silence from the tail to the lead; speech bytes and total recording
length stay exact. Phase0 must reproduce each original embedding array.
All24positive sources remain above the fixed.4 threshold, though peak changes
reach.497; one of24negative sources changes threshold outcome. This is fitted
training data, not proof of accuracy or the cause of a physical miss.
The initial harness fails on relative feature-path resolution before results;
the explicit retry resolves/contains feature paths under the private data root.
See `phase-sensitivity-plan-v1-retry1.json` and
`../results/embedding-phase-sensitivity-v3.json`.

`prepare_phase_views.py --data-root <private-root> --plan
phase-views-plan-v1.json --output <fresh-private-feature-directory>
--workers 4` adds one uniformly selected1..1279sample offset per original
training TTS source, with original waveforms/durations and shifted endpoint
metadata. It rejects held-out overlap, verifies source/model/frontend hashes,
checks speech-byte preservation on every view and hashes PCM/features.
The original arrays and held-out data stay immutable. Bind the completed
feature index and completion hashes before the matched training comparison
in `head-temporal-phase-comparison-plan-v1.json`; this is an unproven
augmentation hypothesis, with no relaxed evaluation threshold.

### Independent training-marker audit

`audit_ctc_endpoints.py` inspects the 178 standalone positive sources from the
already frozen training-discrimination sample. Its plan
`plans/ctc-endpoint-audit-v1.json` preserves 14 command-followed sources
separately; a whole-command endpoint cannot stand in for the wake prefix.
All source hashes, role metadata, original jobs and held-out hash separation
are checked before loading the existing local phoneme model. Model files and
runtime packages are pinned. No wake detector, room recording or playback is
involved, and all original audio, labels and feature arrays stay intact.

```sh
python research/wakeword_p4/embedding/audit_ctc_endpoints.py \
  --data-root "$WAKE_DATA" \
  --parent-plan research/wakeword_p4/embedding/plans/training-discrimination-audit-v3.json \
  --plan research/wakeword_p4/embedding/plans/ctc-endpoint-audit-v1.json \
  --model "$PHONEME_MODEL" --output "$NEW_ENDPOINT_AUDIT_OUTPUT"
python -m unittest discover -s research/wakeword_p4/embedding -p test_ctc_endpoints.py
```

The independent recognizer receives no expected text. Greedy CTC collapses
consecutive repeats before removing blanks and retains every predicted token,
including uncertain text and all-blank cases. Each token run records its local
convolution support (320-sample stride, 400-sample support). The transformer's
full-utterance context and spiky CTC emissions mean these are not exact acoustic
phone boundaries. Compare original markers with the last CTC support and the
unchanged RMS endpoint independently; neither automatically replaces labels.
This plan is frozen before inference and establishes no wake recognition result.

The completed [178-source report](../results/embedding-ctc-endpoints-v1.json)
records all-blank counts and binds all per-source CTC rows by hash. OpenAI's median original
marker-to-last-CTC-support gap is 440 ms, versus 36..225 ms for the other
cohorts; 10/32 OpenAI markers are also more than 200 ms beyond the RMS end.
CTC support typically precedes the full phone tail, so this does not justify
using its timestamp as a label. It independently reinforces the existing
energy-based concern without establishing a cause of acoustic misses.

All-training accounting verifies 1,959 unique standalone sources: 56 exceed
the unchanged 200 ms RMS lateness criterion. The prior file-end restriction
covers 34 and misses 22, including a fixed-sample marker 840 ms beyond the
RMS end. All 56 happen to be OpenAI sources; the rule is provider-independent.
`prepare_training_endpoints.py` and
`plans/training-endpoint-overrides-v1.json` produce a separate manifest for all
four original/environmental views, affecting 224 training endpoints.
No original index or payload is rewritten. The sampler consumer keeps PTQ
representatives on original metadata and binds the override hash before
matched fitting. Preparation/CTC results alone cannot accept or deploy a model.

Preparation now completes all 43,348 original/environmental rows with 224
overrides (112 base, 112 environmental); all 3,068 nontraining rows remain
untouched. The pure-metadata manifest is tracked as
`plans/training-endpoint-overrides-v1-records.json`. The optional
`train_head.py --training-endpoints <manifest>` argument must match the plan's
`training_endpoints_sha256`. It changes positive sampling anchors only;
representative sampling always reads the original row metadata. Omit the
argument for historical behavior. Unknown/duplicate IDs, role/hash/anchor
conflicts, nonfinite values and out-of-range warm histories are rejected.

The [real-data precheck](../results/embedding-training-endpoints-v1.json) loads
40,627 training rows and compares the retained legacy sampler against current
default behavior: 20 batches per member are exact, and both 512-example PTQ
files reproduce their historical bytes. Treatment keeps labels, negative draws
and RNG state exact while changing 35/33 sampled positive histories in those
diagnostic batches. Six sampler tests plus seven CTC/source-boundary tests
pass. No held-out audio or validation recognition score informs this precheck.

`plans/all-training-endpoint-continuation-v1.json` freezes matched dense and
temporal continuations with their original weights, seeds and 10,000-step
recipes. Its member plans bind the manifest; retain the existing controls and
their PTQ hashes. Select four prespecified checkpoints on original validation,
then evaluate exactly one equal mean before any development scoring. For each
member, use a fresh training output:

```sh
python research/wakeword_p4/embedding/train_head.py \
  --data-root "$WAKE_DATA" --plan "$ENDPOINT_MEMBER_PLAN" \
  --initial-weights "$ENDPOINT_INITIAL_WEIGHTS" \
  --training-endpoints research/wakeword_p4/embedding/plans/training-endpoint-overrides-v1-records.json \
  --output "$NEW_ENDPOINT_TRAINING_OUTPUT"
python research/wakeword_p4/embedding/evaluate_head.py \
  --training "$NEW_ENDPOINT_TRAINING_OUTPUT" --data-root "$WAKE_DATA" \
  --plan "$ENDPOINT_MEMBER_PLAN"
```

Both 10k continuations and all eight checkpoint calibrations now complete.
Actual exported PTQ files retain the control hashes, and all source-feature
inputs and initial weights match. Both heads select 2.5k checkpoints. Their
exact model/cached-score identities are frozen before the single equal-mean
calibration in `plans/head-equal-mean-all-endpoints-v1-plan.json`.

The [completed validation report](../results/embedding-all-endpoint-validation-v1.json)
rejects this recipe: at the same .4 threshold, macro recall is 94.3065%, versus
retained control 94.6845%. Both have one event in 7.303778 continuous hours
(upper 95% .649508/h). Worst confusing-corpus false fraction remains 13.2143%,
and approximate endpoint-relative median latency rises from 130.375 to 139.5 ms.
The accounting gap is real, but this bounded-anchor/10k continuation does not
establish a recognition improvement. Stop before development/final scoring or
physical playback; retain the original development-board app. These results
do not settle other timing-supervision or representation hypotheses.

### Generic channel-EQ continuation

`audit_capture_transfer.py` characterizes one explicitly hash-bound synthetic
playback and private microphone capture. It restricts correlation alignment to
complete playback overlap, then reports six-band power ratios/coherence.
The recorded pair has waveform correlation .295 and RMS ratio .0848, with
different attenuation across bands. This diagnostic is post-hoc and contains
room/noise effects; it does not isolate a microphone transfer function or
establish a population cause. No recording or fitted filter is exported.
See `plans/capture-transfer-diagnostic-v1.json` and the aggregate result
`../results/embedding-capture-transfer-diagnostic-v3.json`.

The independent training hypothesis uses generic RBJ peaking filters at
125/315/800/2000/5000 Hz, each drawn uniformly from -12 to +12 dB with Q=.7071.
`prepare_eq_views.py` reuses the completed environmental renderer's exact
source/noise/room RNG sequence, adds separate deterministic EQ gains, restores
whole-mix RMS and uniformly limits peaks to 30000. Samples, rate and speed stay
unchanged. It reads only existing training sources; microphone captures are
not training inputs. The original renderer/head trainer remain byte-identical.

```sh
python research/wakeword_p4/embedding/prepare_eq_views.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_DATA/eq-views-plan-v1.json" \
  --output "$WAKE_DATA/head-features-eq-pilot-v2" --limit 48 --workers 3
python research/wakeword_p4/embedding/prepare_eq_views.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_DATA/eq-views-plan-v1.json" \
  --output "$WAKE_DATA/head-features-eq-v1" --workers 3
python research/wakeword_p4/embedding/audit_eq_views.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_DATA/eq-views-plan-v1.json" \
  --features "$WAKE_DATA/head-features-eq-v1" \
  --pilot "$WAKE_DATA/head-features-eq-pilot-v2" \
  --output "$WAKE_DATA/eq-views-audit-v1.json"
```

Use fresh output paths. The first preliminary 48-view pilot was preserved
locally before adding separate pre/post-EQ PCM hashes to the extraction
metadata; the second pilot and full run bind the final extractor hash.
All 20,140 full views pass source/background/timebase/PCM pairing, feature
hash/shape and split isolation checks. The 48 balanced pilot rows reproduce
the full run exactly; their pre-EQ features reproduce the completed reference
arrays. All 43,730 original indexed arrays remain unchanged. The audit covers
extraction provenance, not recognition quality.

`plans/eq-continuation-v1.json` freezes the two matched continuation recipes
before fitting. Each treatment adds EQ views to its existing dense/temporal
control; initialization, 10,000-step budget, class/engine recipe, training-only
PTQ representatives and original validation policy remain fixed. Completed
index/completion hashes are bound in `plans/head-{dense,temporal}-eq-plan-v1.json`
before training. The matched controls are reused exact completed experiments,
not newly independent trials. After independent checkpoint selection, the
only planned combination is their equal mean; no weight is fitted from the
private capture or development regression.

### Training-only lexical auxiliary experiment

`plans/lexical-auxiliary-{preparation,control,treatment}-v1.json` freeze a
different discrimination experiment before new training or detector scores.
The existing dense64 binary head is continued for exactly10000steps, exporting
only its final checkpoint. Both arms share the original initialization,
base/environment features, seed741, main128-example sampler, weighted BCE,
optimizer/schedule and the same512training-only PTQ representatives. The
unchanged trainer/helpers and all input indexes are hash-bound. Full main and
auxiliary batch digests must match after both arms finish.

Treatment adds a six-class softmax CE with weight.2from the final shared64-unit
ReLU; control weight is0. The auxiliary output is discarded on export: the
runtime input and single sigmoid output/operators remain unchanged. The main
binary labels and negative sampling remain unchanged. A separate64-example
training sampler balances class then TTS engine for target and exact standalone
`Hey Nick`, `Hey Next`, `Hey Nice`, `Hey Mix`, `Hey Nexus`. Other negatives
remain in the main loss. This is text supervision, not phonetic certification.

Preparation uses only original base variants0/1 and verifies source-role,
source/audio hashes, held-out source exclusion and warmed feature bounds.
Positive auxiliary windows use original keyword markers; exact negative short
phrases use the last original10ms RMS frame above max(peak*.01,1), plus the
original2.4s lead. Auxiliary end sampling is.03.. .32s after that anchor. No
waveform, speed, feature payload, binary-training marker or runtime behavior
changes. These RMS anchors are not phonetic truth. The private manifest has
10242views from5072unique synthetic source recordings, not5072independent
speakers; its public aggregate includes hashes and counts per class/engine.

```sh
python research/wakeword_p4/embedding/lexical_auxiliary.py prepare \
  --data-root "$WAKE_DATA" --metadata "$ORIGINAL_FEATURES_JSON" \
  --plan research/wakeword_p4/embedding/plans/lexical-auxiliary-preparation-v1.json \
  --output "$NEW_PRIVATE_AUXILIARY_MANIFEST"
python research/wakeword_p4/embedding/lexical_auxiliary.py train \
  --data-root "$WAKE_DATA" \
  --plan "$FROZEN_CONTROL_OR_TREATMENT_PLAN" \
  --initial-weights "$WAKE_DATA/head-environment-expanded-v2/step-10000/head.keras" \
  --auxiliary-manifest "$NEW_PRIVATE_AUXILIARY_MANIFEST" \
  --output "$NEW_PRIVATE_TRAINING_OUTPUT"
python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --training "$NEW_PRIVATE_TRAINING_OUTPUT" \
  --plan "$FROZEN_CONTROL_OR_TREATMENT_PLAN"
```

Keep the complete fixed validation grid and original strict calibration.
Additionally compare the best macro proxy-window hit fraction satisfying both
continuous upper95%<=.5/h and worst confusing-corpus false fraction<=.01, or
report no such point. Endpoint qualifications remain: these synthetic/proxy
counts do not prove real-user recall or phonetic latency. Inferior outcomes
stop before physical/final scoring. No speaker playback, microphone access,
production audio/AFE/FIFO/GUI changes or dining-room board operation.

Both frozen10000-step runs now complete (control30.57s, treatment32.76s).
The full main-batch digest is
`f2c3c405646eb45213fc932eaf4bc24e9b7ff95cdef65c53d4b386601c3ad733`
in both arms; auxiliary batches and512PTQ examples also match exactly.
The independently inspected exports both have102849parameters,114464bytes,
the same31operators, a single int8 probability output and no auxiliary
softmax. The390auxiliary parameters exist only in training. Model SHA256 is
`586a4d8f1c81faed377a86f8aa32aea14dc0c4ff0260e4141ccbf5bdad4eaf5f`
for control and
`f908c50b7d8e2f7c5daedb5f4f9ad58b83478c2f61d643b4cf5537fa205d4e0f`
for treatment. This verifies graph pruning, not P4 numeric/resource parity.

| Fixed joint validation constraint | Control, weight0 | Auxiliary, weight.2 |
| --- | --- | --- |
| Selected threshold | .99 | .98 |
| Mean proxy-window hit fraction across5TTS engines | 40.4474% | 53.0863% |
| Pooled proxy-window hits /1196variants | 23.0769% | 42.8094% |
| Worst confusing-corpus clip false fraction | .6250% | .7143% |
| Continuous events /source distribution | 0 /7.303778h | 0 /7.303778h |
| One-sided95% event-rate upper bound | .410162/h | .410162/h |

The auxiliary treatment improves the prespecified joint-constraint macro
metric by12.6389percentage points in this matched run. Its per-engine values
are51.7857/66.6667/65.6250/41.6667/39.6875% for Edge/Kokoro/macOS/OpenAI/Piper.
That is useful experimental evidence, but coverage is still too low to
replace Hi ESP. These are reused synthetic/proxy metrics, not native acoustic
recall or independent acceptance. This setting does not advance to a new
physical or reserved-final benchmark. The original board firmware stays active.

Training-representative float/INT8 maximum absolute errors are.232631control
and.047086treatment (means.004607/.002349). Preserve those outliers; do not
attribute recognition changes solely to quantization from this observation.
The quality comparison above uses actual exported INT8 models throughout.

`compare_lexical_auxiliary.py` reproduces the matched-input/export checks and
both frozen constraint selectors, retaining every original threshold row.
Run it with `--control`, `--treatment` and a fresh `--output`. The result is
[`lexical-auxiliary-validation-v1.json`](../results/lexical-auxiliary-validation-v1.json).
Five focused tests cover exact-negative classes, alias preservation, source
tail anchors, deterministic byte-preserving samples, source-role/bounds
failures and simultaneous confidence/near-word constraints. The full goal
remains active and unachieved.

### Complete-word binary supervision, frozen follow-up

`plans/lexical-binary-{control,treatment}-v1.json` keep the preceding lexical
CE weight.2and the exact same main/auxiliary/PTQ batches, original initialization,
seed741,10000steps and final-only checkpoint. Treatment additionally applies
the **existing** weighted binary loss to those64auxiliary histories, with
weight.2. Class0is the unchanged positive label; classes1..5are the unchanged
exact-confuser negatives. Control adds no auxiliary binary loss. The intent
is to train the final sigmoid on complete words as well as their shared
representation, with no new runtime branch or microphone/audio change.

The six-class manifest remains pinned to the previous source identity;
recreate that preparation using commit`36b27da` before using the newer trainer.
No source record, RMS anchor, feature array or main sampler changes. The
new control must independently reproduce the previous lexical-treatment model
bytes and all recorded main/auxiliary/PTQ hashes; keep any mismatch as a
failed replication rather than assuming topology implies equivalence.

Use the same `lexical_auxiliary.py train` and `evaluate_head.py` commands with
the new plans and fresh private outputs. Compare with
`compare_lexical_auxiliary.py --difference-field auxiliary_binary_loss_weight`,
plus `--control`, `--treatment`, and a fresh `--output`. The same full threshold
grid, joint confidence/confuser constraints and endpoint limitations apply.
The sixth focused test checks actual loss-gradient polarity and the original
negative weight across all six auxiliary classes. Historical interface-pilot
success is explicitly not a new model fidelity pass. No physical/reserved
benchmark or production replacement follows from this preparation alone.

Both frozen runs complete, with exact replication of the previous lexical
treatment in the new zero-added-binary-loss control. The final model bytes,
all main/auxiliary batches, PTQ representatives, source inputs and initial
weights match. See
[`lexical-binary-replication-v1.json`](../results/lexical-binary-replication-v1.json).
New treatment SHA256 is
`a2c36dcefc07d1bd2e1ba25e5e871d4ceb540064d3c36c844b027ee4ec045df5`.

The added complete-word binary loss **regresses** the fixed joint-constraint
macro proxy metric from53.0863% at.98to49.4246% at.95. Pooled hits fall from
512/1196to368/1196; worst confusing-corpus false fractions are.7143%/.6250%.
Both have0events /7.303778h, upper95%.410162/h. Retain the entire fixed grids
and reject this setting without new acoustic or reserved-final scoring. See
[`lexical-binary-validation-v1.json`](../results/lexical-binary-validation-v1.json).

`audit_head_events.py` separately replays all retained score streams using
the existing25step refractory and half-threshold rearm rule. No model, label,
threshold or original proxy window changes. All continuous counters reproduce;
three focused tests verify plateau/rearm behavior, agreement with the existing
event counter and invalid-score rejection. Existing production code is only
read as a policy reference, not changed or executed by this replay.

| Positive variants, all1196retained | Control | Added binary loss |
| --- | --- | --- |
| Original proxy-window peaks | 512 | 368 |
| Variants with any latched event anywhere | 515 | 369 |
| Latched event in the original proxy window | 507 | 361 |
| Event before the original proxy window | 8 | 8 |
| Peak in proxy window without an event there | 5 | 7 |

The peak/event discrepancies are all OpenAI synthetic sources. Early events
are preserved; uncertain original markers do not prove those early events are
false wakes. Keep unfiltered traces and window membership separate, rather
than shifting windows to improve counts. No event-based interpretation makes
this inferior recipe a usable replacement. Per-source private traces are
hash-bound by the public aggregate
[`lexical-binary-native-events-v1.json`](../results/lexical-binary-native-events-v1.json).
These are cached policy replays, not P4/acoustic timing or real-user recall.

### Export-only lexical probability gate

`plans/lexical-product-v1.json` freezes an inference-graph experiment using
the original lexical-CE treatment weights, not the inferior extra-binary-loss
weights. No model fitting or source/threshold-grid selection is added. The
single shared trunk feeds the original binary sigmoid and the existing
six-class classifier; final probability is binary times lexical target-class0
posterior. The390classifier parameters now remain at inference, with new
softmax/slice/product operators. Their MCU cost and support require separate
verification; this is not a claim of identical runtime resources.

`compose_lexical_gate.py build` checks every source artifact identity, exact
shared layer types/weights and parameter count. All512original training-only
representatives must reproduce the separately computed float product within
1e-6. The source's10000step recipe is provenance; this composition runs zero
new training steps. Actual output quantization is recorded rather than assumed
to match a standalone sigmoid. Original PTQ arrays and validation grid stay
fixed. Product<=binary in pointwise float arithmetic does not prove fewer
latched events; inspect complete continuous counters and policy replay.

```sh
python research/wakeword_p4/embedding/compose_lexical_gate.py build \
  --source "$ORIGINAL_LEXICAL_TREATMENT" \
  --plan research/wakeword_p4/embedding/plans/lexical-product-v1.json \
  --output "$NEW_PRIVATE_PRODUCT_OUTPUT"
python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --training "$NEW_PRIVATE_PRODUCT_OUTPUT" \
  --plan research/wakeword_p4/embedding/plans/lexical-product-v1.json
python research/wakeword_p4/embedding/compose_lexical_gate.py compare \
  --source "$ORIGINAL_LEXICAL_TREATMENT" --output "$NEW_PRIVATE_PRODUCT_OUTPUT" \
  --report "$FRESH_PRIVATE_COMPARISON"
```

Retain the entire confidence/confuser tradeoff and original source result;
No reserved-final or physical readiness is established by export checks alone.
Source speed, original audio producer/AEC/AGC/FIFO and dining-room board stay
untouched. No microphone or speaker access is used by this experiment.

The frozen export passes every composition check:512float identities have
zero error, one shared trunk has103239parameters, and model size increases
from114464to116392bytes (+1928). Source weights are unchanged; new training
steps are0. The int8 export SHA256 is
`c61af87ec5a4edb57f63ed43a09b756e7d445061517140d5938bd5381fb1297c`.
Its output scale is.0038863569498062134with zero point-128, **not**1/256.
Measured64representative float/INT8 mean/max errors are.005524/.057925.
New graph operators include SOFTMAX and STRIDED_SLICE; no MCU support or
whole-app resource acceptance is implied. See
[`lexical-product-build-v1.json`](../results/lexical-product-build-v1.json).

The actual int8 validation **rejects this frozen gate**: under the same joint
confidence/confuser constraint, source-average proxy hits fall from53.0863%
at.98to39.0952% at.9. Pooled hits are285/1196versus512/1196. Worst confusing
corpus remains.7143%; continuous count is0 /7.303778h, upper95%.410162/h.
Even the confidence-only diagnostic drops from69.7312% to59.4058%. Retain all
threshold rows in
[`lexical-product-validation-v1.json`](../results/lexical-product-validation-v1.json).
Do not advance this inferior setting to physical or reserved-final scoring.

Latched replay counts288positive variants with an event anywhere,280with an
event in the original proxy window, and8with earlier events. Five OpenAI
variants have a peak in that window without a corresponding latched event
there; their earlier events stay recorded. Continuous native counters match0.
Full private traces are retained and hash-bound by
[`lexical-product-native-events-v1.json`](../results/lexical-product-native-events-v1.json).
This rejects one fixed gate, not all possible lexical verification designs,
and does not establish intrinsic source pronunciation or embedding quality.
The best retained matched lexical setting remains the preceding CE-only
model; none has achieved Hi ESP parity or full-system acceptance.

### Training-only lexical transfer probe

`plans/lexical-transfer-probe-v1.json` freezes a diagnostic before scoring.
It asks whether the cached INT8 embedding histories permit a simple six-class
decoder to distinguish the same lexical classes across synthesis engines.
No existing trained head is loaded: for each engine, a fixed ridge decoder
fits only the other five engines. Class/engine groups receive equal total
weight; all standardization and intercepts use fit rows only. The penalty is
fixed at1, without a search. The two predetermined inputs are the full16x96
history and its96-dimensional temporal mean. A source-level within-engine
label permutation provides a negative control for both representations.

Every auxiliary source remains training-only. Select original variant0,
deduplicate original PCM hashes before splitting, and retain one deterministic
history at the midpoint of the existing auxiliary endpoint bounds. Those
synthesis/energy anchors remain uncertain; no new source alignment is inferred.
All per-source predictions and all fold confusion matrices are retained.
Three focused tests check invariance under group replication, missing-group
rejection and source-alias deduplication/held-out-overlap rejection.

```sh
python research/wakeword_p4/embedding/probe_lexical_transfer.py \
  --data-root "$WAKE_DATA" --manifest "$ORIGINAL_AUXILIARY_MANIFEST" \
  --plan research/wakeword_p4/embedding/plans/lexical-transfer-probe-v1.json \
  --output "$FRESH_PRIVATE_DIAGNOSTIC"
```

This is closed-class word-window analysis, not streaming wake detection.
Its top1 fractions cannot be compared to wake recall or FAR, and poor ridge
transfer cannot prove an intrinsic information limit or incorrect pronunciation.
There is no detector threshold, validation/final inference, waveform change,
speaker playback, microphone access or hardware operation in this diagnostic.
Do not advance a probe as a deployment candidate or duplicate GPU dialogue/
AMI/calibration experiments using these different80ms features.

The frozen probe completes all24fold/representation/label runs on5072unique
source recordings. Full-history and temporal-mean six-class macro top1 are
83.9386%/84.8461%, versus15.3693%/16.0108% for their label-permutation controls.
Those averages conceal a substantial **target-class transfer deficit**:

| Excluded engine | Full-history target correct/total | Temporal-mean target correct/total |
| --- | --- | --- |
| Edge | 187/256 | 180/256 |
| Kokoro | 41/60 | 21/60 |
| macOS | 73/87 | 48/87 |
| OpenAI | 113/162 | 115/162 |
| Piper | 74/3000 | 265/3000 |
| Qwen | 90/96 | 69/96 |

Equally averaging target-class fractions across engines gives65.2097% for
full history and52.0302% for the temporal mean. Of the excluded Piper targets,
2536/2593respectively are assigned Hey Nick. Neither summary is streaming wake
recall or an independent real-speaker result. Keep the full fit/excluded-engine
confusion matrices and source hashes in
[`lexical-transfer-probe-v1.json`](../results/lexical-transfer-probe-v1.json).
The simple probes find some cross-engine lexical structure, but neither
establishes adequate target transfer. The time mean's slightly higher six-class
average does not justify preferring it for wake detection. This result does
not isolate source pronunciation, endpoint, data-coverage or representation
causes; further experiments must distinguish them without adjusting endpoints
or thresholds to these diagnostic predictions. No model or board was changed.

### Auxiliary-only RMS-anchor ablation

The earlier independent marker audit finds Piper standalone markers typically
precede the source RMS end, whereas most other engines follow it. The original
auxiliary sampler uses those positive markers but source-RMS anchors for exact
short negatives. `plans/lexical-rms-preparation-v1.json` therefore freezes one
specific training-metadata test before preparing or fitting a candidate:
apply the same existing source-RMS policy to all1959standalone target sources.
Preserve commanded positives because full-file activity cannot locate their
wake prefix. This generic text-based eligibility rule does not select engines
or source clips from classifier predictions. No waveform, original main label,
main sampler or validation endpoint changes.

`lexical_rms_alignment.py prepare` verifies the original auxiliary manifest,
source metadata/jobs/index and each eligible source WAV. It preserves every
record field/order except eligible minimum/maximum auxiliary ends. After
preparation, freeze the actual manifest hash in a treatment plan. Both arms
use the retained CE.2/binary-extra0 recipe, same10000step budget and final
checkpoint, initialization, main/PTQ draws and original19point validation grid.
The only plan difference is the auxiliary-manifest hash. The helper now also
hashes the actual auxiliary source/class draws, independent of changed history
bytes. This instrumentation must preserve the retained control's complete
main/auxiliary batch digests and exported model bytes; verify these explicitly.

```sh
python research/wakeword_p4/embedding/lexical_rms_alignment.py prepare \
  --data-root "$WAKE_DATA" --manifest "$ORIGINAL_AUXILIARY_MANIFEST" \
  --metadata "$ORIGINAL_SOURCE_METADATA" \
  --plan research/wakeword_p4/embedding/plans/lexical-rms-preparation-v1.json \
  --output "$NEW_PRIVATE_ALIGNED_MANIFEST" --report "$NEW_PRIVATE_PREPARATION_REPORT"
```

RMS is an uncertain auxiliary anchor, not phonetic truth or a runtime endpoint
detector. Keep the same joint confidence/confuser constraints and every grid
row; an inferior setting gets no acoustic/final test. Preparation, training
and validation use no speaker, microphone or board. Historical frozen helper
hashes require their original source commits for reproduction; do not rewrite
prior plans to accommodate the added training-metadata instrumentation.

Preparation at `7d013d3` verifies 1959 standalone target source recordings.
3846 feature views from 1879 sources change integer bounds; all other record
fields, negatives and commanded positives remain identical. See
[`lexical-rms-preparation-v1.json`](../results/lexical-rms-preparation-v1.json).
Actual manifest SHA256 is
`8c0887603646990be98b1ada9a2dcaf3e784b71ef5c44eea089d18ef03f53c0c`.
The two training plans freeze before fitting at `869d1e4`.

For each arm, set `RMS_ARM_PLAN` to its control/treatment plan and
`RMS_ARM_MANIFEST` to the corresponding original/prepared private manifest.
Use fresh output directories, the existing initial weights and the same
private feature root:

```sh
python research/wakeword_p4/embedding/lexical_auxiliary.py train \
  --data-root "$WAKE_DATA" --initial-weights "$ORIGINAL_INITIAL_WEIGHTS" \
  --auxiliary-manifest "$RMS_ARM_MANIFEST" --plan "$RMS_ARM_PLAN" \
  --output "$NEW_PRIVATE_ARM_OUTPUT"
python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --training "$NEW_PRIVATE_ARM_OUTPUT" \
  --plan "$RMS_ARM_PLAN"
python research/wakeword_p4/embedding/lexical_rms_alignment.py compare \
  --control "$NEW_PRIVATE_CONTROL" --treatment "$NEW_PRIVATE_TREATMENT" \
  --reference-complete "$ORIGINAL_CE_TREATMENT_COMPLETE" \
  --report "$NEW_PRIVATE_COMPARISON_REPORT"
python research/wakeword_p4/embedding/audit_head_events.py \
  --checkpoint "$NEW_PRIVATE_ARM_OUTPUT/step-10000" \
  --output "$NEW_PRIVATE_NATIVE_EVENT_REPORT"
```

Both runs finish. The new control exactly reproduces retained CE-only model
bytes and both full main/auxiliary batch digests. Actual main draws, PTQ arrays
and auxiliary source/class draw digests also match across arms, while auxiliary
history bytes change as intended. Source/class draw SHA256 is
`ffa458c69815ce1b7eef25fecb036bfca03f2beacd3e5b4bb1b66fa8934e8a1d`.
Thirteen lexical/probe tests pass, including original source-role guards,
command preservation, bound rejection and source/class sequence invariance.

The fixed validation **rejects this auxiliary-only RMS setting**. Joint-qualified
macro proxy-window hits decrease from 53.0863% to 52.6161%; pooled hits fall
from 512 to 497/1196. Both select 0.98, with worst confusing-corpus false
fraction 0.7143% and 0 events over the same 7.303778 h (upper 95% 0.410162/h).
Treatment model SHA256 is
`b94dc1f6f4d47dfbf1b84b6532b13ff73de99ec5438b4c82c62ed225a7e12f66`.
The exported graph retains 114464 bytes, 31 operators and the original external
INT8 interface. Same graph shape is not actual MCU numeric/resource proof.
See every fixed-grid row and comparison check in
[`lexical-rms-validation-v1.json`](../results/lexical-rms-validation-v1.json).

Latched replay has 515/501 positive variants with any event and 507/493 with an
event in the unchanged proxy window, control/treatment respectively. Both
retain 8 earlier-event variants; 5/4 peak hits lack an event in that window.
Those discrepancies remain OpenAI synthetic sources and are not relabelled
from uncertain markers. Continuous counters reproduce 0. Full private traces
are hash-bound by
[`lexical-rms-native-events-v1.json`](../results/lexical-rms-native-events-v1.json).
No P4, acoustic or reserved-final evaluation follows this inferior candidate.
This tests one auxiliary metadata intervention; it does not rule out every
endpoint effect or explain the broader source-transfer deficit. The retained
CE-only setting and the full Hi ESP parity goal remain unchanged.

### Training-only engine adversary

The cross-engine diagnostic finds a large Piper target-to-Nick transfer
deficit; the preceding auxiliary-anchor harmonization does not improve the
retained detector. `plans/lexical-domain-{control,treatment}-v1.json` therefore
freeze a separate representation hypothesis before training: keep lexical
CE at 0.2 and use the original auxiliary histories to learn an engine classifier.
Its Dense6 head sees the shared final 64-unit ReLU. The head minimizes ordinary
engine CE; only its gradient into the shared representation is reversed at
the fixed treatment scale 0.05. There is no gradient-scale search, extra
sampler, warmup or new audio. Control scale 0 disables the adversary and must
exactly reproduce the retained CE-only detector and full batch hashes.

Every auxiliary class already samples all six synthesis engines. Class,
engine, source, window and main/PTQ draws stay identical between arms. Engine
IDs are bound to each actual sampled source, and their full batch digest is
retained. The extra classifier has 390 training-only parameters, with Glorot
seed 743. Its normal classifier gradient and reversed feature gradient are
checked independently; the original source/role/bounds and control tests also
pass (16 lexical tests total). Both auxiliary branches are removed at binary
export, whose interface/operators/bytes must be checked before comparison.

```sh
python research/wakeword_p4/embedding/lexical_auxiliary.py train \
  --data-root "$WAKE_DATA" --initial-weights "$ORIGINAL_INITIAL_WEIGHTS" \
  --auxiliary-manifest "$ORIGINAL_AUXILIARY_MANIFEST" \
  --plan "$DOMAIN_ARM_PLAN" --output "$NEW_PRIVATE_DOMAIN_ARM"
python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --training "$NEW_PRIVATE_DOMAIN_ARM" \
  --plan "$DOMAIN_ARM_PLAN"
python research/wakeword_p4/embedding/lexical_domain.py \
  --control "$NEW_PRIVATE_DOMAIN_CONTROL" \
  --treatment "$NEW_PRIVATE_DOMAIN_TREATMENT" \
  --reference-complete "$RETAINED_RMS_CONTROL_COMPLETE" \
  --report "$FRESH_PRIVATE_DOMAIN_COMPARISON"
```

For each arm, `DOMAIN_ARM_PLAN` names its frozen control/treatment plan and
the output path is fresh. Preserve the same 10000-step final checkpoint,
19-point validation grid and confidence/confuser constraints. Reducing engine
information is only a hypothesis: synthesis domains are not real speakers,
and confusing an engine classifier does not establish real microphone
generalization. No speaker, microphone, board or reserved-final access occurs
in this experiment. An inferior candidate receives no physical/final test.

Both frozen runs at `2240ffa` complete. The control reproduces the retained
CE-only model bytes and full main/auxiliary batch digests exactly. Main,
auxiliary history, source/class, engine-label and PTQ digests match across arms.
The engine-label batch SHA256 is
`35dbd92458c4483a04ac1e94e24103e993c20f95f230732b6558847ac38aa67c`.
The treatment has 390 extra training-only parameters; its exported binary
graph still has 114464 bytes, 31 operators, input scale 1/128 and output
scale 1/256. No engine/lexical softmax remains in that graph. Treatment SHA256:
`8813c8faa35ee8724a053130eafb1721ffff65353230871beecca793695bc8e5`.

The original validation **rejects this fixed engine adversary**: joint-qualified
macro proxy hits decrease from 53.0863% to 47.5595%, pooled hits from 512 to
450/1196. Both points use 0.98, worst confusing-corpus fraction 0.7143%, and
0 continuous events / 7.303778 h (upper 95% 0.410162/h). Retain every grid row
and full pairing evidence in
[`lexical-domain-validation-v1.json`](../results/lexical-domain-validation-v1.json).
Improved representative float/INT8 fidelity is not recognition improvement.
This one negative result does not rule out all domain-invariance approaches,
prove successful invariance, or identify the cause of the transfer deficit.

Cached latched replay gives 515/454 positive variants with any event and
507/446 with an event inside the original uncertain proxy window. Both arms
retain 8 earlier-event variants; 5/4 proxy-window peak hits lack an event there.
Those discrepancies stay OpenAI synthetic cases, not automatically relabelled
false wakes. Continuous event counters reproduce 0. The full private traces
are hash-bound by
[`lexical-domain-native-events-v1.json`](../results/lexical-domain-native-events-v1.json).
No P4, acoustic or reserved-final scoring follows this inferior setting.
The retained lexical reference remains unchanged and is not accepted as
Hi ESP parity; the stock board firmware and audio chain were not touched.

### Quarter-TTS source-activity sampling

`plans/negative-activity-preparation-v1.json` freezes a different sampling
hypothesis before source-span measurement or fitting. The main sampler's
TTS negatives currently choose a uniform end over the entire warmed feature
sequence. For exactly 12 of those 48 draws per batch, keep the original
chosen source but draw a new end from its broad original source-energy span.
Consume every original RNG choice first, then use a separate seed 3000741.
The 32 positives, 48 ambient negatives, other 36 TTS histories, all source/
class draws, original auxiliary batches and PTQ arrays must remain identical.
This differs from teacher mining, which also replaced source selections and
ambient draws. Neither a new loss nor a runtime VAD is introduced.

`negative_activity.py` checks all original training-negative source WAVs,
feature/job/index identities and held-out/positive source exclusion. Use the
first/last original 10 ms RMS block at least max(peak*0.01,1), plus the existing
2.4 s lead. Eligible end positions run from 30 ms after activity start to
320 ms after its end, on the original 80 ms grid. Reject silent/invalid or
out-of-warm-range sources rather than filtering them silently. The manifest
must cover every original/environmental TTS-negative view. All source PCM,
features and labels remain unchanged.

Endpoint membership in this span is **not speech occupancy**: internal gaps
can be silent and histories ending later can still retain the spoken word.
Its theoretical and sampled fractions diagnose the sampler, not missing
phonetic information or source pronunciation. No validation/final scores,
speaker, microphone or board are used by preparation. After preparation,
freeze the actual manifest SHA in both off/on training plans before fitting.

```sh
python research/wakeword_p4/embedding/negative_activity.py \
  --data-root "$WAKE_DATA" \
  --plan research/wakeword_p4/embedding/plans/negative-activity-preparation-v1.json \
  --output "$FRESH_PRIVATE_ACTIVITY_MANIFEST" \
  --report "$FRESH_PRIVATE_ACTIVITY_PREPARATION"
```

The retained CE-only recipe, original auxiliary bounds, 10000-step final
checkpoint, threshold grid and validation windows stay fixed. Keep both joint
and confidence-only diagnostics; the extra 1% confuser rule is a research
constraint, not a demonstrated Hi ESP equivalence definition. The off control
must reproduce the retained detector and full main/auxiliary batch hashes.
Verify the actual all-source hash and the remaining 116 histories per batch
after the original shuffle. Twenty-five focused tests pass, including source/
role/coverage/bound guards and preservation of the original RNG state, labels,
source choices and every unscheduled history across the sampling change.

V1 preparation at `6bf333f` **fails before training**: no 10 ms block meets
the fixed RMS floor in two original OpenAI-TTS training negatives. The audit
checks all 6353 original negative WAV hashes and all 25440 feature views;
neither affected 0.3 s WAV is all-zero (sample peak1, peak block RMS below1).
They account for eight original/environmental views. No sources, labels or
waveforms are removed or changed. Public evidence is
[`negative-activity-source-audit-v1.json`](../results/negative-activity-source-audit-v1.json)
and [`negative-activity-preparation-failure-v1.json`](../results/negative-activity-preparation-failure-v1.json).
The original preparation helper/plan are unchanged; no training or validation
result is attributed to this failed v1 preparation.

`plans/negative-activity-preparation-v2.json` explicitly binds that audit and
failure before further preparation. `negative_activity_fallback.py` retains
the **exact original uniform endpoint**, without consuming the separate
activity RNG, whenever original source energy is undefined. This applies to
every environmental view too. All other scheduled views use the original v1
interval rule; full source/view identity coverage is required. Undefined cases
are counted separately, with no claimed energy-span membership. Both arms
also hash every fallback history to prove preservation. Twenty-nine focused
tests pass, including mixed/undefined activity, full original ranges and RNG
preservation. This explicit exception does not loosen the RMS rule, clamp
invalid intervals or filter sources. No detector scores motivated it.

```sh
python research/wakeword_p4/embedding/negative_activity_fallback.py \
  --data-root "$WAKE_DATA" \
  --plan research/wakeword_p4/embedding/plans/negative-activity-preparation-v2.json \
  --audit "$FROZEN_PRIVATE_TRAINING_SOURCE_AUDIT" \
  --output "$FRESH_PRIVATE_ACTIVITY_MANIFEST_V2" \
  --report "$FRESH_PRIVATE_ACTIVITY_PREPARATION_V2"
```

Preparation at `67c09b0` succeeds with every original source/view retained.
The resulting manifest SHA256 is
`70efdcbf8f72c6171463a19712ea4013bd8e605e595dad51626d11a96fd5be5b`.
See [`negative-activity-preparation-v2.json`](../results/negative-activity-preparation-v2.json).
The new `negative-activity-{control,treatment}-v2.json` plans bind this exact
manifest, the current helper bytes and the retained domain-control completion
before either run. They differ only in `focus_negative_activity`. Use
`lexical_auxiliary.py train` with the original auxiliary manifest and extra
`--activity-manifest` argument for both arms, then the original
`evaluate_head.py` and `compare_negative_activity.py`. Historical plans keep
their original source hashes; they are not updated to the current helpers.
