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
