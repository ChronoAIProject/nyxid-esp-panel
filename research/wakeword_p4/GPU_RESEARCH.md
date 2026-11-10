# Hey Nyx GPU continuation

Imported from `origin/research/macos-custom-wakeword` on 2026-11-10.
Source commit: `b59c05e0dcea3fc28b2574b1bbdf301c7a64baa6`.
Historical Mac/P4 reports remain historical evidence, not results from this host.
The unrelated `research/voice` random-array classifier only verifies GPU setup.

## Acceptance gates

- Rebuild AI-generated speech from pinned local engines; preserve complete
  voice/source partitions before augmentation and retain source WAV hashes.
- Train on RTX 4090 using actual waveform-derived 10 ms MicroFrontend features.
  Keep the previous rejected candidates as baselines, not production models.
- Select checkpoints and detection thresholds using validation only; freeze the
  final test plan before scoring the final model. Report per-engine recall,
  continuous false events/hour and wake-end-to-detection delay distributions.
- Validate actual quantized streaming inference, including reset/rearm behavior.
  Do not substitute float window classification for exported streaming results.
- Compare Hi ESP and Hey Nyx using matched phrase/acoustic conditions and the
  same negative audio at matched false-wake working points. Positive phrases
  necessarily differ. Report exposure and uncertainty for both engines.
- Require the one-sided 95% false-event/hour upper bound <= 0.5. With zero
  events this needs at least -log(0.05)/0.5 = 5.992 hours of held-out exposure;
  reused clips/augmentation are not independent original recordings.
- If WakeNet9 cannot run without its supported target, leave its comparison
  unverified and continue independent work. Board-free training is feasible;
  firmware resource/acoustic acceptance remains a separate future gate.

## First reconstruction

Install `kokoro-onnx==0.4.9 scipy soundfile pymicro-features` into Python 3.11.
Use `training/kokoro_samples.py` with the hash-checked v1.1 release assets.
The generator assigns whole English voice IDs to train/validation/test before
synthesis. Output is 16 kHz mono PCM16. These are AI-generated voices, not
recordings of independent humans. Its existing test split was used in prior
research; treat re-created scores as regression, never a fresh holdout.

`training/generate.py --device cuda` now selects NVIDIA for the pinned Piper
model. Its original Mac path remains available. On this Linux/4090 host it
generated 13,920 clips (6,000 positives + 6,000 negatives for training; 480 of
each class for each validation/regression split). Whole source speaker IDs
are disjoint; `results/4090-piper-manifest-audit.json` records the completed
audio/hash audit. Speaker blends are not new independent speakers.

This run uses SciPy polyphase resampling (320/441) instead of torchaudio;
waveforms are not claimed equal to the previous Mac run. Apply
`piper-lazy-torchaudio.patch` to the pinned generator before importing it on
this host. The research entry point calls the low-level CUDA generation API,
not the upstream file-generation API that still requires torchaudio. Exact
source hashes and GPU/runtime details are in `4090-piper-generation.json`.
These reports establish corpus reconstruction, not recognition quality.

Stage results are committed and pushed to `research-hye-nyx-voice-gpu` after
verification. Local weights/audio remain ignored; reports record hashes and
commands. Pending or failed evaluation must remain explicit in published notes.

## Linux GPU training reproduction

The observed environment is pinned in `requirements-linux-gpu.lock` (Python
3.11, TensorFlow 2.21.0, PyTorch 2.15.0). TensorFlow uses CUDA 12 libraries;
PyTorch uses CUDA 13. Do not import both runtimes into the same training process.
`training/gpu_python.py` supplies pip-installed CUDA 12 library paths locally.
GPU discovery and actual matrix execution on `/device:GPU:0` passed on 4090.

After generating the Piper/Kokoro sources, downloading/hash-checking the
LibriSpeech dev-clean archive in `results/4090-data-provenance.json`, and
extracting it into `.local/wakeword/LibriSpeech/dev-clean`, run from repository root:

```sh
python3 research/wakeword_p4/training/features.py --dataset .local/wakeword/hey-nyx/kokoro \
  --corpus .local/wakeword/LibriSpeech/dev-clean \
  --output .local/wakeword/human-features --workers 12
```

Run from repository root for the subsequent commands:

```sh
python3 research/wakeword_p4/training/stream_data.py \
  --source .local/wakeword/hey-nyx/piper-gpu --source .local/wakeword/hey-nyx/kokoro \
  --backgrounds .local/wakeword/backgrounds --human-features .local/wakeword/human-features \
  --output .local/wakeword/features-gpu-v1 --workers 16
python3 research/wakeword_p4/training/audit_features.py \
  --features .local/wakeword/features-gpu-v1 --output /tmp/features-audit.json
python3 research/wakeword_p4/training/gpu_python.py \
  research/wakeword_p4/training/train_stream.py --device cuda \
  --upstream .local/wakeword/microWakeWord --features .local/wakeword/features-gpu-v1 \
  --output .local/wakeword/model-gpu-v1 --architecture official64 \
  --steps 40000 --eval-interval 10000
```

The 58,256 rendered sequences are correlated variants of 14,564 source clips,
not independent trials. Source exposure for human negatives excludes the
inserted 200 ms utterance-boundary silence. Float checkpoint selection is
validation-only; the exported INT8 model requires separate validation
calibration before test/regression scoring. Event recall includes latch and
refractory behavior, and reports latency relative to synthetic keyword-end
annotations. Run `python3 research/wakeword_p4/training/test_events.py` for the
premature-wake, refractory and continuously-high-score regression scenarios.

The rebuilt test voices and dev-clean data are **regression** sets because
prior experiments used them. A fresh final positive holdout and >=5.992 hours
of independent-source negative exposure still need to be frozen. No candidate
has yet passed the goal's final acceptance gates.

## Executed candidate v1: rejected

RTX 4090 training completed 40,000 steps in 368.7 seconds including export.
Validation selected step 30,000; the INT8 model is 63,032 bytes, SHA-256
`06a05930d1d8a36f6c6de535ebb14781c270204587a50aa431ce992606365383`.
Quantized validation froze threshold 0.992 before regression scoring. No
threshold in the grid met both validation constraints; this is a reported
fallback, not a viable operating point. See `results/4090-training-v1.json`,
`4090-threshold-v1.json` and `4090-quantized-v1.json`.

| INT8 regression measurement | Result |
| --- | ---: |
| Mean recall across two TTS sources | 57.32% |
| Piper recall | 87.55% |
| Kokoro recall | 27.08% |
| Pooled recall (dominated by Piper) | 86.08% (1,694/1,968 variants) |
| Confusing-negative clip false detections | 0.714% |
| Continuous source audio exposure | 1.3582 hours |
| Continuous false events | 1 |
| Observed events/hour | 0.7363 |
| One-sided 95% upper bound | 3.4929/hour |
| Detection latency among hits, median / p95 | 120 / 200 ms |

These are synthetic/regression measurements, not real-user acceptance.
Correlated render variants do not establish independent trial counts.
The latency annotation is approximate and includes no capture/network delay.
Historical Mac voice fixtures were unavailable here and are **unmeasured**.
Hi ESP comparison remains unverified; see `4090-hiesp-comparison-status.json`.

The float-vs-INT8 validation diagnostic (`4090-streaming-v1-comparison.json`)
passed repeated-reset reproducibility on eight clips. Mean absolute score
drift was 0.000564, maximum 0.06168; one Kokoro event shifted by one 30 ms step.
This does not establish whole-corpus equivalence or target-device numerics.
Poor float-window vs quantized-streaming recall must not be attributed to
quantization alone without isolating context, state and scoring effects.

Next experiments expand Kokoro speed coverage with the same whole-voice
partitions. The first fractional-speed attempt failed before generating audio
because the library truncated speed 0.75 to integer zero; the corrected
adapter preserves the caller's fractional speed. This failure and its scope
are retained in `4090-kokoro-speed-failure.json`. New generation/training
remains in progress. More correlated speed variants are not new speakers.

## Context diagnosis and reserved negative exposure

The expanded diagnostic covers all 48 Kokoro positive validation variants at
the already frozen 0.992 threshold. Float streaming detects 21, INT8 detects
22; thus quantization alone does not explain the earlier float-window recall.
The valid kernel-5, stride-3 first convolution consumes 179 of 180 input
frames. Original phase-0 windows are 10 ms out of alignment with the stream.
Their maximum probability difference reaches 0.94655. Shifting the window by
one feature frame reduces the maximum difference to **6.56e-7** without
changing weights, audio or labels. The report is
`results/4090-v1-kokoro-context-diagnosis.json`.

`train_stream.py` now derives the window phase from input size/kernel/stride
before validation. Candidate v1b reruns the same 40,000-step, two-source
recipe with aligned validation/checkpoint selection. No quality improvement
is claimed until the actual export is evaluated. This numerical alignment
check covers the inspected architecture and validation data, not every model.

Separately, `training/freeze_negatives.py` verified the official archive MD5s,
screened reference transcripts for literal Hey Nyx variants, checked speaker-ID
isolation and extracted **10.7450 source hours across 73 speakers** from
LibriSpeech test-clean/test-other. The manifest and per-array hashes are in
`results/4090-frozen-negative-manifest.json`. No model scores were computed on
this set. Source hours exclude inserted silence and padding. These are
human negative speech recordings, not real-user positive wake recordings.
At zero events, that exposure would give a Poisson one-sided 95% upper bound
of about 0.279/hour; no zero-event result is claimed. New independent positive
holdout coverage and the matched Hi ESP comparison remain unresolved.

## Candidate v1b and the next corpus

Aligned validation selected step 10,000 of the v1b 40,000-step rerun.
The actual quantized regression result is still rejected: mean source recall
61.64% (Piper 79.53%, Kokoro 43.75%), confusing-negative false detections
1.384%, and zero continuous events in 1.3582 source hours. The one-sided 95%
upper bound is **2.2057/hour**, above the 0.5/hour goal. Detection latency
among hits is median 120 ms, p95 170 ms. Full reports are
`4090-training-v1b-aligned.json`, `4090-threshold-v1b-aligned.json` and
`4090-quantized-v1b-aligned.json`. The model SHA-256 is
`c866e74cc7c782b39b44665b05959c3cc60640d5b4ef29ab3a6ac68791bf989b`.
Training reruns are not claimed bit exact on GPU; different selected weights
and thresholds prevent attributing quality differences to alignment alone.

The completed Kokoro speed expansion has 3,220 WAVs across the same 28
English voice IDs, using speeds 0.75/0.9/1/1.1/1.25. Hash/audio/voice audits
passed. Together with Piper it yields 68,560 rendered sequences from 17,140
source WAVs (`4090-features-v2-audit.json`). Candidate v2 keeps the architecture
and 40,000-step budget, uses equal engine sampling (`--source-exponent 0`)
and latched-event float validation (`--event-validation`) after causal history
warmup. Its final quantized evaluation still includes startup behavior. This
is a multiple-change experiment, not an isolated speed-ablation study.

Before any scores, `comparison_holdout.py` generated 468 paired/common-negative
WAVs using 26 unused international Kokoro presets: 78 Hey Nicks, 78 Hi ESP and
312 common negative phrases. `audit_comparison.py` verified voice/audio
disjointness from development and exact pairing by voice/speed. See
`4090-comparison-holdout-generation.json` and `4090-comparison-holdout-audit.json`.
These are English phrases through international carrier voices, not recordings
of 26 independently established human speakers. No recognition scores have
been computed on this challenge; pronunciation review and matched acoustic
rendering are still pending. It does not establish a Hi ESP comparison.

## Balanced multispeed v2: rejected

The completed 40,000-step RTX 4090 run selected step 10,000 and validation
threshold 0.97. Its INT8 model SHA-256 is
`2a163828374e68860f434332683f34a2dc80a3c3c3de68bd2db172f3a5304ddc`.
Regression mean source recall is **57.604%** (Piper 65.625%, Kokoro 49.583%);
pooled recall is 1,379/2,160 correlated variants (63.843%). Confusing-negative
false detections are 0.7386%. Zero continuous events in 1.35816 source hours
still gives a one-sided 95% upper bound of **2.2057/hour**, failing the goal.
Hit latency is median 130 ms, p95 230 ms; 90 premature positive events were
also recorded. Expanded speed coverage changes the regression population,
so this is not a controlled comparison with earlier candidates.
See `4090-training-v2-balanced.json`, `4090-threshold-v2-balanced.json`, and
`4090-quantized-v2-balanced.json`. The small validation-only float/INT8 check
passed reset reproducibility and matched event steps on eight clips, with
mean absolute score error 0.00137; this does not establish general equivalence
(`4090-streaming-v2-comparison.json`).

## Paired challenge pronunciation correction

Before any wake-classifier scoring, tokenizer inspection found that `Hi ESP.`
renders ESP as a word. The original challenge is retained but withdrawn from
the intended spelled-letter comparison. Protocol 2 explicitly uses
`Hi E S P.` and records phonemes. Its 468 clips passed waveform/hash/pairing
audits; manifest SHA-256 is
`40b4ed9e88a4c8c7d7a1b5d6ec94522c15cfc39a91ed763ffcadd996d2f468eb`.
Independent pinned Whisper tiny.en screening found text agreement on 32/78
Hey Nyx clips, 28/78 Hi ESP clips, and 236/312 negatives. Original Hi ESP
agreement was 0/78. ASR mismatch can reflect ASR error, and agreement is not
phonetic ground truth. No clips were removed or relabeled; acoustic quality
remains unresolved. These reports are consistency screens, not recognition
scores or evidence of Hi ESP parity. Both the corrected challenge and the
10.745-hour independent negative reserve remain unscored by wake classifiers.

Qwen3-TTS VoiceDesign is being prepared as an additional training-only source
at pinned revision `5ecdb67327fd37bb2e042aab12ff7391903235d3` in an isolated
environment. Torch/TorchAudio imports and RTX 4090 detection succeeded; model
download and actual synthesis have not yet been verified. No Qwen quality or
new independent speaker-count claim is made.

## CUDA VoiceDesign source (training only)

`training/qwen_samples.py` completed **360/360** waveform-QA-accepted clips
(72 positives, 288 confusing/common negatives), 430.88 seconds of standalone
audio, from 24 shuffled description conditions. These conditions and seeds
are not established speaker identities and are exclusively training data.
The pinned model's main safetensors SHA-256 is
`391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522`;
all model/config hashes, generation parameters, source/plan/manifest hashes,
and runtime versions are in `4090-qwen-v1-generation.json`.
RTX 4090 BF16/SDPA synthesis took 511.2 seconds, including model loading but
excluding weight hashing and download. The initial run reported a missing
Python header during optional Triton compilation, then continued successfully;
`python3.11-dev` and `sox` were subsequently installed. These successful WAVs
are evidence of CUDA synthesis, not wake recognition quality.

The first three positive smoke clips all passed waveform QA, but independent
Whisper tiny.en text agreement was only 1/3 (two transcribed as Hey Next).
This cannot distinguish ASR errors from synthesis pronunciation errors.
Before examining full-source ASR results or training a classifier, the
training selection policy was declared: use only independent ASR text
agreement, including the audit's explicit Nyx/Nix/Nicks aliases. Preserve
all original WAVs and labels, retain excluded paths/hashes, and do not select
or filter evaluation data. `select_training_asr.py` enforces training-only
input, exact manifest coverage and unchanged waveform/text hashes. Agreement
is not phonetic ground truth and can bias accent coverage.

Reproduction (the environment inherits global Torch; use the Qwen-specific
lock for the actual full environment, not as a replacement for GPU wheel
index configuration):

```bash
virtualenv --system-site-packages .local/wakeword/qwen-env
.local/wakeword/qwen-env/bin/pip install qwen-tts==0.1.1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 USE_TF=0 \
  .local/wakeword/qwen-env/bin/python research/wakeword_p4/training/qwen_samples.py \
  --model .local/wakeword/qwen3-design --source .local/wakeword/qwen-source.json \
  --output .local/wakeword/hey-nyx/qwen-cuda-v1 --profiles 24
```

`stream_data.py --base-features` appends training-only sources while copying
the existing binary feature prefix and evaluation records unchanged. Its
output must pass the normal feature audit and explicit base-prefix equality
checks before training. The prepared source-comparison experiment retains v2's architecture,
40,000-step budget, equal-engine sampling and event validation, and adds only
the screened training source. GPU reruns are not bit-exact, so this remains
an experiment rather than a proven causal attribution. It will first use
`evaluate_stream.py --validation-only`; independent final fixtures remain
unscored until a defensible candidate is frozen.

The completed ASR screen agrees on 30/72 positives and 268/288 negatives.
The training-only selection therefore contains 298 clips; all parent audio
and labels are retained. The resulting v3 feature set passed its audit and
an explicit byte-for-byte comparison of the entire v2 binary prefix and
record list (`4090-v3-base-preservation.json`). Existing validation/regression
features are unchanged. No v3 training has started.

The subsequent primary-source review in `BEST_PRACTICES.md` changes the next
priority: expand realistic room/noise/negative coverage before running the
source-only experiment. Official precomputed negative features advertise
20 ms steps while our frontend uses 10 ms; they have not been imported.
The official-notebook MIT environmental RIR source was downloaded at pinned
revision `b824a1ef2821f112fda0b9cb26e4278c62b425bb`; 270 mono 16 kHz WAVs passed
upstream hash, format, finite-sample and nonzero-energy checks
(`4090-mit-rir-audit.json`). They are prepared for training augmentation and
have not yet been applied to a candidate. This is no new recognition result.

## Measured-room experiment v4: rejected on validation

`append_room_features.py` appended 28,600 training-only renderings to the
unchanged v2 feature set. Both new variants convolve a measured MIT RIR;
one uses colored noise and one a training-source background, at -5 to 10 dB
SNR. All 270 audited responses were sampled. The RIR onset before the first
sample within 20 dB of peak is removed, followed by L2 normalization; the
reverberant speech level is normalized over the source-utterance interval.
This is a combined room/noise intervention, not an isolated RIR experiment.
No new speaker identities or original source WAVs are introduced.

The audit covers 97,160 sequences and 48,549,947 frames, binary SHA-256
`ae5d8fe17542569b2de97b239245c71ad33a33356529a65c951025dd6c3639ad`.
`4090-v4-base-preservation.json` verifies the entire v2 binary prefix and
existing index records byte-for-byte; additions are exclusively training.
A 16-case comparison against commit `6a3b08b` confirms the renderer's
original four variants still produce bit-identical PCM. The first comparison
detected a changed RNG call order in the original colored-noise branch;
that default path was corrected before the passing regression report.

The same official64, 40,000-step, equal-engine, event-validation recipe on
RTX 4090 selected step 40,000. INT8 model SHA-256 is
`54b92ea0c149eab1f617a3068531b8a3886dd3d6df6ec87f110c2436c74669d3`.
At quantized validation threshold 0.99, source-mean event recall is
**49.974%** (Piper 64.531%, Kokoro 35.417%), versus v2 validation's 57.109%.
Pooled recall is 1,324/2,160 correlated variants (61.296%); confusing-negative
false detections are 0.4545%, maximum source 0.875%. Zero continuous events
in 1.20481 source hours still gives a one-sided 95% upper bound of
**2.4865/hour**, not acceptance. Hit latency is median 130 ms, p95 200 ms.
No regression or final challenge scores were computed for this candidate.
The eight-clip float/INT8 diagnostic passed reset reproducibility; mean score
error is 0.00406 and float/quantized keyword hits are 4/3 of four positives.
That small subset does not establish general equivalence. Reports use the
`4090-*-v4-room*` names. Training reruns are not bit-exact, so the observed
decline is not a causal estimate of RIR augmentation's effect.

## Expanded negative source and controlled v5 candidate

MUSAN original audio was obtained from the pinned `corypaik/musan` mirror
revision `76f9882cfa4475efe11508ac9aa32722f84ca5b7`, with every selected parquet
verified against its upstream LFS SHA-256. The official 11 GB archive transfer
was stopped after measuring slow progress; the partial archive remains local,
was not extracted and was not used for training. No equivalence of the whole
mirror to that official archive is asserted.

`prepare_musan.py` decoded the two declared music shards and both noise shards,
then extracted actual 16 kHz waveform features through our common **10 ms**
MicroFrontend. This avoids the incompatible 20 ms precomputed feature source.
The result is 132 music recordings / **8.54676 source hours** and 930 noise
recordings / **6.22631 source hours**. Music selection is the first two shards,
not random genre-balanced sampling. Short clips are repeated to six seconds
for training windows; exposure counts the original source duration only.
The speech category was not imported, and vocal/ambient speech has not been
exhaustively transcribed. Ten upstream README/license mapping files are
retained locally; provenance hashes are in `4090-musan-licenses.json`.
No audio or model weights are redistributed in git.

`attach_training_negatives.py` verified prepared indices, feature arrays and
source WAV hashes and screened exact WAV/decoded-PCM duplication against
existing environment audio. None was found. Re-encoded/cropped source
independence is not proven. The resulting v5 set reuses v4's generated
feature binary and all evaluation records, adding only training ambient
features. Combined continuous negative training exposure is **18.7485 hours**,
up from 3.97543. Hashes/config are frozen in `4090-v5-training-protocol.json`;
the normal feature audit passed. Candidate v5 uses the same architecture,
40,000 steps and source sampling as v4, first evaluating quantized validation.
At this milestone v5 training is in progress; no quality improvement is claimed.

### Completed v5 validation: still not an acceptance candidate

The 40,000-step run completed in 366.1 seconds and selected step 20,000.
INT8 SHA-256 is
`0912d1730aa5bc57fd1e77d82844edd357b401bc0b13ef3a5ca4a86104d7b130`.
At validation-only threshold **0.95**, mean source event recall is **59.010%**
(Piper 75.521%, Kokoro 42.500%), pooled 1,552/2,160 variants (71.852%).
Confusing-negative detections are 0.6534%, maximum source 1%; 17 premature
positive events were recorded. Hit latency is median 120 ms, p95 210 ms.
Zero continuous events in the unchanged 1.20481 validation source hours still
has the same **2.4865/hour** one-sided 95% upper bound. No regression or final
fixture was scored for v5. See `4090-training-v5-negatives.json`,
`4090-threshold-v5-negatives.json` and `4090-validation-v5-negatives.json`.
The eight-clip numeric diagnostic passed explicit reset reproducibility and
both paths detected 3/4 positive clips. Mean absolute score drift is 0.00348,
but the maximum reaches 0.40455; this is not general numerical equivalence.
The full INT8 validation result, rather than this small check, governs recall
(`4090-streaming-v5-negatives-comparison.json`).

On the identical validation population, v2/v4/v5 mean source recalls are
57.109% / 49.974% / 59.010%. This does not establish statistical improvement:
GPU training is not bit-exact, checkpoints and thresholds differ, and source
renderings are correlated. In particular v5's Kokoro recall is still lower
than v2's 46.667%. Broader negative coverage alone has not solved the core
cross-source positive generalization problem. These candidates do not justify
consuming the independent final reserve or claiming Hi ESP parity.

`4090-v5-base-preservation.json` separately verifies reused binary/index files,
unchanged human features and original ambient records, plus 1,062 training-only
ambient additions. The next priority is positive-source/phonetic quality and
training generalization, including a controlled additional-source experiment
and a pretrained-feature desktop diagnostic baseline if needed. That baseline
would be a diagnostic, not evidence of MCU footprint or WakeNet equivalence.

For future runs, `train_stream.py` now records hashes for human and ambient
indices and available augmentation provenance as well as the generated index.
This matters because v4 and v5 intentionally share the generated index but
have different ambient training inputs. The already completed v5 run retains
its original source hash; `4090-v5-training-protocol.json` binds its ambient
and human indices explicitly instead of rewriting historical training output.

## Controlled Qwen addition v6

The screened Qwen training source (30 positive, 268 negative original clips)
was appended to v5, retaining the four original render modes and adding the
same two measured-room modes only where absent. The new
`append_room_features.py --only-missing` option rejects no old data and avoids
regenerating earlier room variants. Auxiliary feature copying now resolves
all paths in human/ambient indices, including the separately attached MUSAN
directories; the older append helper only carried the original two directories.

The result has 98,948 sequences from 17,438 source WAVs; 1,788 new renderings
are exclusively Qwen training data. Full binary-prefix and index comparison
plus unchanged human/ambient indices are recorded in
`4090-v6-base-preservation.json`. Negative exposure stays 18.7485 training
hours. The audit and RIR configuration are in `4090-features-v6-qwen-audit.json`
and `4090-v6-qwen-augmentation.json`.

The same 40,000-step official64 recipe selected step 10,000 (364.7 seconds
on RTX 4090). Equal engine sampling now gives each of three engines one third
of positive and confusing-negative draws; this heavily oversamples the small
Qwen source. Description conditions are not physical speaker identities.
INT8 SHA-256 is
`324f0097d642bd30ca4e3c9d53d847d3bd90772833e125fe89882848ad6469cd`.
At validation threshold **0.8**, mean source recall is **69.948%**, Piper
76.979%, Kokoro 62.917%; pooled recall 1,629/2,160 (75.417%). Confusing-negative
false detections are 0.7102%, maximum source 0.75%. Three premature positive
events occurred. Hit latency is median 130 ms, p95 210 ms. Zero continuous
events in unchanged 1.20481 validation hours still has a **2.4865/hour**
one-sided 95% upper bound. This is not acceptance or Hi ESP equivalence.
See `4090-training-v6-qwen.json`, `4090-threshold-v6-qwen.json` and
`4090-validation-v6-qwen.json`.

The small eight-clip float/INT8 diagnostic reproduces after reset, with 4/4
positive hits on both paths, mean score drift 0.00130 and maximum 0.07032.
It is not general numerical equivalence. Compared with v5's 59.010% on the
same validation population, the v6 result warrants a development-regression
check. Single GPU reruns, changed source probabilities and correlated variants
prevent a causal or statistical improvement claim. The independent positive
challenge and 10.745-hour negative reserve remain unscored.

### v6 development regression: rejected

The full INT8 development-regression run reproduced the earlier validation
measurements and selected threshold **0.8** before evaluating regression
audio. `4090-v6-threshold-freeze-check.json` verifies the same model,
threshold and validation metrics; no threshold was tuned on regression.
Regression mean source recall is **61.120%** (Piper 73.906%, Kokoro 48.333%),
pooled 1,535/2,160 correlated renderings (71.065%). Confusing-negative false
detections are 0.7670%, maximum source 0.9375%. There are 74 premature positive
events, all in Kokoro. Hit latency is median 120 ms, p95 220 ms.
One continuous false event in 1.35816 source hours is **0.7363/hour**, with
a one-sided 95% upper bound of **3.4929/hour**. The candidate remains rejected.
The report is `4090-quantized-v6-qwen-regression.json`; these familiar development
fixtures are not the independent final reserve. The validation gain has not
established comparable performance on the development voice holdouts.
Next priority is a trained pretrained-feature diagnostic on the same waveform
protocol to investigate the generalization bottleneck. Merely adding further
negative hours or declaring the stronger validation result a success would
not resolve this failure. Independent final fixtures remain unscored.

## Official pretrained-feature diagnostic preparation

The official openWakeWord code is pinned at
`368c03716d1e92591906a84949bc477f3a834455` in a separate system-site environment.
Its v0.5.1 release feature models were downloaded with release asset IDs,
sizes and local SHA-256 hashes recorded in `4090-pretrained-embedding-probe.json`.
The release API has no upstream digest for these older assets; byte hashes
pin what was downloaded rather than proving an independently published hash.
Only the feature backbone is used here; no pretrained wake-word classifier
is substituted for a trained Hey Nyx model.

`embedding_probe.py` successfully streams actual training WAVs from all three
engines, with both labels, through the official mel and 96-dimensional
embedding models on CUDA. It asserts that both sessions use the CUDA provider
and verifies repeatability after reset. Upstream `AudioFeatures.reset()` seeds
embedding history with random noise; the probe replaces that history with
embeddings computed from deterministic silence before processing each stream.
This is a feature/runtime probe, **not recognition accuracy**, and no frozen
final fixtures are used.

The first GPU attempt correctly rejected silent CPU fallback: ONNX Runtime
GPU 1.31.1 requires CUDA 13 and could not find `libcublasLt.so.13` through the
TensorFlow CUDA 12 loader. `gpu_python.py --cuda-major 13` now prioritizes the
existing pip CUDA 13 libraries for that process only. The default CUDA 12
loader still excludes them, preserving TensorFlow's launch behavior.
`requirements-oww-diagnostic.lock` records this separate environment with the
GPU runtime overriding the inherited CPU module. No MCU footprint, INT8
export, trained diagnostic classifier or comparison with WakeNet is claimed.

## Batched pretrained-feature consistency

Initial cache attempts failed strict comparison against official single-stream
inference. CUDA TF32 caused drift; after disabling it, an axis-free global
ReduceMax in the official mel graph still coupled dynamic range clipping across
examples. An actual waveform probe found a 0.50484 mel difference; the embedding
network itself matched individual inference. Failed caches were not trained.

`embedding_cache.py` derives an identified mel graph: ReduceMax over non-batch
axes, then broadcast Max/Min replacing dynamic Clip (ONNX bounds require scalars).
Original models remain intact. Derived model SHA-256:
`3ff4a5bc0e08a6427d5216b60c0dbc1787b17ed2ae3e984c12730cf5786372fc`.
With TF32 disabled, smoke passed 36 waveform reconstruction cases and four
sequential comparisons (maximum absolute error 4.58e-5; atol/rtol 1e-4).
Incomplete smoke caches are rejected by the trainer. The full cache checks its
entire first batch and uses only training/validation audio. No recognition
accuracy is implied by these consistency checks.

`train_embedding_head.py` implements a float desktop reference with 16 causal
embedding frames and validation-only selection. No trained result yet; this is
not an MCU export, INT8 candidate, or Hi ESP equivalence evidence.

The complete cache finished in 291.2 seconds: 94,418 sequences, 8,230,203
96-dimensional frames; 88,719 training and 5,699 validation sequences.
All 64 first-batch sequential comparisons passed (max error 6.10352e-5),
and 36 generated waveform frontend reconstruction cases matched exactly.
Binary length and split membership were audited; hashes and CUDA providers are
in `4090-embedding-cache-v1.json`. The 40,000-step reference-head run has started;
no development regression or independent final scores have been computed.

The first head run reached step 10,000 but failed report serialization because a
premature-event count was a NumPy int64. No checkpoint/result was published from
that run. The event predicate now explicitly returns Python bool; a fresh v1b
run retains the same seed/configuration and preserves the failed log locally.

The v1b reference completed 40,000 steps in 110.6 seconds on RTX 4090.
Validation selected step 40,000 / threshold 0.985: source-macro recall 30.026%,
Piper 6.719%, Kokoro 53.333%; pooled 257/2,160 correlated variants (11.898%).
There were zero ambient events in 1.20481 hours, upper 95% bound 2.48647/h;
maximum source confusing-negative rate 0.9375%. Hit latency median 251.75 ms,
p95 369.125 ms. This reference failed to improve validation recall and is
rejected; no regression or independent final evaluation is warranted.
Report: `4090-embedding-head-v1b.json`. The custom head, loss and positive-window
sampling differ from the upstream full training recipe, so this failure does
not establish that pretrained embeddings or the upstream method are unsuitable.
Next diagnostic should audit the positive-window distribution against full
stream inference and align training sampling/loss with the official recipe
before another large data expansion. No Hi ESP comparison was completed.

## Reference-head diagnosis and architecture control

The deterministic positive-window audit (`audit_embedding_head.py`, up to 300
renderings per source/split, no regression/final audio) finds validation Piper
median raw score 0.8555 at training-style endpoints and median valid-window peak
0.9317. Both are below selected threshold 0.985; this suggests separation under
the confusing-negative constraint, not simply universally absent positive
activation. These are raw diagnostic score quantiles, not smoothed event recall.
Sampling indices agree with the cache timeline: head score index i ends at
(i+1)*80 ms, and sampled indices ceil(keyword_end/80ms)..+2 match the trainer.

Pinned upstream `openwakeword/train.py` uses Linear -> LayerNorm -> ReLU twice,
whereas v1 normalized the flattened input and used dropout. The v2-upstream
experiment changes only that head architecture; seed, 40k steps, Adam1e-3,
negative weight20, sampling, cache and event validation are unchanged. It uses
logits with BCEWithLogitsLoss, equivalent to sigmoid+BCE for this binary loss.
This is a single-factor architecture experiment, not a reproduction of the full
upstream recipe: upstream additionally schedules negative weights/learning rate
and filters low-loss samples. These differences remain explicit.

The architecture-only v2 run completed in 116.7s. Selected step30k/threshold0.98:
source-macro recall42.370% (Piper28.906%, Kokoro55.833%), pooled689/2160;
zero ambient events, upper95%2.48647/h, max confusing-source false0.9375%.
This improves v1b's validation recall but remains inadequate; regression/final
remain unscored. At v1b threshold0.5, macro recall87.9% came with max-source
confusing false8.3% and two ambient events. Lowering the threshold therefore
exposes the separation problem; it is not an accepted improvement.

Next controlled experiment keeps the v2 head and all other settings, selecting
one max-scoring window from eight random candidates within each sampled
training confusing-negative row. Original sampling often selects easy context
rather than the confusing phrase. Mining uses only training cache rows, with
no gradient during selection and evaluation mode to avoid dropout selection
noise. Validation still independently chooses checkpoint/threshold; no final
fixtures enter mining. This is a targeted variant, not upstream full-recipe
reproduction. Source code hash is now captured at process startup.

Validation phrase attribution reproduced v2's exact confusing-negative rate
(20/3520 variants). The largest contributors are Kokoro "Hey Nick." (6/80),
"Hey Next." (5/80), and "Hey Mix." (3/80). The phrase audit uses only existing
validation rows and is not hard-negative mining input. This supports the
phonetic-confusion diagnosis; adding generic silence or an unrelated noise
filter would not specifically address these errors. Counts are correlated
renderings, not independent human trials. See `4090-embedding-v2-confusions.json`.

The v2 positive phrase audit independently reproduces all689 detected variants.
Piper standalone recall is32.97%; command-followed groups range20.07–33.19%.
Kokoro spelling groups (Nyx/Nicks/Nix) are53.75–57.5%. Thus Piper misses are not
restricted to command-followed clips, and no single spelling group explains
Kokoro behavior. These remain validation diagnostics, not independent trials.
`audit_embedding_confusions.py --label 1` uses the same smoothed event policy
and keyword hit window; default label0 preserves the negative phrase audit.

The v3 mining run completed40k steps in214.4s; validation selects20k/threshold0.9.
Source-macro recall51.380% (Piper22.344%, Kokoro80.417%), pooled622/2160;
zero ambient events/1.20481h, upper95%2.48647/h; max confusing-source false0.9375%.
Latency median224.375ms/p95312.909ms. This remains rejected and is not better
than the existing v6 MCU candidate on source-macro validation recall. The
reference gains primarily occur in Kokoro; Piper recall is worse than v2.
Single runs with changed RNG consumption do not establish statistical effects.
No regression or independent final fixture was scored in any of these runs.

Next optimization control should address another concrete upstream deviation:
`auto_train` uses target LR1e-4 with20% warmup, one-third hold and cosine decay,
whereas these diagnostics used constant1e-3. Upstream also ramps negative weight
and filters low-loss samples; these remain separate factors to investigate,
not claims of a reproduced recipe. Preserve the existing validation criteria
rather than relaxing the confusing-negative cap to make a candidate pass.

## Upstream learning-rate control

The v4-lr run changes v3's constant1e-3 learning rate to the pinned upstream
first-stage1e-4 target with20% warmup, one-third hold, and cosine decay.
Architecture, 8-candidate training-only mining, negative weight20, seed, dataset,
40k steps and validation criteria stay fixed. All40,000 schedule values were
compared against the upstream function extracted from its hash-pinned source:
maximum difference1.3553e-20 due to warmup arithmetic order, passing rtol1e-14.
The initial bit-exact assertion failed only at that floating-point precision;
no training source was changed to alter the live run. The reusable verifier
and `4090-embedding-lr-parity.json` record the comparison. Negative-weight
scheduling and low-loss filtering are still not reproduced.

## Ambient sampling coverage defect

The current uniform-row ambient sampler gives24 human speaker streams only2.12%
of environmental-negative sampling opportunities although they contain17.20%
of original negative exposure (3.22486h). Conversely930 short noise rows receive
82.23% of draws despite33.21% of exposure. Music receives11.67% of draws versus
45.59% of hours. This occurs in both the reference and MCU trainers; adding hours
alone did not proportionally expose the model to them.

`ambient_sampling.py` supplies an optional duration-weighted policy based on
original source seconds, excluding repetitions and inserted silence. A100,000
draw audit observes17.163% human draws (expected17.201%). This is a coverage
check, not proof of improved recognition; duration weighting is an explicit
experimental choice rather than a universal best-practice requirement.
`train_stream.py --ambient-sampling duration` will test only this change against
v6 on the same actual MicroFrontend waveforms and MCU architecture. Default rows
preserves prior sampling behavior. No final fixtures enter training.

The v4-lr reference finished40k steps in216.1s, selecting40k/threshold0.992.
Source-macro recall50.781% (Piper28.229%, Kokoro73.333%), pooled718/2160;
zero ambient events/1.20481h (upper95%2.48647/h), maximum confusing-source false1%.
Latency median236.969ms/p95335.581ms. This does not improve v3's51.380% source-
macro recall and remains rejected. Upstream LR scheduling alone did not resolve
this reference's generalization/separation problem; no regression/final scored.

The deployable v7-duration experiment has now launched on RTX4090: the same
v6 feature population, official64 MixedNet, equal-engine sampling,40k steps,
event-based validation and negative weight20, changing only environmental
source selection from uniform rows to original-duration weights. Its actual
INT8 export must pass quantized validation separately before any further
regression evaluation. No data were added and final fixtures stay frozen.

## v7 training export and raw-waveform pronunciation pilot

v7 completed40k steps on4090 in392.7s, selected40k, exported63,032-byte INT8
SHA256`2bee3fb56c05c288e4f8f8677ed412c9e583f50ca08b0cb5cd2314174cdc4a16`.
Float validation source-macro recall66.406% at0.98 (Piper71.979%, Kokoro60.833%).
Quantized validation is running separately; this is not acceptance or regression.

A training-only deterministic Piper pilot selects100 positive and100 negative
raw WAVs by SHA256(path), with no waveform filtering or relabeling. The reusable
selection script reproduced the exact initial manifest hash. Independent
Whisper tiny.en CPU INT8 initially reports19/100 positive text agreement, but
that scorer only applies wake aliases to standalone clips. A separate recount
preserving command suffixes gives32/100 positive agreement (19/40 standalone,
13/60 command-followed), versus75/100 negatives. Both original and corrected
counts remain published.45/100 positive transcripts start "hey next", a phrase
explicitly labeled negative elsewhere. This is a serious consistency concern,
not proof of acoustic mispronunciation: small ASR lexical bias and errors may
explain it. Before filtering or synthesizing more from the same recipe, seek
an independent stronger ASR check and inspect intended phonemes. Current
validation/final waveforms and labels remain unchanged. No wake model scores
were used to select this pilot. Source/ASR model hashes are in the pilot reports.

## v7 quantized validation: rejected

At validation-selected INT8 threshold0.97, v7 source-macro recall is67.865%
(Piper74.896%, Kokoro60.833%), pooled1584/2160 correlated variants (73.333%).
Confusing-negative false0.5682%, max source0.875%;12 premature positive events.
Zero ambient events/1.20481h still gives upper95%2.48647/h, exceeding0.5/h.
Latency median130ms/p95190ms. This is lower source-macro validation recall than
v6's69.948%; no development regression or independent final evaluation is run.
Changing ambient coverage alone did not establish improved recognition.

The validation-only evaluator now records metadata and full feature-binary
hashes, and retains5680 validation clip traces locally with testing/mac=null.
The small8-clip streaming comparison reproduces explicit resets;4/4 positive
hits in both float and INT8. Mean score drift0.002878, max0.25409, showing why
full quantized evaluation matters. These are narrow consistency diagnostics,
not MCU numeric equivalence or acceptance. See v7 reports and streaming audit.

Intended Piper phonemes for Nyx/Nix/Nicks are exactly identical /nɪks/;
Next has /nɛkst/, Nick /nɪk/, Mix /mɪks/. Thus simply respelling Nyx as Nicks
would not change this generator's phoneme input. This verifies only intent,
not waveform pronunciation. A pinned larger Whisper small.en reference is
being downloaded to recheck the exact same200 training WAVs before deciding
whether the tiny-ASR disagreement warrants a data-generation change.

## Larger-ASR check changes the pronunciation diagnosis

Whisper small.en (`Systran/faster-whisper-small.en` pinned at
`d1d751a5f8271d482d14ca55d9e2deeebbae577f`, model LFS SHA verified) completed
the exact same200 Piper training WAVs on4090 CUDA FP16. Alias-aware agreement
is80/100 positives (36/40 standalone,44/60 commands) and79/100 negatives.
Only7 positive transcripts start "hey next", versus45 with tiny.en. This large
same-waveform discrepancy shows the tiny-ASR concern cannot justify mass
quarantine or a claim that most Piper positives are mispronounced. Models share
the Whisper family and are not phonetic ground truth; capacity and runtime
precision both differ. All data and labels remain intact. No validation/final
or wake-classifier scores enter this ASR check. Reports retain original raw
matching and the command-alias recount separately.

The audit tool now accepts explicit CUDA/CPU and model descriptions, verifies
the actual runtime device, and hashes its own source. The same pinned small.en
is now auditing all360 original Qwen training clips, because the existing
30-positive subset was selected with tiny.en and may unnecessarily limit
training diversity. No revised subset or additional training run yet exists.

## Qwen audit recovery and v8 data control

The small.en check on all360 original training clips agrees with62/72 positives
and270/288 negatives. Relative to the existing tiny-selected30+/268- subset,
34 positive and10 negative additional WAVs pass. Two old positives and eight
old negatives disagree with small.en; they are preserved, not removed or
relabeled. The union is64+/278- Qwen source clips. This preserves earlier
training behavior while testing expanded coverage; ASR agreement remains an
imperfect screen and24 synthesis description conditions are not24 humans.

`select_training_asr.py --exclude-manifest` emits only newly agreeing source
rows, verifies hashes and rejects exact duplicate WAVs from the previous subset.
The v8 feature set adds264 training renderings (four existing regimes plus two
measured-RIR regimes) from44 WAVs. Exact byte comparison verifies the complete
old binary prefix and index rows; human/ambient indices and all validation/
regression rows are unchanged. Added WAV hashes are disjoint from all base WAVs.
Full audit:99,212 sequences,49,605,057 frames,17,482 source waveforms;
voice/source split checks pass. Independent final fixtures remain untouched.

The v8-small MCU training run is launched for40k steps on4090, comparing against
v6 with the same rows ambient policy, official64 architecture, equal-engine
sampling, negative weight20 and event validation. Only the recovered training
source addition changes. v7-duration remains a rejected separate sampling
control. Next: validate actual INT8 export, quantify float/INT8 drift, then
consider development regression only if validation evidence warrants it.
