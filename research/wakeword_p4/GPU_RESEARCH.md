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

## v7 attribution while v8 trains

The validation trace attribution independently reproduces1584 positive hits
and20 confusing-negative triggered variants at the frozen threshold0.97.
Piper per-variant recall:81.458% quiet,84.167% microphone-floor noise,
66.250% colored noise+synthetic reverb,67.708% real background+synthetic reverb.
Kokoro per-variant recall stays58.333–63.333%, including61.667% quiet.
All12 premature events are Kokoro. These show separate acoustic-condition and
source generalization deficits; they do not establish causes. Variants remain
correlated, and inference/thresholds were not recomputed. The attribution tool
accepts only validation-only traces and does not read other audio.

## v8 quantized validation: rejected

The v8 data-addition control completed40k steps in358.0s on4090, selected20k;
63,032-byte INT8 SHA256
`77121d0c5d9cda091ba57fe2f95131885ea081a9e73741138c3bdd9503d6354d`.
At frozen validation threshold0.9, source-macro recall63.594% (Piper64.271%,
Kokoro62.917%), pooled1385/2160 (64.120%), confusing-negative false0.5682%,
max source1%;6 premature events. Zero ambient events/1.20481h still gives
upper95%2.48647/h. Latency median120ms/p95200ms. Validation trace attribution
reproduces counts. No regression or final scores are warranted.

| MCU candidate | Same-population INT8 validation macro recall | Threshold | Ambient95% upper/h |
| --- | ---: | ---: | ---: |
| v6 Qwen tiny-selected | 69.948% | 0.8 | 2.48647 |
| v7 duration sampling | 67.865% | 0.97 | 2.48647 |
| v8 Qwen recovered additions | 63.594% | 0.9 | 2.48647 |

These are single-run development comparisons, not causal/statistical proof
that recovered Qwen audio is harmful. v6 remains the strongest of these on
this source-macro validation metric, but its regression already failed.
The8-clip float/INT8 comparison passes explicit reset reproduction and4/4
positive hits in both paths, mean score drift0.001406/max0.052972. This narrow
check is not recognition acceptance or Hi ESP equivalence.

## Chinese development audio preparation

To address absent Chinese coverage, `download_aishell_subset.py` pins
`AISHELL/AISHELL-1` revision`bbe295d530192a4cd41644b711c9aecd087df653`.
OpenSLR33 explicitly lists Apache License v.2.0; its source page is retained
and hashed. Before audio/model inspection, a fixed seed selects28 archives
from the first64 sorted speaker IDs:8 training,16 validation,4 development
regression, total1,055,437,060 archive bytes. Every downloaded archive must
match its pinned LFS SHA256 and size. Original source partition, transcript
coverage and PCM format still require extraction audit before any training
or scoring. This is read Mandarin, not a substitute for far-field/TV testing.
No Chinese wake classifier scores or new training runs exist yet; independent
final fixtures remain frozen. Public ESP-SR host searches still yielded no
usable native WakeNet runner; absence from these searches is not impossibility.

The AISHELL download and extraction audits completed. All28 archives match
pinned LFS hashes/lengths; all retained WAVs are from original official train,
16kmono PCM16, with disjoint planned speakers and no exact PCM duplicates.
Retained9,722 utterances:2,756 train/5,598 validation/1,368 development regression;
original durations3.56643h/6.91399h/1.72310h.13 of9,735 source WAVs are quarantined:
12 lack reference transcripts and one has zero frames. No model-based selection
or label changes occur. Initial strict attempts stopped at the missing
transcript and then the empty WAV; partial directories/logs remain locally.
The corrected fresh extraction preserves every raw archive and records exclusions
with identities and hashes. These corpus audits are not recognition evidence.

## Frozen v6 Mandarin validation result

Continuous MicroFrontend extraction completed28 speaker streams/4,592,264
frames, with exact chunk-continuity checks against concatenated first-two WAVs
for every speaker. Raw WAV and decoded PCM hashes were verified again. The
200ms utterance gaps and explicit0..159-sample alignment padding are counted
as inserted silence, excluded from exposure. Cache and index hashes are published.

v6 was scored only on the16 new Chinese validation speakers at its previously
frozen0.8 threshold; model SHA matched the old calibration report. Result:
**one false event in6.913985h; observed0.144634/h, one-sided95% Poisson upper0.686126/h**.
Three-second feature-prefix inference reproduced exactly after explicit state
reset for every stream. All events, including startup, are counted. The run
reads no new training/development-regression or independent final audio.

This fails the numerical false-event upper-bound target on this read-Mandarin
development population. The existing English regression also failed (one
event/1.35816h, upper3.4929/h). Low positive recall, absent far-field/TV testing
and missing matched WakeNet comparison remain unresolved. Do not pool exposure
to conceal failed populations. No model was retrained or threshold adjusted.
The initial prose mistakenly inferred zero total events from the final three
speaker log lines; the authoritative JSON always recorded one event. This
correction uses the summed per-speaker events and full report, not the log tail.

## v9 training-only negative-window mining control

v9-mining has started on4090 against the v6 feature set and settings, selecting
one highest-scoring window from4 candidates within each sampled training
confusing-negative row. Each row/source is still drawn from training only;
no validation/regression/final windows enter mining. Inference-mode selection
updates neither gradients nor batch statistics. The original candidate is
included, and extra candidate endpoints use isolated RNG4726, preserving the
main726 RNG for baseline source/positive/ambient sampling. A1000-row invariant
audit confirms the baseline endpoint and main RNG state stay identical for
one versus four candidates. This is a sampler check, not recognition evidence.

INT8 representative calibration explicitly disables mining, preserving the
baseline sampling path. Architecture official64,40k steps, equal engines,
ambient rows sampling, negative weight20 and existing event validation remain
fixed. First500 actual training steps completed without runtime failure;
results remain pending. The Mandarin dataset is not appended to this control:
its frozen v6 result remains a separate new-language development stress test.

## New board-free baseline route: released P4 QEMU

Espressif QEMU release`esp-develop-9.2.2-20261109` includes an x86_64 Linux
RISC-V emulator and`esp32p4-rom.bin`. The17,544,120-byte asset matches its
upstream release SHA256
`afa59465495295b8aa4f804768983fa30b0bbf33d0f68a047f9caa436b9164c0`.
After installing missing pixman/SDL2/slirp runtime libraries, actual
`qemu-system-riscv32 --version` and`-machine help` succeed and list`esp32p4`.
Python3.11 lacks tar extraction filters; the original archive was retained,
then member types and resolved path/link containment were explicitly checked
before extraction. No firmware or WakeNet inference has yet run.

This changes the next action from looking for native x86 libraries to testing
system emulation. Machine availability does not verify optimized PIE/DSP
instruction support or numerical equivalence. Exact firmware ESP-SR2.6.1 and
Hi ESP model pins must be established before recognition comparisons. An
isolated ESP-IDFv5.5.4 checkout is downloading for a no-board firmware boot
probe; production firmware and attached devices are not involved.

The v9 mining run has completed40k steps and exported INT8; validation-only
quantized calibration is running. Training metadata is published separately
from final recognition evidence. Independent final fixtures remain unscored.

## v9 mining INT8 validation: rejected

v9 completed40k steps in554.2s, selected20k, INT8 model SHA256
`3e083c7e8044d037201d3397a818d88cb5e85d19389e54d0f8cd798be09a99d1`.
At validation-selected0.5, source-macro recall63.646% (Piper83.125%,
Kokoro44.167%). Pooled1702/2160 (78.796%) is dominated by Piper variants and
must not be treated as a source-balanced gain. Confusing false0.8523%, max
source0.9375%;33 premature events. Zero ambient events/1.20481h, upper95%
2.48647/h. Hit latency median80ms/p95180ms. Training-only hard window selection
improved this candidate's Piper point estimate but did not resolve cross-source
recognition and does not surpass v6's69.948% macro validation recall. No
regression/final scored. Attribution reproduces counts; streaming consistency
report records score drift and reset checks separately.

The isolated IDF install initially stopped because the bundled OpenOCD version
probe lacked libusb. This is an SDK installation dependency failure, not a
firmware/recognition result. The missing runtime library is installed and a
fresh retry log preserves the original failure; no hardware probe is requested.


### 官方最佳实践再次复核与 P4 固件启动探针

重新获取乐鑫纯 TTS 定制说明及两个开源项目 README；内容哈希和
ESPHome HTTP 403 如实记录于 `results/4090-best-practices-refresh.json`。
已更新 `BEST_PRACTICES.md` 中过时的待办和实验状态，继续优先数据覆盖、
跨来源泛化、实际流式 INT8 验证和匹配误触条件的比较。

官方 ESP-IDF v5.5.4 安装完成，`qemu_probe/` 最小 P4 固件完整编译成功。
使用生成的 flash_args 合并为 4 MiB 镜像，无开发板、无烧录。
QEMU 25 秒运行只输出 Adding SPI flash device，未出现应用启动或计时标记。
因此固件启动尚未验证，WakeNet 推理更未验证。保留镜像与调试日志，
下一步定位 ROM 启动路径和仿真兼容性。完整构建哈希见
`results/4090-qemu-p4-firmware-boot-probe.json`。目标保持进行中。


### P4 无板启动与计时已实际通过

后续 HMP 寄存器显示 CPU 已在应用的 `esp_cpu_wait_for_intr` 中；
此前用“没有串口输出”推断未启动证据不足，未据此否定 QEMU。
在应用加入独立可读内存标记，并改为 `-serial file:...`、`-snapshot` 后，
完整 ROM/bootloader/app 日志和两个串口标记均被捕获，内存标记交叉验证通过。
实测 `elapsed_us=97984`，芯片 revision=301（3.01，IDF 日志格式为 v3.1）。
QEMU 正常 quit 返回 0。先前超时记录保留为历史，不再作为启动失败结论。

`qemu_probe/run_probe.py` 按 ELF 符号读取实际固件标记，保留原始 HMP/串口
日志及 ELF、闪存、QEMU 哈希。结果见 `4090-qemu-p4-boot-markers.json`。
这证明无板 ROM/应用/FreeRTOS/计时链路工作，不证明 WakeNet 的 PIE/DSP
内核可执行，更不是识别率或 Hi ESP 比较结果。下一步加载精确 2.6.1
ESP-SR 和 Hi ESP 模型，使用开发音频验证实际 detect 调用。


### Hi ESP 模型创建探针：确证当前 QEMU 不支持所需 PIE 指令

从组件注册表下载 ESP-SR **2.6.1**，component_hash 与主项目锁文件一致
（7ba1e97d…b4d14）。未改模型或推理库。Hi ESP 的 `wn9_hiesp` 约 284 KiB，
已由官方打包工具装入 model 分区，完整固件编译并从 ROM 启动。

第一次启用 IDF PSRAM 初始化时，设备 ID 读取报“PSRAM chip is not connected”。
核查官方 QEMU 源码，确有 0x48000000 起的 64 MiB 普通 RAM，但控制器缺少
该交易。仅在仿真探针关闭硬件 PSRAM 初始化，逐 4 KiB 页写读一个字后，
把其中 16 MiB 注册为外部堆。这个配置不能部署到板子，不代表真实 PSRAM 时序。

`esp_srmodel_init` 与 Hi ESP 模型枚举通过；create 尚未通过。实际异常：

- ROM `strtod` 的 `frm` CSR 读取触发非法指令，QEMU lazy-FPU 分类没有覆盖
  CSR 读取。尝试先执行普通浮点指令后，仍捕获 1 次 CSR fault。
- 随后 27 次重启均出现 PIE fault：MEPC=0x4003b430、MTVAL=0x03a1801f。
  精确 ELF 反汇编为 `esp.ld.128.usar.ip q0,a1,0`；addr2line 对应
  `dl_esp32p4_sr_memcpy.S:55`，调用方 `dl_nn_get_sigmoid_table`。

因此当前官方 P4 QEMU 能启动普通固件，却不能直接执行这个精确 WakeNet
二进制。没有完成一次 detect，没有计算召回或误触；准备的零 PCM 只用于
运行时兼容性探针，也没有实际推理完成。不能据此判断 Hi ESP 效果。
`qemu_wakenet_probe/` 保存源码、配置和依赖锁；结果和首个异常周期见
`4090-qemu-p4-wakenet-create-probe.json` 与 `4090-qemu-p4-wakenet-create-first-fault.txt`。
原始失败日志和 ELF/闪存均保留在 ignored `.local/wakeword/`。
后续保留 Hi ESP 比较未完成，独立数据与模型改进仍可继续，目标保持 active。


### v10：训练期 SpecAugment 受控实验（进行中）

在 v6 同一特征集、同一 official64 架构、相同 40,000 步/学习率和
seed=726/等来源采样/行均匀环境采样上，仅打开训练期时频遮挡。
每样本最多一条时间遮挡（0..10 帧，100 ms）和一条频率遮挡（0..4/40 bins），
宽度含 0 均匀采样，填 0；独立 RNG seed=10726，不消耗原采样 RNG。
验证、推理、INT8 代表校准均关闭遮挡；困难负样本候选数仍为 1。
该参数是受控试验，不声称等于官方 notebook 默认值。

真实训练特征的 128 样本输入审计通过：关闭时不改变输入或随机状态，
开启时原输入不变，可复现，所有变化值均为 0。审计不是识别证据。
代码 `spec_augment.py`，输入审计 `4090-v10-specaugment-input-audit.json`。
训练路径 `.local/wakeword/model-gpu-v10-specaugment`，未访问最终集。

另准备下一项独立实验 `features-gpu-v11-mandarin`：仅将已审计的 AISHELL
8 名训练说话人、3.566425 小时原始语音的连续 MicroFrontend 流追加到
human 训练索引。基准二进制、关键词索引和全部验证/开发回归条目保持不变。
该准备不代表已启动中文训练；v10 不使用这些新增数据。


v11 中文训练对照现已启动：仍为 40,000 步、与 v6 相同参数，SpecAugment
关闭，只使用上面新增的中文训练流。按原有行均匀采样，中文占环境采样行
约 0.7%，不声称按时长充分覆盖；本次不同时改变采样权重。
准备审计见 `4090-v11-mandarin-training-features.json`。

进一步复核 DiPCo 远场对话来源：HF 镜像 `huckiyang/DiPCo` 固定版本
`e2b29d3d0d88692c744feb15e290f7316b68014e` 的 LICENSE 明确为
CDLA-Permissive-1.0。来源审计见 `4090-dipco-source-review.json`。
尚未下载或使用其音频；后续必须按会话/说话人隔离，重复参与者的 S09/S10
不能跨划分，同一会话 35 个麦克风视角不能虚增曝光小时数。


### v10 INT8 验证改善，继续开发回归

40,000 步训练 465.4 秒，最佳浮点检查点 20,000 步；25,537 参数，
INT8 63,032 字节，SHA256
`53d00ce7d3ef831056e7d395a2742c5e2892fb37857d12003632f882057a9a90`。
量化验证阈值固定 **0.9**，来源平均 **77.6823%**（v6 69.9479%，+7.7344pp）。
Kokoro 85.4167%，Piper 69.9479%；相较 v6 有来源间取舍，池化召回只有
71.6667%（1548/2160），不能把来源平均与池化互换。
近音负样本错误 0.88068%，最差来源 1%；提前事件 0。
环境 0 次/1.204811 小时，95% 单侧上界仍为 **2.48647/h**；
中位延迟 130 ms、p95 220 ms。当前只是验证改善，未通过目标验收。

因验证改善而启动 v10 开发回归、float/INT8 重置一致性和中文验证，
不改变 0.9 阈值，不访问独立最终集。训练与验证完整报告分别见
`4090-training-v10-specaugment.json`、`4090-quantized-v10-specaugment-validation.json`。


### v10 回归拒绝；中文统计门槛通过仅适用于该语料

严格保持验证阈值 0.9，v10 开发回归来源平均 **57.8906%**，低于 v6
61.1198%；Piper 68.2813%，Kokoro 47.5%，池化 65.9722%。
最差来源近音误触 1.09375%，超 1% 开发约束。提前事件 60，均为 Kokoro；
其中 57 个漏检样本虽在合法窗口内有高分，但没有新的合法 rising event。
不能用窗口峰值替代事件召回，也没有因此修改标注/放宽时间窗。
时间遮挡可能影响触发时序只是后续假设，不是已确证原因。
回归环境 0 次/1.358162 小时，上界仍 2.20573/h。
因此 v10 **不晋级验收**，不评分独立最终集。

另在冻结 0.9 阈值下，AISHELL 全 16 名验证说话人的逐流事件之和为 **0**，
6.913985 小时单侧 95% 上界 **0.433286/h**，中文朗读语料的门槛通过。
该值没有与英文环境小时数池化，不能代表电视/远场混合场景。
float/INT8 8 个验证流的重置可复现；4/4 正样本命中两者一致，
平均绝对分数差 0.00138942，最大 0.0979591；窗口相位校正保持数值对齐。
一致性子集不是额外识别率验收。
结果分别见 v10 regression、aishell-frozen-validation、streaming-consistency
和 regression-event-attribution JSON。

### v11 中文负样本对照拒绝

40,000 步 440.0 秒，选择 10,000 步（没有检查点满足全部 float 开发约束）。
INT8 验证最优回退阈值 0.994，来源平均仅 **40.0781%**，Piper 31.8229%，
Kokoro 48.3333%，最差来源近音误触 **1.0625%**，无合格工作点。
环境 0 次/1.204811 小时，上界仍 2.48647/h，不能据此宣称误触达标。
候选拒绝，不再消耗开发回归或最终集；这不证明中文负样本一般无用，
仅证明这一次行均匀追加与固定训练配置没有改善。
模型 SHA256 `8244c3d2d581a493fed073e8e40e69c4a262eed1f646cc0a52bc05f38454bbda`，
63,032 字节。完整训练/量化验证报告保存为 v11-mandarin JSON。
目标保持进行中；尚无 Hi ESP 匹配基线，也未完成最终验收。


### v12 频率遮挡消融与真实远场对话开发语料

v12 继续同一 v6 数据/official64/40,000 步/seed=726，只启用频率遮挡
0..4 bins、关闭时间遮挡。它检验 v10 提前事件的潜在时序影响；不能据此
假定该假设已成立。INT8 验证前不宣称改善，独立最终集继续冻结。

DiPCo 固定 HF 版本成功下载 5 个官方 dev 会话、单一通道/会话，
共有 16 名参与者。初次 U01.CH1 统一通道计划在清单阶段失败，原目录保留；
镜像缺文件，因此在任何音频评分前，改为每会话字典序第一个可用远场通道：
S02/S04=U01.CH1，S05/S10=U04.CH1，S09=U02.CH6。不同设备位置属于分布限制。
官方 eval 没有下载，不把相关的多麦克风视角累加为额外曝光时间。

S02/S04 训练 **1.265503h**，S05 验证 **0.754488h**，S09/S10 开发回归
**0.712196h**；共享参与者 S09/S10 始终同划分。全部音频 LFS 大小/SHA256、
PCM 格式/去重、转录 session/speaker 和字面目标词排查通过。
字面转录没有目标词不等于每一段重叠语音的逐音素真值，没有模型筛选。
README 与完整 CDLA-Permissive-1.0 LICENSE 均保留在本地，哈希进入报告。

提取 5 条原始连续录音的真实 MicroFrontend：983,579 帧。
不插入话语间空白，10 秒块共享同一前端状态，只在录音末尾对齐最多 159
个样本且不计曝光；每流前 20 秒 chunk/concat 特征精确一致。
`prepare_manifest_human_features.py --continuous-recordings` 明确选择此模式，
原 AISHELL 多话语拼接默认行为保留。全部元数据与来源见 dipco 三份 JSON。
该数据尚未加入训练，先在冻结的 v6/v10 阈值下检查 S05 开发验证，
不调整阈值，不使用 S09/S10 回归，不使用任何独立最终音频。


### DiPCo 冻结阈值诊断：v10 对话误触暴露

S05 单通道原始连续远场验证 **0.754488h**，重置前 3 秒输出可复现。
v6 原阈值 0.8：**0 次**，95% 单侧上界 **3.97055/h**。
v10 原阈值 0.9：**1 次**，观测 **1.32540/h**，95% 单侧上界 **6.28753/h**。
两者分别使用原验证阈值，没有针对新会话重选，也没有把多模型的音频时长
相加；短时长不足以证明任一模型达标。v10 中文朗读上的 0.433/h 上界
不能推广到真实远场对话。仅 S05 已做分类器诊断，S09/S10 开发回归和
官方 eval 未评分，训练会话未被挖掘为专用误触片段。
完整结果见 `4090-v6-dipco-frozen-validation.json`、
`4090-v10-dipco-frozen-validation.json`。


v12 训练 388.4 秒、最佳 20,000 步，INT8 63,032 字节，
SHA256 `729b722bd020451b708e92bd8af454dd844129a1645611707ffbd1ffb071c1ef`。
验证正在执行。另启动 v13 DiPCo 训练会话追加对照，维持 v6 参数、
40,000 步、关闭 SpecAugment，仅新增 S02/S04 两条训练连续录音。
特征索引/二进制与原有验证及开发回归不变，S05 没有加入训练。
真实对话占环境采样行 2/1133（约 0.1765%），不同于其原始时长占比，
本轮不额外改变采样权重。来源审计见 `4090-v13-dialogue-training-features.json`。


### v12 频率遮挡单独未改善

INT8 验证阈值 **0.98**，来源平均 **65.7552%**，Piper 61.5104%、
Kokoro 70.0%；池化 62.4537%，提前事件 1，最差近音误触 0.875%。
环境仍为 0/1.204811h，单侧 95% 上界 2.48647/h。
低于 v6 来源平均 69.9479%，因此拒绝此候选，不进一步消耗开发回归/最终集。
这不证明时间遮挡就是 v10 提前事件的原因；消融没有重现 v10 验证改善。

为避免只围绕同一从零训练小网络反复调参，另固定下载官方 Apache-2.0
microWakeWord Hey Jarvis v2 模型作为预训练迁移研究来源：仓库 commit
`40ff33f57f8fc6ad71a75ef085abbf742495b225`，模型 SHA256
`21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77`。
它的 10 ms / [1,3,40] 输入协议一致，但 30/60 通道、单深度卷积核与当前
32/64 MixConv 不同，不能直接加载或声称兼容。仅完成张量结构和许可审查，
没有进行推理/迁移训练。若采用，须先重建匹配网络并在真实特征上验证数值
一致性，再对比相同架构从零训练，不能把别的唤醒词模型当 Nyx 识别结果。
详见 `4090-microwakeword-transfer-source-review.json`。


### 预训练 MCU 模型的可训练网络重建诊断

新增 `reconstruct_mww_pretrained.py`，严格校验上面固定模型 SHA。
重建 150 帧上下文、30/60 通道的对应网络，22,801 参数，映射全部 10 个
卷积/深度卷积/全连接算子。按每通道量化尺度解码权重；已折叠 BN 的偏置
进入冻结的等效仿射 BN，均值/方差设为恒等参数。这些是重建参数，
不是恢复了上游原始 BN 统计，不能声称取得原始训练检查点。

仅在 12 条现有 **训练** 序列上，连续浮点与原官方 INT8 输出平均绝对差
1.18218e-5，最大 0.00125093；没有使用验证/回归/最终音频。
这只是窄范围重建诊断，不是 Nyx 识别证据，也不足以证明全域数值一致：
输入并非 Hey Jarvis 正样本，须先增加其目标词输入和中间激活检查。
尚未重训/迁移。权重留在 ignored `.local/`；报告
`4090-microwakeword-pretrained-reconstruction-v1.json` 保留每序列分数范围，
不以低概率区间的一致性取代充分验证。


重建诊断补充限制：上述 12 条输入的原官方 INT8 输出 **全部为 0**。
因此最大 0.00125 的输出差只覆盖接近零概率区间，不能认定重建已通过。
下一步必须使用单独合成的 Hey Jarvis 目标词探针覆盖高概率区间，并检查
中间激活，再决定是否值得开展同架构迁移/从零训练对照。


### v13 真实对话训练追加未改善

40,000 步 371.2 秒，选择 20,000 步，25,537 参数、INT8 63,032 字节，
SHA256 `c01618615abfa362460a826d711a1cd14d3ef14c82eeea64aca748e519cf940f`。
固定原始验证集校准阈值 **0.97**，来源平均 **62.6823%**，
Piper 63.2813%、Kokoro 62.0833%，池化 63.1481%（1364/2160）。
提前事件 7，最差近音误触 0.9375%；环境 0/1.204811h，95% 上界 2.48647/h。
未超过 v6，拒绝此候选，不继续开发回归、DiPCo 回归或独立最终评估。
这只否定本次固定行采样配置，不能推断真实对话训练普遍无用。
完整训练/量化验证报告见 `4090-training-v13-dialogue.json`、
`4090-quantized-v13-dialogue-validation.json`。

本轮完成了 v12/v13 训练与 INT8 验证、DiPCo 真实远场开发语料和旧模型的
冻结阈值诊断，并建立预训练权重重建代码；没有合格新候选，目标仍 active。
接下来优先完成原预训练目标词/中间激活探针，避免将低概率区间的一致性
当成迁移依据。最终集继续冻结，Hi ESP 无板比较仍未完成。


### 官方预训练目标词与中间激活检查通过

新增 `generate_teacher_probes.py` 与 `probe_mww_reconstruction.py`。
4090 使用固定 Piper 检查点，从原训练说话人中选 4 人，每人生成
Hey Jarvis、带后续指令的 Hey Jarvis、Hey Nicks、Hey Next 各一条，共 16 条。
这批音频仅为兼容性探针，不计入 Nyx 召回率或独立评估。

在真实 MicroFrontend 特征、相同量化输入上，对照官方 INT8 与重建浮点网络。
8 条原目标词均达到高响应；输出平均绝对差 0.000612103，最大 0.085649。
9 个中间层平均误差为 0.0239–0.4531 个原激活量化 LSB；局部最大可达
38.66 LSB，不能声称位精确。首条目标音频的状态重置重复性通过。
预先定义的诊断门槛通过，仅允许继续迁移研究，不代表 Nyx 质量达标。

报告：`results/4090-mww-teacher-probe-v1.json` 与
`results/4090-mww-teacher-probe-generation-v1.json`。
本轮再次在线核对四项官方来源并更新 BEST_PRACTICES.md。
后续同架构 scratch/transfer 对照尚未执行；目标仍 active，最终集未评分。


### v14/v15 同架构初始化对照启动

训练器新增 teacher60 架构与可配置上下文，重建权重 SHA 强制校验。
两组均 150 帧、22,801 参数、固定 seed726、40,000 步、每 5,000 步验证，
v6 同一数据、engine exponent0、按环境行采样，无 SpecAugment 或困难挖掘。
Adam/LR/负样本惩罚保持原配方。两组冻结四个 BN 层；迁移组保留重建仿射参数，
随机组为默认 BN 参数，所以比较的是整体初始化方案，不是只改变卷积核的消融。

v14 随机初始化，v15 加载经过目标词/逐层探针检查的官方模型重建权重，
然后恢复与 v14 完全相同的随机 Dense 分类头。头部 SHA256 均为
`a41cc84a5e15a42a13ee9fe02e38242fca4dd9afe67a19ba736587970b46fc54`。
初始化记录见 `4090-v14-teacher-initialization.json`、
`4090-v15-teacher-initialization.json`。流式/窗口数值检查也改为读取训练上下文。
两个 GPU 训练进程已实际启动；当前尚无完整训练或 INT8 验证结论。


### v14/v15 训练与 INT8 验证完成

两组均完成 40,000 步；v14 425.2 秒，v15 432.4 秒。训练报告见
`4090-training-v14-teacher-scratch.json`、`4090-training-v15-teacher-transfer.json`。
v14 选择 35,000 步，INT8 52,584 字节，SHA256
`3663193972558120c25e41dd684606a9fe355f3b8f6483133c8eec42352dbee7`；
v15 选择 30,000 步，54,440 字节，SHA256
`d527fda86027d532d896af0f0ac1ee03bdfd6ab71c4b4ce2f18f9407cde3abe8`。

完整 INT8 事件验证：v14 阈值 0.97，来源平均 77.8125%，Piper78.9583%、
Kokoro76.6667%，1700/2160；提前事件14，中位延迟130ms、P95 200ms。
v15 阈值0.95，来源平均75.7292%，Piper81.4583%、Kokoro70.0%，
1732/2160；提前事件3，中位延迟130ms、P95 240ms。两组均超过v6的验证来源平均，
但迁移没有超过同架构随机初始化，且单次种子实验不构成普遍结论。

两组环境均0/1.204811h，单侧95%上界2.48647/h，尚未满足0.5/h。
验证报告见 `4090-quantized-v14-teacher-validation.json` 与
`4090-quantized-v15-teacher-validation.json`。两组已启动开发回归与小样本
流式/窗口/量化比较，独立最终集未评分。


### 同架构开发回归：迁移改善，但仍未验收

验证阈值不变。v14 开发回归来源平均61.9271%，Piper75.5208%、Kokoro48.3333%，
提前事件86，最差近音误触1.875%；v15来源平均67.6042%，Piper78.5417%、
Kokoro56.6667%，提前事件68，最差近音误触0.46875%。v15超过v6开发回归61.1198%，
但来源差距与提前事件仍明显。两组环境0/1.358162h，95%上界2.20573/h，
不是0.5/h达标。不能因验证排序反转而在回归上重选阈值或修改事件规则。

每组8条固定验证探针，两条推理路径重置重复性均通过，4/4正样本均命中。
150帧窗口与流式浮点在相位对齐后最大误差v14为2.38419e-7，v15为1.78814e-7。
INT8/浮点平均绝对差v14为0.0099911、最大0.192154；v15为0.00134874、最大0.327686。
因此只证明小样本重置与相位正确，不能称INT8数值等同。实际INT8全验证/回归决定质量。
报告见 `4090-quantized-v14-teacher-regression.json`、
`4090-quantized-v15-teacher-regression.json`、两份`teacher-streaming-comparison.json`。

v14不再扩展评估；v15已启动固定阈值0.95的AISHELL与DiPCo开发验证，
仍不访问独立最终集。目标active，Hi ESP比较未完成。
