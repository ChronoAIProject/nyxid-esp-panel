> 最新评测资格说明（2026-11-10）：官方Hi ESP测试样本已在QEMU触发，
> 但能量代理结束点比事件晚666–738ms，固定-100..+800ms窗口会排除这些触发。
> 因此历史配对集“9/12、1/12”等是代理窗口命中数，不能作为真实漏检率或音素延迟排名。
> Hi ESP在配对集实际有事件的目标音频是5/12。原始事件/分数不改，最终集继续封存。
> 反证见results/4090-hiesp-official-control-endpoint-audit-v1.json。

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


### v15 固定阈值中文/远场开发验证：仍失败

保持英文验证阈值0.95，不重新标定。AISHELL16个验证说话人原始6.913985h，
2次误触，单侧95%上界0.910588/h，超过目标0.5/h。DiPCo S05原始0.754488h
零误触，上界3.97055/h；时长不足以验收。逐流前3秒状态重置检查通过。
报告 `4090-v15-mandarin-frozen-validation.json` 与
`4090-v15-dipco-frozen-validation.json`。不可将两语料随意合并掩盖中文失败。

结论：本轮同架构迁移在开发回归来源平均召回上有改善，但没有解决提前事件、
跨合成来源差距或中文误触；不接受v15为达标模型，也不消耗独立最终集。
后续优先检查提前事件的训练对齐/困难近音负样本，以及真实负音频采样权重。
此前简单追加中文和对话行均失败，不能只重复同一低概率采样配置。
所有训练/INT8验证/开发回归/探针/新增语料检查均完成；研究总目标仍active，
Hi ESP相同音频、匹配误触比较仍未完成。


### v16/v17 真实负音频采样对照

将v11仅训练中文8流与DiPCo仅训练S02/S04两流组合；验证/开发回归行不变，
所有human路径可解析，关键字特征索引/二进制不变。修复追加脚本的固定
human-additions目录名，改用源索引哈希命名，允许保留前一次追加的符号链接。
新增组合特征来源报告 `4090-v16-combined-human-features.json`。

六类环境来源按行采样：中文0.7011%、远场对话0.1753%、英语朗读2.1034%、
MS-SNSD3.9439%、噪声81.5074%、音乐11.5688%。新增分层策略给每类1/6概率，
类内按原始时长采样；只影响每批48个ambient负样本，不改变正样本或近音负样本。
概率归一、正值、各层质量相等与评测行不变检查通过。
审计见 `4090-v16-combined-sampling-audit.json`。

v16按行，v17分层，同组合数据、teacher60/context150、预训练初始化和相同随机头、
冻结BN，40k步、每5k验证，其余同v15。两个4090训练进程已实际启动。
这是采样分布对照，不预设分层必然改善；最终集未访问。


v15已有开发回归轨迹归因（无新推理/阈值修改）：68次提前事件全在Kokoro，
Piper为0。提前偏移110–400ms，中位155ms；Kokoro另有60个未命中样本在
有效窗口内峰值仍超过阈值，但没有新事件。不能用峰值替换事件，也不能仅由
这些偏移推断标注错误。新增可复现脚本 `diagnose_early_events.py`，统计与既有
报告的正例数/命中数/提前数逐项核对。报告
`4090-v15-premature-event-diagnostic.json`；本次未触及独立最终集。


### v16/v17 实际 INT8 验证结果

40k训练完成，v16耗时428.3s、选择15k步；v17耗时429.6s、选择10k步。
两者均22,801参数、INT8 54,440字节。v16模型SHA256
`4a4feae50cc83c659a1fcb8012cd3b420067d7210557fe3fa96423f1e54065a5`；
v17模型SHA256 `69861afc52bc8bea5e1afbf0748f1f586c75e797452f2314aa9ee02c0ff060f2`。
训练报告与配置见 `4090-training-v16-human-rows.json`、
`4090-training-v17-human-stratified.json`。

固定原v6完整INT8验证，各自重新校准后阈值均0.95。v16来源平均59.4531%，
Piper46.4063%、Kokoro72.5%，提前事件2；v17来源平均70.4167%，
Piper70%、Kokoro70.8333%，提前事件0。v17中位延迟140ms、P95 220ms，
最差近音误触0.6875%。两组环境均0/1.204811h，95%上界2.48647/h。
报告见 `4090-quantized-v16-human-validation.json`、
`4090-quantized-v17-human-validation.json`。

分层明显优于同数据按行，但验证召回未超过v15；因更均衡且无提前事件，
v17继续开发回归、固定阈值中文/DiPCo验证和小样本导出检查。
v16拒绝，不进一步评分；独立最终集保持冻结。


### v17 中文改善，但开发回归/远场仍失败

保持阈值0.95，AISHELL16个验证流6.913985h为0事件，单侧95%上界
0.433286/h，满足此中文朗读语料的0.5/h目标；v15同语料2事件、上界0.910588/h。
这不是跨场景验收。DiPCo S05为1事件/0.754488h，上界6.28753/h，失败。
英文开发回归来源平均66.1458%，Piper66.4583%、Kokoro65.8333%，
比v15的67.6042%略低、来源差距减小；提前事件40（v15为68），
最差近音误触0.875%。环境1事件/1.358162h，95%上界3.49286/h，仍失败。
分层减少部分错误但产生/保留其他错误，拒绝将此候选视为达标模型。

8条固定验证探针的两路径重置重复性通过，4/4正样本两路径均命中；
INT8/浮点平均绝对差0.0025154、最大0.126196，仍不是数值等同或MCU实机证明。
报告 `4090-quantized-v17-human-regression.json`、
`4090-v17-mandarin-frozen-validation.json`、
`4090-v17-dipco-frozen-validation.json`、`4090-v17-streaming-comparison.json`。

本轮两组训练、量化验证及v17扩展开发检查全部结束，失败证据保留并推送。
下一步需检查远场误触的训练覆盖及正词内部时序，不能通过重新调回归阈值、
扩大允许窗口或评分独立最终集来制造改进。总目标active，Hi ESP比较未完成。


### v18/v19 训练正词前缀负监督实验启动

检查确认现有训练只有正词结束后30..270ms的正窗口，以及近音词/环境负窗口，
没有对正词未结束的片段显式负监督。加入可选prefix-negatives参数，默认0保持原配方。
每批仍32正/96负，其中原48个近音负槽的前8或16个改为训练正词前缀：
裁剪终点=原keyword_frame减均匀15..35帧，标签0、权重20；独立RNG20726，
仍生成原始全部样本以保留基线RNG序列，然后替换，不更改标签或评测窗口。
所有37,980条训练正例的最早裁剪均能容纳150帧（最小终点178帧），无需越界补齐。

v18每批8条，v19每批16条，其余同v17：组合人声负数据、六类分层采样、
teacher60/context150、原官方重建初始化且重置相同随机分类头、冻结BN、40k/5k验证。
两个RTX4090进程已实际启动。该时序监督使用能量代理结束标签，并非音素真值，
可能增加延迟或损伤召回；必须用原INT8事件评测判断，不能预设会修复提前事件。
独立最终集继续冻结。


v17 DiPCo既有误触在S05/U04.CH1约1079.18s，邻近转写为普通讨论、
无目标词字面文本，不能凭转写定位具体声学原因。保留上下文及转写哈希于
`4090-v17-dipco-event-transcript-context.json`，不把验证音频挖入训练。
另修正未来人声评测报告的limits文案：此前DiPCo报告错误沿用“read Mandarin”，
实际来源与事件均为DiPCo；历史原始结果不改写，脚本现在使用实际corpus-name。


### v18/v19 前缀负样本训练与INT8验证完成

两组完成40k步，v18耗时446.2s、选35k，v19耗时452.5s、选40k；
均INT8 54,440字节。v18 SHA256
`d23f01fbea7cc9d9722265ae26ed60650038380aa0198919862af46add0c9240`；
v19 SHA256 `4d4db51422775b400b6a5a101b5c01cc9177c95fa7fb61333795c0ca699f5151`。
完整配置与历史见 `4090-training-v18-prefix8.json`、`4090-training-v19-prefix16.json`。

INT8验证v18阈值0.97，来源平均66.5104%，Piper73.4375%、Kokoro59.5833%，
提前事件0；v19阈值0.95，来源平均78.3073%，Piper79.1146%、Kokoro77.5%，
1705/2160、提前事件3，中位延迟130ms、P95 220ms，最差近音误触0.875%。
v19超过v17的70.4167%，但未证明开发回归改善。两者环境均0/1.204811h，
95%上界2.48647/h，仍不是0.5/h达标。验证报告见
`4090-quantized-v18-prefix-validation.json`、`4090-quantized-v19-prefix-validation.json`。

v18拒绝，不继续扩展；v19启动原开发回归、固定阈值中文/DiPCo与小样本
流式导出检查，独立最终集未评分。


### v19开发回归改善，仍未达到总目标

冻结验证阈值0.95。开发回归来源平均69.4271%，Piper74.6875%、Kokoro64.1667%，
1588/2160（池化73.5185%），提前事件17（v17为40，v15为68）。
中位延迟130ms、P95 210ms，最差近音误触0.625%。环境0/1.358162h，
95%上界2.20573/h，仍不足以证明0.5/h。

AISHELL6.913985h为0事件，上界0.433286/h；DiPCo S05 0/0.754488h，
上界3.97055/h。中文语料目标满足，远场仍因曝光不足不能验收。
这些是多轮开发使用的数据，不能作为独立最终质量证明，也不随意合并掩盖场景缺口。
两条推理路径在8条固定探针中状态重置重复性通过，4/4正样本均命中；
INT8/浮点平均绝对差0.00192886、最大0.161313，不能称数值等同。
报告见 `4090-quantized-v19-prefix-regression.json`、
`4090-v19-mandarin-frozen-validation.json`、`4090-v19-dipco-frozen-validation.json`、
`4090-v19-streaming-comparison.json`。

前缀16条方案有开发改善，但8条方案下降，单种子结果不代表普遍因果。
v19仅保留为更有希望的研究候选，不视作产品达标；来源泛化和17次提前事件仍待解决。
本轮两组训练与全部计划开发检查完成，最终集继续冻结；目标active，Hi ESP比较未完成。
下一步应扩大真实连续英文/远场开发覆盖，并在现有说话人隔离下独立审计正词时间标签，
不能凭开发集全零就直接宣称低误触或通过调整回归阈值提高分数。


### v19额外中文/远场开发回归首次检查

扩展evaluate_frozen_human.py的显式split参数：默认validation不变，testing仅指
已预划分的开发回归，不是独立最终集。报告分别标明development_regression_scored
与independent_final_scored，避免把内部split名称当作最终测试。
保持原模型SHA与阈值0.95。AISHELL4个回归说话人1.723096h为0事件，
95%上界1.73858/h；DiPCo S09/S10共0.712196h为0事件，上界4.20633/h。
均不足以单独验收。报告 `4090-v19-mandarin-frozen-regression.json`、
`4090-v19-dipco-frozen-regression.json`。这些开发回归此前未对唤醒模型评分，
本轮首次使用后明确作为开发证据，不能再称未使用的独立保留集。

另外开始下载OpenSLR官方LibriSpeech dev-other，校验值来自官方md5sum.txt：
`c8d0bcc9cca99d4f8b62fcc847357931`。新脚本prepare_libri_development.py会
先审计与既有人声及最终manifest的说话人ID隔离、目标词字面转写，
保留逐文件FLAC/PCM哈希及原始时长，再生成统一10ms前端连续流，
核对前两段拼接一致性。整个dev-other预先指定为开发验证，不作最终集。
下载仍运行，尚无该语料评分；独立最终音频未访问。


### 新英文困难朗读开发集揭示剩余误触

官方dev-other下载完成且MD5通过，SHA256
`12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365`。
初次准备因本机tarfile不支持filter参数在解压前失败，原日志保留；
改为先检查所有成员的路径/常规文件或目录类型再解压后完成。33说话人、
5.121185原始小时，逐文件FLAC/PCM哈希、转写哈希、统一前端和首两段拼接检查通过。
与现有真人开发语料及最终负样本LibriSpeech ID无交叉；不代表与所有TTS声纹
已做物理身份去重。索引SHA256
`37459cd527105354fc6abd99e6032e7905f53c433a58618206eed5fdbbae789f`。
来源报告 `4090-libri-dev-other-feature-audit.json`。全部预定为开发验证，
不是新的独立最终集，不能再称未使用数据；朗读拼接不等于自然连续对话。

v19与v17均使用先前冻结英文阈值0.95，未重选。v19为1事件/5.121185h，
95%上界0.926322/h；v17为2事件，上界1.22936/h。两者均未通过0.5/h。
报告 `4090-v19-libri-dev-other-validation.json`、
`4090-v17-libri-dev-other-validation.json`。不把它与其他开发语料合并以掩盖失败。

v19事件位于说话人1255流334.58s，对应1255-90407-0014.flac约1.66s，
转写为“AND THERE ON THE NETS THEY SAT”。上下文与逐文件哈希见
`4090-v19-dev-other-event-context.json`。这提示近音词问题，但不是音素因果证明；
该验证音频不加入训练，阈值不改。另存v19提前事件归因报告
`4090-v19-premature-event-diagnostic.json`：17次全在Kokoro、偏移110..180ms，
与v15的68次相比减少但未消除。

本轮完成额外中文/远场回归及新增困难英文开发集的固定阈值对照，均有真实
音频依据。v19仍非达标模型，下一步需扩充训练来源的近音词覆盖并独立审计
正词边界；独立最终音频未评分，Hi ESP比较未完成，总目标active。


### v20近音词训练补充与质量检查

新增generate_near_negatives.py，固定原Piper检查点，RTX4090合成32个原训练
说话人×24个近音短语=768条。类别涵盖nets/necks/mixes/next/nice/six等，
受开发错误启发，不是独立挑战集。说话人与原Piper验证/开发回归身份无交叉，
逐WAV哈希核对；原dev-other误触录音没有加入训练。

small.en CUDA FP16全768条独立ASR审计完成。Hey Necks与Hey Next两条
分别转写为Hey Nix及其命令后缀，保守隔离而不重标，原始WAV全部保留。
新增quarantine_negative_aliases.py，核对全量审计覆盖、音频和期望文本。
其余766条保留，包括其他ASR文本不一致者，避免只留下ASR容易的负样本；
ASR只是筛查而非发音真值。来源、审计、隔离报告为
`4090-v20-near-generation.json`、`4090-v20-near-asr-report.json`、
`4090-v20-near-quarantine.json`。

最初未隔离的768条已做特征准备，仅留作数据检查，不训练；最终训练特征
重新从766条准备4种基础渲染，再追加2种MIT RIR/-5..10dB渲染。
原有波形特征和评测行保持不变；最终追加检查/训练尚未完成。


最终v20特征审计通过：原98,948条记录逐项不变，原features.bin完整字节前缀SHA
仍为`f0b193654755aa24c0bc8c52c11062b983cef2e5c0113f875bcfe1aeafdce852`；
human/ambient索引相同。新增766×6=4,596训练负序列，新索引SHA
`20cc1b317346aa0b496de4242ff5dc175e2de81dee7c352635bc61e0d18436bb`。
Piper近音负样本40,596条，新语料占11.3213%；engine等概率时仍只是部分负批量，
没有另加来源权重。审计 `4090-v20-near-feature-audit.json`。

v20已实际启动4090训练，完全沿用v19配方与预训练/随机头初始化：teacher60、
context150、冻结BN、六类ambient分层、prefix16、40k步/每5k验证。
该实验同时改变训练语料内容/分布，单种子不证明普遍因果；当前尚无训练结果。


### v20近音词补充未改善验证，拒绝候选

4090完成40k步397.2s，选择20k检查点。22,801参数，INT8 54,440字节，
SHA256 `5f1da6341ed14219c3ca5af4cb20f3e0a4ba74f527b0ce585f4acacb5fee0290`。
训练报告 `4090-training-v20-near.json`。

原完整INT8验证重新标定阈值0.95，来源平均68.8802%，Piper69.0104%、
Kokoro68.75%，1490/2160，提前事件0，中位延迟150ms、P95 230ms；
最差近音误触1.0%。环境0/1.204811h，95%上界2.48647/h，仍不是0.5/h达标。
完整报告 `4090-quantized-v20-near-validation.json`。
相比v19来源平均78.3073%、中位130ms、最差近音误触0.875%，本候选未改善，
不做开发回归、dev-other复评分或独立最终评分，拒绝v20。

这一结果只否定本次Piper近音短语补充+现有均匀来源采样配置，不能证明
近音负例普遍无用，也不能由未经评分的dev-other推断那一次nets误触已被修复。
已保存合成/ASR/隔离/特征不变性/训练/INT8证据，工作树无待运行任务。
下一步应优先检查训练源覆盖与模型选择的偏差，避免围绕单次验证误触
继续补少量同一TTS短句；保留v19作为研究参考，总目标active，Hi ESP比较未完成。


### 扩展验证标定揭示阈值无法消除的饱和误触

新增calibrate_multicorpus.py，仅用原英文validation网格与dev-other/AISHELL/
DiPCo的validation分数，逐文件核对分数哈希、原事件数和曝光；拒绝开发回归、
最终集、不同模型或平滑窗口的输入。预设门槛为各环境语料分别0事件且近音
最差来源<=1%，再最大化正例来源平均召回，不将不同语料合并掩盖问题。

v19五分数规则在现有0.5..0.994网格内无合格阈值。0.95到0.994之间，
dev-other始终1事件，而正例来源平均由78.3073%降到56.5365%。
未产生新阈值，因此不重跑开发回归。报告 `4090-v19-expanded-calibration.json`。
进一步核对模型输出uint8、scale1/256、zero0，原1255负流的平滑峰值
恰为最大可表示值255/256=0.99609375（4个平滑采样点），阈值等于该值
仍有事件，超过该值则任何模型输出都无法触发。报告
`4090-v19-score-ceiling-diagnostic.json`。这是当前模型/五分数政策的阈值限制，
不是证明唤醒词任务本身无解。

随后预先固定15分数平滑实验（30ms步长），先只跑validation。
评测器新增可配置窗口，默认5保持数值不变；修正平滑后时间戳并在报告中保存窗口，
额外人声评测自动继承冻结窗口；多语料标定拒绝不同窗口混用。
默认数值逐元素一致、常数响应和窗口长度检查通过。15窗口可能增加延迟或漏检，
尚无结果，关键词允许窗口仍-100..+800ms，最终集未访问。


### 15分数平滑失败；启动容量扩展诊断

固定15分数（450ms）实验在完整INT8验证重新标定阈值0.7，来源平均55.3906%，
Piper42.0313%、Kokoro68.75%，972/2160，提前事件0，中位延迟280ms、P95 350ms。
环境0/1.204811h，上界2.48647/h。明显低于原五分数78.3073%、中位130ms，
因此拒绝，不做开发回归或额外语料评分。报告 `4090-v19-smooth15-validation.json`。

阈值和平滑未补救错误，下一步尝试容量而非继续围绕短句/后处理微调。
新增widen_mww_pretrained.py，以Net2Wider式通道复制/输入权重均分，
将30/60通道教师扩大为60/120通道，并先在原16条兼容性探针上验证初始函数。
目前尚未训练宽网络；不以容量扩展或函数接近代替Nyx识别证据。


宽网络实际为70,801参数（原22,801），成功映射全部加权层，原16条目标词/
非目标兼容性探针平均绝对差6.33831e-9、最大5.36442e-7，高响应门槛通过。
权重SHA256 `f00bd48d5a7dde94eaac64cd108fd9db68792b0bdcc6af87e1d05a07367f4337`，
报告 `4090-mww-pretrained-widened-v1.json`。初次扩容误把首层40个输入频带也
翻倍，形状保护立即拒绝；修正为仅复制已扩大的通道轴后通过，失败日志保留。
这不是原教师INT8位精确等效，也不是宽模型流式/量化或Nyx质量证明。
后续训练必须重置分类头，报告资源开销与原始INT8事件指标。

扩展标定防混用实测：开发回归输入被拒绝，15分数英文网格与5分数额外语料
混用也被拒绝，均未写出标定结果。当前无后台任务待完成；本轮明确否定
阈值/15分数补救并准备容量对照，不评分独立最终集，总目标active。


### v21宽网络训练启动

训练器新增teacher120（首卷积60、四块120通道），严格校验已通过原16条
探针的扩容权重SHA `f00bd48d5a7dde94eaac64cd108fd9db68792b0bdcc6af87e1d05a07367f4337`。
RTX4090已实际运行：70,801参数，150帧、冻结BN、同v19组合人声数据、六类ambient
分层、prefix16、40k/每5k验证、seed726与原优化器/LR。没有使用v20近音补充。
加载扩大教师后重置Dense头；宽头尺寸不同，其随机初始权重与v19不相同，
报告 `4090-v21-wide-initialization.json` 保留其哈希，不能声称单独隔离参数数量效应。
采用新随机头也避免全部复制通道与相同头导致永久对称；是否有用须靠实际训练/量化评测。

后处理恢复原5分数（15分数已拒绝），允许关键词窗口不变；最终集未访问。
当前尚无宽模型质量、INT8资源或MCU性能结论。


训练期间复核官方QEMU CPU配置：新增qemu_probe/inspect_cpu.py，machine none、
CPU espressif、暂停状态下仅QMP读取CPU定义与QOM属性，不启动固件。
查询成功，309个属性中无PIE命名属性；标准F/D/V选项不能直接当成Espressif
自定义PIE支持。完整命令、原始响应及QEMU/脚本哈希见
`4090-qemu-cpu-properties.json`。没有推理、也没有证明未来仿真器永远不支持；
此前真实WakeNet创建阶段PIE非法指令仍是运行时阻碍证据。


### v21宽模型训练/原网格INT8验证

40k步442.0s，70,801参数，按原验证规则选25k检查点。INT8 117,464字节，
SHA256 `3779893abd8c58c1377ee59768a6b7b23a85955500487b66f4902b34c794a3de`。
训练报告 `4090-training-v21-wide.json`，浮点所选来源平均80.4427%、阈值0.8。
原11点网格INT8重新标定阈值0.9，来源平均74.5833%，Piper72.9167%、
Kokoro76.25%，1583/2160，提前事件8，中位120ms、P95 200ms；
环境0/1.204811h、95%上界2.48647/h，仍不满足0.5/h。
报告 `4090-quantized-v21-wide-validation.json`。

发现原网格0.8到0.9间隔大：量化0.8来源平均80.2344%但近音最差1.25%超限，
0.9合格但召回74.5833%。为避免只给新模型调参优势，预先固定步长1/256、
范围[0.5,1)、并保留全部旧阈值，对v19和v21都做完整validation-only重新标定。
CLI新增可选threshold-step，默认不变，校验有限值与0.001..0.05范围，
在报告中记录完整网格。密集结果未出，不使用开发回归选阈值；最终集未访问。


### 同网格复核与宽模型扩展验证：没有可接受收益

两模型各用138个固定候选阈值，所有原11个阈值的完整指标逐项精确复现，
确认仅网格改变，无额外推理差异。v19选0.9140625，来源平均81.6146%，
Piper84.0625%、Kokoro79.1667%，提前事件3，最差近音1%；
v21选0.84765625，来源平均78.5677%，Piper80.0521%、Kokoro77.0833%，
提前事件11，最差近音1%，中位120ms/P95 190ms。
各自报告与完整阈值网格见两份`4090-v*-dense-validation.json`及
`4090-v*-dense-threshold-grid.json`。默认旧网格不变，非法step NaN和0
实测在输出目录创建前被拒绝。

v19密集网格配合既有扩展validation仍无零事件工作点，其负例已达到输出上限。
v21在英文密集标定阈值0.84765625下：AISHELL2/6.913985h，95%上界0.910588/h；
dev-other2/5.121185h，上界1.22936/h；DiPCo0/0.754488h，上界3.97055/h。
逐分数哈希、原冻结事件与曝光检查通过。报告三份`4090-v21-*-validation.json`。

随后严格只用这些validation轨迹按既定多语料规则标定，v21在0.98可使
原环境及三种额外语料分别零事件，但正例来源平均降至58.5677%，
Piper45.8854%、Kokoro71.25%，1052/2160；dev-other虽零事件，单侧95%上界
仍0.584969/h，远场3.97055/h，不能据此验收。两份
`4090-v19-dense-expanded-calibration.json`、`4090-v21-dense-expanded-calibration.json`
保存所有候选点，不用开发回归选择阈值、不合并场景掩盖不足。

结论：宽模型参数3.105倍、文件2.158倍，在原英文约束下召回不及同网格窄模型；
能用高阈值移除已知验证误触，但召回代价太大，没有可接受的新候选。
因此不继续开发回归或独立最终评分，不声称容量路线普遍无效。
本轮训练、量化及全部预定validation检查结束，无后台任务待完成。
后续需改变预训练表示/训练来源覆盖与选择方案，而不是继续对同一小样本调阈值；
Hi ESP比较未完成，总目标active。


### v22固定BN统计量、训练仿射参数

此前迁移方案把重建BN的统计量与gamma/beta全部冻结；这保留初始化映射，
但也不允许各通道仿射适配。新增batchnorm_policy.py的affine策略：使用标准
BatchNormalization权重格式，将该实例训练调用固定为training=False，仍保留
可训练gamma/beta；不克隆主模型、不额外初始化随机层，因而保留原随机头与采样。
标准推理/上游流式克隆仍使用相同权重与不变统计量。原freeze-batchnorm行为保留，
冲突参数被拒绝。完整训练在所选检查点再次断言统计量未变、仿射参数确实更新。

probe_bn_affine.py仅用两个原training波形特征窗口核验：初始化inference与
training调用输出逐元素不变；3步后仿射哈希变化、统计量哈希不变；流式克隆
相位对齐后误差0。最初检查在转换后调用原窗口模型，因上游转换会改源层mode
而失败；修正为转换前计算窗口参考后通过，失败日志保留。
诊断报告 `4090-bn-affine-policy-probe.json`，不计为识别质量。

v22已在RTX4090实际启动，同v19的teacher60/context150、原组合人声特征、
六类ambient分层、prefix16、40k/每5k验证与优化器；BN策略改为affine，
仍从原重建权重初始化并重置随机头。Dense初始SHA与v19完全相同：
`a41cc84a5e15a42a13ee9fe02e38242fca4dd9afe67a19ba736587970b46fc54`。
初始化报告 `4090-v22-bn-affine-initialization.json`。后续使用同一密集INT8验证网格，
开发回归不参与阈值选择，最终集未访问。


### 原始说话人映射审计与较大训练语料准备

Piper配置speaker_id_map可追溯至原LibriTTS/LibriSpeech ID。核查官方
SPEAKERS.TXT后，Piper904个声线全在train-clean-360（该集921人），
其中232个为本研究验证/开发回归声线；直接加入整份360会跨身份。
train-clean-100的251人与全部Piper声线无交叉，dev-clean/dev-other/test-clean/
test-other原始ID亦无交叉。保存配置与元数据哈希及完整交叉ID表于
`4090-librispeech-piper-speaker-audit.json`。该核查不覆盖无身份映射的其他TTS。

为补足目前只有约3.225h原始英文真人训练，开始下载官方train-clean-100，
URL https://www.openslr.org/resources/12/train-clean-100.tar.gz ，官方MD5
`2a93770f6d5c6c964bc36631d331a522`。顺序传输实测慢，主动终止自己的curl，
保留所有已收前缀，使用download_libri_training.py的8路16MiB范围请求：
每段校验206/Content-Range/长度，组装时前缀再次哈希，最终校验MD5和总长度，
保留逐段SHA与档案SHA。当前仍下载，尚未解压/提特征/参与训练或评分。
新增语料将明确仅训练；实际文件身份/转写/连续前端仍需通过后续审计。


### 官方最佳实践复核（2026-11-10）

见 `BEST_PRACTICES.md` 与 `results/4090-best-practices-sources.json`。官方确认纯TTS与无板离线训练路线成立；主要调整优先级为扩大真实负语音/远场覆盖，而非继续孤立加宽模型。openWakeWord约30000h负样本是其发布模型自述规模，不是本项目强制目标；MCU部署仍以microWakeWord为主。最终集保持封存，Hi ESP对比未完成。

v22完成40k，选择10k：BN统计哈希保持不变，gamma/beta已更新。密集INT8验证阈值0.92578125，source-macro召回72.5521%（Piper68.4375%、Kokoro76.6667%），0早触发；低于v19同网格81.6146%，拒绝扩展评测，不访问最终集。发布训练与验证报告。

### train-clean-100训练专用提取准备

新增prepare_libri_training.py：官方MD5后才解压；禁止路径逃逸和链接/设备；
要求官方元数据与实际目录均为251人，排除既有人声/Piper/最终manifest身份。
仅访问最终manifest的ID，不访问最终音频。转写与FLAC必须一一对应、无重复，
目标短语字面命中则停止待审计。逐FLAC/PCM与特征哈希、原始时长、插入静音
分别记录；输出全部training，category=unspecified并入现有英文人声采样层。
复用开发语料worker但保留三参数默认行为；用原训练2428的两个FLAC共15.375s
核验新旧路径特征逐元素一致、首两段连续前端一致，见
`results/4090-libri-training-worker-probe.json`。此为数据管线验证，不是模型质量。
档案仍下载，当前未将新增语料加入训练；完整特征生成需等档案验证通过。

### v23预注册：固定模型，扩大独立英文负语音

`experiments/v23_librispeech100.sh`固定v19模型/初始化/冻结BN/prefix16/40k与验证协议，
只加入审计后的train-clean-100训练负语音；不沿用失败的v22 affine设置。
准备完成后才执行，append再次核对索引和各流特征哈希、保留原评测行及二进制。
新增语料并入human:unspecified，英文类别总概率仍1/6；预计40k步抽32万英文窗口，
重叠计时约133.33h，不是独立曝光或覆盖保证。阈值仍在原验证集密集网格选择；
只有值得继续的候选才跑扩展负集验证，最终集不在runner内。
预注册 `results/4090-v23-librispeech100-plan.json`，不计为实验结果。

### train-clean-100完整下载校验通过

373个范围块完成，完整档案6387309499字节，官方MD5
`2a93770f6d5c6c964bc36631d331a522`匹配，SHA256
`d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`。
逐范围SHA与来源见 `results/4090-librispeech100-download.json`。
提取脚本已通过身份集合、转写/音频对应与目标字面筛查，进入251人连续前端
特征生成。此时尚无完整特征cache、未开始v23训练，不作识别效果结论。

### v23数据审计完成，4090训练启动

251人全部提取，原始100.59087965h，全部training、category unspecified，
索引SHA `aea736f8ad92748fa32798aa7af0006f8fa417e792968bf972b223cb05aac6aa`。
每流前两段连续性校验通过；完整逐FLAC/PCM来源留在本地哈希索引，
发布摘要与251流特征哈希。append核实旧验证/开发回归行和关键词特征不变。
v23已在4090开始训练；初始权重和随机分类头SHA与v19一致，BN frozen。
发布 `4090-librispeech100-training-features.json`、
`4090-v23-librispeech100-feature-addition.json`、
`4090-v23-librispeech100-initialization.json`。尚无最终训练/INT8评分结论。

### v23训练与原验证结果：召回退化，正在核查误唤醒

40k完成426.4s、选40k，BN统计/仿射哈希均不变。INT8模型SHA
`061c4a4a378b8dac2d0e13a8e6d53739b469fcb889565f5c1e7db9fdb6695236`。
密集INT8验证阈值0.94140625、source-macro71.171875%（Piper81.9271%、
Kokoro60.4167%），早触发1，延迟中位120/P95190ms，原ambient零事件
但只有1.2048h，上界2.486/h；不合格。低于v19密集验证81.6146%。
小验证子集流式/整窗相位对齐、float/INT8显式重置通过；平均score漂移
0.00145534、最大0.152816，仅诊断。该probe采用训练浮点验证0.95，
不是后续INT8工作点，不据其4个正例宣称召回。
为检验新增负语音是否消除v19的饱和误报，仍在冻结INT8阈值运行
dev-other/Mandarin/DiPCo验证；这检验本轮负数据假设，不改变成功标准。
未运行开发回归或最终集。

### v23扩展验证失败；v24检验较长优化

v23冻结阈值0.94140625：dev-other4/5.121185h，上界1.78738/h；
Mandarin3/6.913985h，上界1.12145/h；DiPCo2/0.754488h，上界8.34446/h。
原1255位置仍334.61s触发。密集扩展验证无零事件可用阈值，拒绝v23。
数据扩大本身未解决问题，不使用最终集掩盖开发失败。

v23末段浮点macro25k57.9167%、30k62.5%、35k64.5313%、40k69.375%，
最优在末步，尚不能排除大语料欠优化。预注册v24从相同初始化训练120k，
沿用相对LR计划（降点20k/32k改为60k/96k），其余数据/架构/采样/
验证间隔相同。这测试更长训练及对应LR计划，不是固定LR下的纯步数消融。
不保证改善，不改变误唤醒及Hi ESP目标。

### Hi ESP无板运行突破：经典ESP32实际完成detect

官方Xtensa QEMU同版esp-develop-9.2.2-20261109包含esp32目标和PSRAM设备。
ESP-SR2.6.1的esp32/libwakenet.a实际包含wn9/9l/9s符号，虽然HiESP菜单
依赖仅S3/P4/S31，官方ESP_SR_EXTERNAL_MODEL_PATH可原样打包wn9_hiesp。
新探针用经典ESP32构建，不改模型/推理库，无P4寄存器RAM或FPU补丁。

正常探测4MB模拟PSRAM、初始化wn9_hiesp：512samples、16kHz、mono、
threshold0.63(DET_MODE_95)。两次独立启动均完成100次真正detect(零PCM)，
所有返回0，monitor读回stage104验证destroy/deinit已完成，flashSHA一致。
发布qemu_esp32_probe源码、两次UART与报告、库/模型/镜像哈希。

这解决经典ESP32路径的零音频推理兼容，不解决P4 PIE。尚未输入真实
语音波形，不是召回/误唤醒证据；模拟器host耗时不作为硬件延迟。
HiESP质量比较仍未完成，但已有可执行引擎，可继续建设真实PCM回放协议。

### HiESP真实PCM回放与同音频开发链路打通

新增qemu_esp32_replay，采用QEMU Xtensa simcall只读文件，不使用不兼容的
OpenOCD semihost ABI。每次从新模型状态按512样本送真实16k/PCM16，
核对源PCM SHA、完整读取CRC32、调用次数、尾段补零和stage104。
6段训练声线开发TTS均CRC一致；两Hi E S P提示各触发1次（2.784/2.976s
音频时间），Hey Nicks与普通句各2段不触发HiESP。首正例两次启动事件与CRC一致。
同一PCM送Nyx v19原验证阈值0.9140625，只有两个Hey Nicks提示触发
（2.75/2.81s音频时间）。两引擎保留各自前端；仅开发功能协议，无匹配FAR。

调试暴露setter请求0.63而getter0.62496001，分别保存不假定相等；
detect -1为合法CHANNEL_VERIFIED，单独计数；monitor初始提示拆包曾造成
完成标志误判，修复为先消费完整初始prompt。代表性失败UART亦发布。
没有语音发音审计、足量负音频标定或真实用户正例，不报告总体召回/FAR/
硬件延迟，更不宣称Nyx等同HiESP。最终集未访问。v24仍正常训练。

### HiESP连续DiPCo验证：与Nyx核验同源PCM

完整S05_U04.CH1.wav 2716.15625s连续送入HiESP，整段不切断/重置；
43458500样本、84880次detect、CRC32与宿主一致、stage104完成，0事件。
原始PCM SHA `396ce6d7dfe4fefeb74f9e5e0c8d3783baa2583d6af0c64e5c77ac9270c6b28a`，
加60样本末尾对齐后等于既有Nyx前端stream_pcm_sha，特征/原score哈希和
事件计数重算通过。v19冻结0.95亦0事件；v23冻结0.94140625为2事件。
这不是匹配总体FAR的质量结论；仅0.754488h，零事件上界3.97055/h，
不足0.5/h，且该开发会话已多次使用。没有正例总体比较或最终集评分。
新增prepare_hiesp_negative.py仅解码既有validation；metadata可读取其他split，
不读其音频。回放增加可配置host超时和每分钟音频进度，不改detect算术。
发布manifest、完整UART、回放与同PCM比较报告。v24继续训练。

### v24完成120k：原INT8验证改善，但尚无负集结论

1280.7s完成，按原验证规则选110k；初始分类头及预训练来源保持一致，
BN frozen校验通过。INT8 SHA
`21e605dfc061fdc0823b4ab75a5be593955dd747103f33b0ccedea2e673ba825`。
密集阈值0.9765625，source-macro79.6875%（Piper79.375%、Kokoro80%），
早触发2，中位120/P95180ms。好于v23的71.1719%，仍低于v19的81.6146%；
原ambient零事件但1.2048h上界2.486/h，不作为验收。
三个额外开发负集和小样本流式一致性已运行，尚待结果；不访问开发回归/最终集。

### v24扩展验证失败，停止延长训练假设

冻结INT8阈值0.9765625：Mandarin0/6.913985h上界0.433286/h，
DiPCo0/0.754488h上界3.97055/h，dev-other3/5.121185h上界1.51404/h。
116(577.07s)、1630(488.27s)、7601(471.2s)触发；116和7601平滑分数
达到255/256上限。1255原v19误报位置在此工作点不再触发，但新增误报
使总体仍失败。密集扩展标定无可用零事件工作点，不通过加阈值规避。
小验证导出重置通过，score平均漂移0.000984/max0.237896，仅小样本诊断。
不进入开发回归或最终集。与HiESP同源DiPCo比较增加v24，二者本会话
均0事件仍上界3.97055/h，不等于总体性能相当。

100h数据+120k步改善v23召回但不足以解决真实负音频饱和误报；下一步
优先在training专用真实语音池挖高分难负窗口，不继续孤立增加步数。
验证误报只诊断，绝不将这些验证音频/窗口加入训练。

### v25训练真实语音难负例：挖掘与审计完成

固定v24 INT8扫描已审计train-clean-100全部251个training流100.59088h，
非验证/最终；5分数平滑>=0.5、候选至少约2s间隔、每流最多64，选157个
窗口/105人。逐流score SHA与候选frame_end/转写/来源哈希均保存。
候选不是识别质量或新独立曝光。源时间线与已插入200ms/对齐规则核对。

审计扩展到+/-12帧抖动的完整输入足迹，包含跨句拼接；199个FLAC与
解码PCM逐个核对。原人工转写字面别名筛查隔离0个，但不是音素真值。
HardAmbient只允许canonical training行、同来源索引/特征哈希和合法窗口；
实际抽128个窗口重复种子完全一致，伪造validation记录被拒绝。

v25预注册：同v24数据/模型/初始化/120k/LR计划/验证，仅每批48环境槽
替换4个为审计难例，独立RNG30726，原draw不变，label0 weight20，
剩44广覆盖槽。故英文环境槽概率从1/6改为23.6111%，其他各15.2778%，
是有意强化真实近音负例，不宣称采样层权重不变。代表性INT8校准沿原
广覆盖训练采样，禁用hard替换。最终集未读取。


## 配对开发诊断：v19 与官方 Hi ESP（2026-11-10）

72条相同PCM、原生前端、每条重置；4个已用验证声线各3速度，
每引擎12条目标、48条共同负例、12条另一唤醒词负例。
Nyx使用既有密集验证阈值0.9140625；Hi ESP请求0.63、读回0.62496001，
二者不是匹配一般误触率的工作点。
固定能量代理端点-100..+800ms窗口内：Nyx9/12，Hi ESP1/12。
Hi ESP另外4条在窗口前触发，故目标音频上有触发的实际是5/12，不能把其余11条都描述为引擎完全没响应。
两者共同负例和另一唤醒词负例均零事件。Nyx命中延迟中位140ms、P95252ms；
Hi ESP唯一窗口内命中-62ms。仅为能量代理、音频时间，不是音素/硬件延迟。

严格ASR中Hi ESP含High ESP等同音拼写及可疑字母读法；没有据此删样本、
改别名或事后移动窗口。这组小样本不能证明Nyx超过Hi ESP，也不是代表性召回。
完整原始事件及逐条分类见results/4090-paired-development-v19-hiesp-v1.json。

v25继续训练，新增experiments/v25_validate_expanded.sh用于训练及密集验证成功后，
执行中文、DiPCo、dev-other验证、扩展工作点标定和小样本浮点/INT8流式一致性检查。
全部限定既有validation，独立最终集继续封存。


## Hi ESP中文验证流重建（2026-11-10）

从既有AISHELL审计manifest仅选16位validation说话人，重建6.913985191小时
源朗读音频；每句后200ms静音和10ms对齐严格复现Nyx前端PCM，
全部源WAV/PCM和原特征文件哈希通过。插入静音不增加曝光分母，
只在说话人之间重置。training/testing音频未读取，独立最终集未访问。
重建摘要见results/4090-hiesp-aishell-reconstruction-v1.json。
官方Hi ESP请求阈值0.63的QEMU回放已启动，当前尚无完成分数；
不得将重建成功当作零误触或验收通过。


## Hi ESP中文回放完成与同PCM审计（2026-11-10）

16/16说话人完成，全部CRC和stage104通过，0运行失败。6.913985191小时
源音频出现1次Hi ESP事件：aishell_S0044，流内261.792秒；
误触率0.144634386/h，单侧95%上界0.686125930/h。
相同PCM审计、原特征哈希和Nyx分数哈希/事件重计数全部通过；
Nyx v19原阈值0.95为0事件，v23阈值0.94140625为3事件，
v24阈值0.9765625为0事件。Nyx零事件上界0.433285897/h。
这些是既有中文朗读验证结果，不能外推一般场景或证明Nyx达到Hi ESP；
各模型工作点并未按一般误触率匹配，正样本比较和独立验收仍未完成。

证据：results/4090-hiesp-aishell-validation-v1.json和
results/4090-hiesp-nyx-aishell-pcm-audit-v1.json。
审计工具同时识别早期报告显式validation角色及combined exclusion flag，
未知/回归/最终报告布局仍拒绝；未改变原分数或阈值。


## v25困难环境负例实验：拒绝（2026-11-10）

120000步，1192.3秒，选110000步；INT8 SHA256
`13f26513d9e44d8586f502f32b56710f5010a06b70a5d6fccef5bce0d8a4dcb8`。
使用与v24相同预训练初始化/冻结BN/架构，在48个ambient槽位中
用训练专用审计困难负例替换4个；不读取验证音频来挖掘或训练。

密集INT8验证阈值0.994，来源平均召回70.2604%（Piper73.0208%、
Kokoro67.5%），低于v24的79.6875%；提前事件2，命中延迟中位130ms、P95190ms。
原环境1.204811小时0事件，上界2.48647/h。扩展验证：
中文6.913985小时1事件，上界0.686126/h；
DiPCo0.754488小时1事件，上界6.287529/h；
dev-other5.121185小时2事件，上界1.229363/h。
扩展密集标定没有符合既定零事件条件的工作点，threshold=null，不回退伪造可用点。

小样本浮点/INT8流式重置通过；平均量化分数偏差0.00181846、
最大0.31983246。该探针不代表全语料或MCU数值等价。
完整配置、原验证、阈值网格、三负语料、扩展标定、量化探针
见results/4090-v25-*.json。候选拒绝，不进行开发回归或独立最终评分。
困难负例只选出偏置样本，不增加独立音频小时；当前比例从头重训
没有解决跨语料误触且降低召回，不能据此声称困难负例方法普遍无效。


## v25饱和分数诊断与v26计划（2026-11-10）

使用已验证phase1对齐的150帧浮点窗口，检查四个现有验证误触流的
INT8平滑峰值对应5个原始窗口。两个英文峰值浮点均值为0.999001和
0.999888，证明这些负例的过度自信不是仅由INT8舍入产生。
中文峰值浮点均值0.991856、INT8为0.994531；DiPCo浮点均值0.991146、
INT8达0.996094，显示局部量化上移也值得关注。
该诊断不替代完整浮点流式重计数，未重新选阈值/修改标签。
见results/4090-v25-float-peak-audit.json。

v26以v24为基线，只把训练BCE改为label_smoothing=0.02，
即0/1训练目标变为0.01/0.99，原样本惩罚权重不变；
其余架构、数据、初始化、冻结BN、prefix16、120000步和学习率计划相同，
没有v25的hard ambient替换。默认epsilon0保留既有损失路径。
这仅是减少过度自信的受控假设，未证明改善；不将soft labels用于评测。
完整预定计划见results/4090-v26-plan.json，运行脚本
experiments/v26_label_smoothing.sh。独立最终集不评分。


## 多语料开发误触描述性汇总（2026-11-10）

仅合并已逐PCM审计的dev-other、AISHELL validation和DiPCo S05，
共50个说话人/会话、12.789657865小时源音频。三项Hi ESP回放模型、
阈值请求/读回、ELF、flash、QEMU哈希一致；无重复说话人/会话，
所有CRC/stage104与分母、计数复核通过。

固定原阈值：Hi ESP1事件，Nyx v19 1事件，v23 9事件，v24 3事件。
前两者描述性合并上界均0.370914106/h；v24上界0.606244249/h。
这不改变先前候选拒绝：重复开发数据不是独立验收，
相等的观测事件数也不是一般场景误触率等效证明。尤其远场只有0.754488小时，
合并不能掩盖Hi ESP中文1事件或Nyx英文饱和事件。
保留各语料结果和既有逐语料零事件选模门槛；没有据此重选阈值，
也没有以此触碰最终集。证据见results/4090-hiesp-nyx-pooled-development-v1.json。

v26继续运行，扩展验证脚本experiments/v26_validate_expanded.sh
在训练/原密集验证完成后执行，同样不读取独立最终集。


## 饱和峰值诊断的实际流式交叉核验（2026-11-10）

为排除用非流式窗口解释流式峰值时的相位误差，对v25四个既有误触流
最大分数位置，重放600帧模型预热后的实际SavedModel流式输出与INT8输出。
保持既有连续前端特征，浮点5输出与phase1窗口逐项误差≤1e-6，
量化5输出平均与原完整流分数峰值精确重现（容差1e-12），四项均通过。
这加强“两个英文负例在浮点阶段已近1”的局部归因；不是全语料浮点误触计数。
见results/4090-v25-float-peak-audit-v2.json，旧v1证据继续保留。

v26与v24初始化JSON内容及哈希一致，显式记录在
results/4090-v26-initialization-audit.json。本轮未改变运行中的训练代码、
样本或验证协议，尚无v26最终识别质量结果。


## 训练专用远场麦克风视角补充（2026-11-10）

固定选择既有训练会话S02/S04的U02.CH1和U03.CH1，
从原固定DiPCo revision下载4条通道，共2.531006944小时通道波形。
原LFS大小/SHA、16k单声道PCM16、解码PCM哈希、会话持续时间、
原转写及参与者训练划分全部核验。未访问验证/回归/最终音频。
这些是同两场对话的相关麦克风视角：新增物理说话人0、新增独立对话小时0。
不能将通道数或通道小时当作独立训练/验收覆盖。

同一10ms MicroFrontend连续提取完成，911156帧，逐流前两块连续性核验通过。
元数据保留same_conversation_as/independent_conversation，
另建features-gpu-spatial-v1候选缓存；关键词特征及validation/testing行不变。
原v26数据没有修改，当前尚未用此缓存训练或评分。
后续若试验，应保持远场类别总采样质量固定，仅在类别内增加位置多样性，
是否有效仍需受控流式评估。

results/4090-spatial-training-*.json保存源、缓存、组合和相关性审计。
通用追加报告的added_speakers是流ID，不代表新增真人；
added_original_hours为原通道波形小时，不代表独立对话小时。


## 与Hi ESP观测事件预算一致的开发诊断（2026-11-10）

作为独立诊断，保留原环境0事件、近音来源误检≤1%约束，
用固定Hi ESP在同批12.789657865小时负音频的1事件作为观测预算，
扫描v19已有固定密集阈值网格，最大化原英语验证来源平均召回。
三项负语料报告SHA必须匹配既有同PCM审计，逐流score哈希/原事件重计数通过。
选择阈值0.94140625：中文0、DiPCo0、dev-other1事件；
合并观测数与Hi ESP相等，名义单侧95%上界0.370914106/h。
原合成验证来源平均召回79.2708%，中位延迟130ms、P95210ms。

此点**没有通过既有逐语料零事件门槛**。诊断独立保存并标记
not_for_final_acceptance，未替代原门槛、未授权最终测试。
相等开发观测数不是一般场景FAR等效，也不是独立置信保证。

随后仅从已有配对开发分数重新计数，未用其正样本选阈值，
窗口不变：Nyx9/12命中，中位150ms、P95266ms；
Hi ESP仍1/12窗口内命中，另4条提前触发。共同/交叉目标负例均0。
这些是重复合成小样本，Hi ESP字母发音/ASR和能量端点限制仍在，
不能声称Nyx优于Hi ESP。完整曲线与逐条事件见
results/4090-v19-hiesp-development-calibration-v1.json及
results/4090-v19-hiesp-paired-budget-diagnostic-v2.json。


## 正/近音样本的真实对话背景增强准备（2026-11-10）

核查原MS-SNSD训练背景仅45条设备/家具声音，没有叠加真实对话。
原环境负例虽有DiPCo，不能代替正样本在交谈背景中的训练覆盖。
另建features-gpu-dialogue-mix-v1：仅使用已审计S02/S04训练通道，
为每个原始training关键词/近音来源增加两个确定性混音，新增29196序列。
前景整段RMS标定，5..15dB SNR，1.8..4.5s对话预热、1s尾部；
不改端点代理，不修改标签，不给新增混音再叠RIR，原RIR变体保留。
背景随机片段可能包含停顿，不能称每条都有重叠语音。

完整原特征二进制前缀SHA、全部原索引前缀及validation/testing行核验通过。
Piper/Kokoro/Qwen各正负1条共6个实际重渲染探针，
波形哈希、元数据及特征数组与缓存逐项一致。
最大裁剪比例1.4315%，平均0.1067%，已报告而非静默忽略。
新增独立对话小时0；当前运行v26数据完全不变，尚未基于此缓存训练/评分。

记录在results/4090-dialogue-augmentation-v1.json及
results/4090-dialogue-augmentation-audit-v1.json。未来训练配置会记录
dialogue-augmentation.json哈希；该记录项不影响已启动v26的采样/损失。


## 官方Hi ESP正控制与端点代理反证（2026-11-10）

同一ESP-SR提交b9dc43a提供test_apps/esp-sr-wakenet/main/samples/hiesp.h，
官方test_wakenet.cpp明确把此数组作为int16输入detect。提取62464字节
（31232样本、1.952s），保留所有源PCM，构造0/1.8s前导及1s尾静音两个控制。
两个控制均在既有QEMU/未改模型/阈值请求0.63下发出1事件：
1.344s与3.072s，CRC和stage104全部通过。未知录音/模型开发来源，
不能计独立说话人、召回或验收。

重要：同一能量代理算法给出结束点2.01s和3.81s，
触发相对代理为-666ms和-738ms，因此原固定窗口会将它们排除。
这不是模型完全没有响应，也不是已知音素延迟。现有配对集
Hi ESP有原生事件的目标音频为5/12，固定代理窗口内为1/12；
Nyx为9/12。它们仍不能用于代表性真实召回/延迟排名。
原结果、窗口、标签和原始事件全部保留，不为提高分数移动窗口。
后续需要独立于唤醒分数、经过核验的端点标注方案。

源URL/哈希、提取代码、两个完整回放和端点反证见
results/4090-hiesp-official-control-*.json；
历史结果资格说明见results/4090-paired-endpoint-qualification-v1.json。


## v26标签平滑：出现零事件开发工作点，但召回代价明显（2026-11-10）

120000步、1249.9s，选择55000步；INT8 SHA256
`96108b06672edd55268e2c3e1e7d02baf501722cc19df67a27b118ff6eba14a2`。
训练BCE平滑0.02；其余v24架构/初始化/冻结BN/数据/prefix16一致，
无hard ambient替换。原密集阈值0.828125，来源平均代理窗口命中78.8021%
（Piper80.5208%、Kokoro77.0833%），原环境0事件，提前1。
扩展中文1、DiPCo1、dev-other2事件。

扩展同一密集网格选择0.96875后，原环境与三项扩展负语料分别均0事件，
但来源平均代理窗口命中降至56.40625%（Piper48.2292%、Kokoro64.5833%）；
总命中1081/2160，代理延迟中位160ms、P95260ms，无提前事件。
中文零事件上界0.433286/h，dev-other0.584969/h，DiPCo3.970551/h，
原环境2.486474/h，各场景曝光局限不因零事件而消失。
这满足既有开发零事件选模条件，尚未通过独立质量验收；
不能把低代理窗口命中、合成集或未核验端点当作超过Hi ESP的证据。

小样本重置/相位检查通过，量化分数平均偏差0.00698635、最大0.28047830；
原工作点及扩展工作点各保存一份探针。
完整训练、阈值网格、负语料和扩展标定见results/4090-v26-*.json。
不做独立最终评分，保留候选与负面取舍，不将标签平滑宣称为已解决问题。

下一项v27与v24只差训练对话混音数据，仍为原BCE、无hard ambient/
无标签平滑/无新增空间通道负例。两项干预分开比较，
预定计划与runner已保存，仍按同一验证流程判断。


## 独立于唤醒分数的MFA对齐诊断（2026-11-10）

为处理能量代理端点反证，按官方方法安装CPU MFA3.4.3、
Kaldi5.5.1172 CPU版本，使用SQLite、4作业、single_speaker禁用说话人自适应。
显式包锁含289个固定包URL/MD5；未改全局PyTorch或当前v27数据。
固定MFA english_us_arpa v3.0.0声学模型及词典（下载SHA已保存），
输入仅24条既有配对开发目标和2条官方控制，意图文本固定HEY NICKS/HI E S P，
没有读取唤醒分数、ASR筛除样本或修改旧窗口。

26/26生成TextGrid及alignment_analysis.csv，词序列均完整，耗时43.181s。
results/4090-mfa-development-alignment-v1.json完整保留词/音素区间、
原始质量指标及各文件SHA；不将未标定似然或自动边界当作人工音素真值。
当前没有基于这些边界重算/宣布召回或延迟，也未触碰最终集。
官方控制首条自动末词p结束1.52s，原能量代理2.01s；
这支持继续检查端点协议，但不自动使任何旧触发成为正确命中。

复现脚本experiments/mfa_development_alignment.sh、CPU显式锁、
模型下载器和输入/工具来源报告均已保存。后续需要独立质量核验，
尤其字母词发音及短音素，才可决定可用于什么级别的时延比较。


## 成熟实践复核及 MFA 平移失败（2026-11-10）

一手依据及具体差距见 [BEST_PRACTICES_EVALUATION.md](BEST_PRACTICES_EVALUATION.md)，固定来源与响应哈希
见 results/4090-best-practice-sources-v1.json。TTS 训练/MCU 流式/INT8 路线已有成熟实现，
但不意味着任意新词自动达到商业引擎；上游也明确提醒定制模型仍需大量实验。
优先使用代表性连续负音频、同误唤醒点比较、独立标注与冻结最终评测，
不把更多训练步数、同数值阈值或自动能量端点当作质量证据。

固定 500ms 零前缀与 20ms 容差，26/26 MFA 复跑完成；
3 段末端/末音素边界超容差（-80、-30、-40ms）。
最大 80ms，失败保留；没有改旧窗口、读唤醒分数或最终集。
完整输入、对齐与审计见 results/4090-mfa-*shift500* / 4090-mfa-shift500-*。
自动对齐尚不能视为音素真值，也未据此重新发布延迟胜负。


## v26 同 Hi ESP 观察事件预算诊断（2026-11-10）

已对 AISHELL、DiPCo、Libri dev-other 三项分别重做同 PCM、特征、
完整 QEMU CRC/stage104 和 Nyx 分数哈希审计；不重新运行或改变 Hi ESP。
v26 原阈值0.828125分别1/1/2事件，共4/12.789657865h，
Hi ESP分别1/0/0，共1。原点名义合并上界0.715697/h。

按既有事件预算诊断规则（不替换零事件候选门槛），
v26选择0.94140625，分别1/0/0，与 Hi ESP观察总数相同。
其原验证来源平均代理窗口命中65.33854%（Piper61.92708%、Kokoro68.75%），
总1354/2160；代理中位150ms/P95250ms，仅记录历史窗口语义。
v19同预算79.27083%，因此标签平滑并未改善该开发取舍，不推进最终评分。
两者同总数的名义合并上界0.370914/h不代表独立达标或总体误触率相等。
完整来源、网格与每语料计数见 results/4090-v26-hiesp-*.json。


## v27 对话混音完成与导出初态诊断（2026-11-10）

120000步，1266.8s，选65000步；INT8 54440字节，SHA256
`b6ff9666312eb0563da67b9479ef0f70c5a350202107db9d25dd75088be03f76`。
仅改变训练对话混音，原BCE/初始化/架构/冻结BN/prefix16保留。
原密集阈值0.95703125，来源平均代理窗口命中72.34375%，
Piper76.35417%、Kokoro68.33333%，1630/2160，提前2；
原环境0、中文0、DiPCo1、dev-other0。DiPCo峰0.96484375，1861.49s。

扩展零事件阈值0.96875，来源平均代理窗口命中70.0%
（Piper73.75%、Kokoro66.25%），1575/2160，提前1。
原环境及三项新增语料分别0事件，但各语料曝光上界仍不足以全部达到0.5/h。
代理延迟中位130ms/P95230ms，端点语义限制保留，不能解释为音素延迟优胜。
同PCM审计已通过；与Hi ESP相同总1事件预算的诊断仍选原点0.95703125、
72.34375%，低于v19该诊断79.27083%，但比v26零事件点56.40625%改善。
不推进最终验收，不改变既有门槛。

原8段流式探针重置/相位通过；分数平均偏差0.03230367，最大0.87761802。
新诊断保留所有开头分数/事件，最大差都在第7步（音频0.26s）；
完整150帧+相位上下文后平均0.00105660、最大0.22486654。
扩展点有1段浮点/量化触发差30ms（110 vs109步），不能宣称逐事件一致。

SavedModel有4个非零初始缓冲：浮点强制零态与导出初态确实不同；
改用相同导出初态后最大偏差仍0.76837753，说明初态差异不能解释全部问题。
TFLite初始化子图有显式量化常数，零值编码-128，并非仅凭整数零重置的推测。
下一步应受控核查代表性量化数据：现导出强制全0/全26两帧，
全26并非真实语音特征分布，可能放大内部量化范围；这是待验证假设。
完整原报告/新诊断见results/4090-v27-*.json，历史数据不覆盖。


## 新增独立 AMI 训练会议（2026-11-10）

已下载官方AMI单通道远场Array1-01四场：ES2002a/ES2005a/IS1000a/TS3005a，
共1.292257795小时、16个不同参与者。分别同时满足Kaldi固定revision的train划分
和AMI原始seen_type=training；参与者与两种划分的dev/eval并集不交叉。
初选ES2003a虽在Kaldi train，但AMI原标development，故仅凭元数据排除，
未下载/评分其音频。四场训练转写逐说话人核验及字面目标别名筛查已完成。

来源为AMI Meeting Corpus（https://groups.inf.ed.ac.uk/ami/corpus/），CC BY4.0。
注释v1.6.2和Kaldi划分固定SHA；音频SHA为首次本地完整下载记录，
不是声称官方提供的校验和，复现可用--expected-audit固定这些音频哈希。
不声称跨语料生物身份已验证，仅保证AMI元数据参与者划分；
字面筛查也不等同穷尽同音词或发音真值。未读保留集音频或唤醒分数。
源码training/prepare_ami_training.py，来源及清单见results/4090-ami-training-*。
特征提取使用连续MicroFrontend，无插入间隔，尚未加入任何已训练模型。


AMI连续特征已完成：465206帧，四条流的前两块连续性检查全部通过，
没有人工间隔；仅末尾不足10ms对齐填充不计曝光。
已构建features-gpu-dialogue-ami-v1候选，关键词二进制和所有验证/回归行均保持不变；
新增四条流均为训练专用farfield-dialogue类别。尚未开始该数据候选的训练。

量化校准对照已导出两份相同权重模型：固定144段真实训练特征
（12组每组12段，各150帧，真实范围0..25.7421875），对照组先喂全0/全26，
处理组省略两条人为极值。两者均保持同一个SavedModel与保存初态。
原导出另有不同校准采样，不能将它与新模型直接作单变量因果对照。
新模型分别进行原密集验证；不提前假定省略极值会解决初态或识别问题。


## v27 固定真实校准样本的导出对照结果（2026-11-10）

新对照(全0/全26前缀)SHA f8e16775726ccb3085c86971538319cbdfdd4a722dc54313b874d43a9720369b；
仅真实代表样本SHA d36e8a986386bb7010ba8f057e8a9116ffbb496874a84fc805401d3c49eb3730。
二者均54440字节，无权重重训。固定144段特征、保存状态及图，
实际特征范围0..25.7421875。只省略人为极值两帧是两份新模型间唯一干预。

原验证密集点分别0.89453125/0.87109375，来源平均代理窗口命中
77.03125%/78.38542%；三项负语料两者均0/1/0。
扩展零事件点分别0.984375/0.98，来源平均代理命中
63.61979%/65.59896%（1393/1455个，共2160变体）。
去极值在本次固定采样下略改善，但两者均低于原v27零事件点70%，不替换。
新8段探针最大偏差均0.47584814；完整上下文后最大0.24049154/0.13111654。
全程保留冷启动数据，不以改善后的局部偏差证明整体质量。
results/4090-v27-calibration-*-v1.json保留完整来源、网格及每语料计数。

下一项v28只在v27训练数据中追加已审计AMI会议，
保持原始导出校准流程以隔离数据干预，继续120000步、相同初始化、
冻结BN、原BCE和prefix16。仍按同一密集/扩展零事件开发门槛，最终集不评分。
runner见experiments/v28_ami_dialogue.sh，未预判结果。


## Kokoro 原生时长输出与旧音频对应失败（2026-11-10）

实际检查固定Kokoro ONNX图，输出为waveform和duration，旧封装只返回第一个。
读取原生token时长，对24条原有开发目标重新合成；
每条均验证原生duration总和×600等于24kHz未裁剪波形长度，
记录实际库trim偏移、全部token/时长以及新音频SHA。没有调用唤醒分类器。

然而24/24新波形均不等于原保存PCM，其中3条长度也不同。
不能将本次推理时长回填到旧音频：所有original_pcm_candidate_end_seconds均为null。
完整失败保留，不按匹配程度筛选；新诊断波形在
.local/wakeword/synthesis-duration-development-v1（未评分），
原生时长与失配报告见results/4090-synthesis-duration-development-v1.json。
候选仅属于本次保存的新波形，也不构成真实音素标注：
原生duration含重音/空格token分配，声码器时移及听觉边界仍未核验。
因此没有凭此重算召回或宣布延迟优势。源码training/audit_synthesis_durations.py。

v28已启动实际CUDA训练（PID210525），运行初期1500步，
初始化JSON与v27逐项完全相同。训练完成后runner自动执行
原密集验证、三项扩展负语料、扩展校准及流式探针；不触碰独立最终集。


## 流式初始常数及外部音素参考核查（2026-11-10）

直接解析原v27及校准对照两份TFLite的资源初始化子图，
通过VAR_HANDLE/ASSIGN_VARIABLE匹配全部6个SavedModel状态缓冲。
三份模型所有初始化常数都与浮点值的最近INT8量化结果逐元素相同，
最大初值偏差均不超过该状态量化步长的一半。
这排除了“导出常数被错误清零”这一具体解释，
不等于运行时重置、量化推理或MCU正确性证明，冷启动分数偏差仍保留。
可复现审计training/audit_export_initializers.py，
results/4090-v27-*export-initializers-v1.json，未读音频或唤醒分数。

为独立核验MFA，读取LDC官方公开TIMIT样例WAV/.phn/.txt
（https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav 等）。
仅作为非唤醒工具质量控制，不进入训练、选阈值或独立验收集。
46797个16kHz样本，公开末非静音axr结束于44586样本，即2.786625s；
公开音素表末端46720与WAV差77样本，原样保留并记录。

按事先20ms容差和相同english_us_arpa3.0.0配置复跑，
MFA末词/末音素R结束2.84s，较公开参考晚53.375ms，**未通过20ms容差**。
词序列全部保留，运行32.685s；请求4作业，但MFA因只有1条自动使用1作业。
不使用这一个样例推导总体误差、给旧边界减53ms或调整Nyx/HiESP窗口。
参考axr与MFA音素表不同，比较仅为同一末端边界，不声称全音素对齐一致。
这为“自动对齐不能直接当作精确音素真值”提供外部实际反例，
目标词的独立端点质量核验仍未完成，延迟优胜结论继续禁止。
来源SHA、完整音素参考、TextGrid收集及实测见results/4090-mfa-ldc-sample-*.json；
复现experiments/mfa_ldc_sample_reference.sh，不访问唤醒分数或最终音频。


## 扩展独立会议的预定采样（2026-11-10）

在不改当前v28的前提下，固定下一批训练会议：
ES2006a/ES2007a/ES2008a/IS1001a/IS1002b/TS3008a/TS3009a/TS3010a，
每场仅Array1-01，分别按ES3/IS2/TS3的字典顺序首个合格会议选取。
两套划分都须为training，排除二者dev/eval参与者并集、
既有四场全部16位参与者及本批已选参与者；选择仅使用官方会议元数据。
预计32位新AMI参与者，实际小时数和完整下载校验完成后再报告，
不能按元数据duration提前当作已获得曝光。未做任何唤醒评分或最终音频访问。
下载/审计工具新增--sessions及--exclude-audit，默认原四场不变；
本批将单独准备，绝不向正在运行的v28输入目录追加。


## AMI 对话正负混音候选准备（2026-11-10）

在固定v28数据前缀上，对每个原始训练源的正例和混淆词各追加两个
AMI对话变体8/9，共29196条；仍使用原四场已审计训练会议，
不引入下一批8场或任何保留集声音。源审核SHA固定
ba4449e1a371e5594cb28968d409b85ff7f3be944a8b217f6c666f1a453d3023。
完整原5.17354816GB特征前缀和全部原索引行保留；所有新增行仅training。
Piper/Kokoro/Qwen各正负两组共6条实际波形重渲染，
元数据、渲染PCM哈希和整个特征数组逐项一致。

SNR仍5..15dB（按前景全片RMS，不是仅说话段），lead1.8..4.5s、tail1s，
不修改端点代理、不增加新RIR。最高削波2.21560255%，平均0.11071743%；
对话截取可能含停顿，不能宣称每条都同时说话。
增加的是相关增强，不是新独立会话小时或新物理说话人。
report/audit见results/4090-ami-mixtures-*.json及4090-ami-mixtures-v1.json。

候选features-gpu-ami-mixtures-v1与当前v28输入目录分开；
v29预定仅改变这项混音数据，相同初始化/冻结BN/BCE/prefix16/120000步
及原始量化导出流程；runner experiments/v29_ami_mixtures.sh。
待v28结束后再开始，不同时改变新增8场负语料或导出校准。
最终集保持冻结，不预判混音能改善泛化。


## 混音背景词区间覆盖诊断（2026-11-10）

独立于分类器，使用同一固定AMI注释包中仅四场training逐词时间，
合并所有参与者词区间，测量其对每条新增混音前景插入区间的覆盖。
29196条中8781条（30.0760%）在近似前景区间没有任何背景词覆盖；
18352条至少一半区间被词覆盖，平均覆盖0.625038，中位1.0。
不据此宣称零覆盖必然无声或有覆盖就有同时可闻语音。

前景区间使用整个TTS源长度（含停顿），插入位置只有10ms精度；
由位置取整导致的覆盖样本数误差不超过320样本，
此外官方词标注与实际声学的时差也未校准。
没有过滤变体、改标签、改当前训练或读唤醒分数。
这量化了既有“背景可含停顿”的限制：不能把新增变体数等同
同量的同时讲话困难样本，后续如需约束背景说话须另立受控干预。
源码training/audit_mixture_word_overlap.py，结果results/4090-ami-mixtures-word-overlap-v1.json。


## v28：AMI连续背景单变量结果（2026-11-10）

仅在v27数据中追加原四场AMI训练负语音，初始化与v27一致；120000步，
1246.8秒，选择70000步。INT8模型54440字节，SHA256
`c452815c3c2210bdfd4f8b516ca591f2682674b2118a2476d6e4aa7fe545b237`。
原验证选择阈值0.994，来源宏平均代理窗口命中69.6614583%，
Piper80.15625%/Kokoro59.1666667%，1681/2160、提前1次。
代理延迟中位130ms/P95 190ms，不是音素延迟。原ambient0/1.204811h。

扩展负验证Mandarin/DiPCo/Libri-dev-other分别0/0/1，
Libri为1/5.121184826h，单侧95%上界0.926321677/h。
扩展零事件校准无可行阈值，拒绝本候选，不访问最终集。
误触来自libri_dev_other_1585的83.78s，INT8平滑峰0.99609375；
600帧预热后复放确认峰值、float流式与窗口一致，float对应五帧均值
0.9994423985，表明该处高置信混淆也存在于浮点模型，不能只归咎量化。

三套same-PCM审计均通过；官方HiESP在相同Mandarin/DiPCo/Libri上为1/0/0，
仅表示开发数据观察事件总数同为1，不证明总体误触率相等或达到HiESP。
流式probe重置复现通过，平均漂移0.0051134449、最大0.3842321634；
抽查8条中6条float/INT8事件步完全一致，不能推广为全语料等价。
全部报告见results/4090-v28-*.json。v29仍只增加已声明AMI正例/混淆词混音，
不在运行中更改数据或导出配置。


## 追加8场独立AMI会议：准备审计完成（2026-11-10）

原定8场全部下载并完成连续前端特征：实际2.975060729小时、
32位新增AMI参与者，与先前16人交集为零；不是跨语料生物身份隔离证明。
8条流合计1071009帧，各流前两段连续前端等价检查通过。
源manifest SHA256 `4444b3ad507e9dbcccd826eb2988d1af33f47be0a75769ede36cb1732753970e`，
feature index SHA256 `02362bbc65cc2c104a07f2af49c93f66f84cb329f2017ef95e6710c19a80864c`。

仅training负语音追加至独立候选features-gpu-ami-extra-v1，
关键词binary/index与全部验证、开发回归行不变；未更改运行中的v29。
追加来源、原始manifest、cache、特征index和组合provenance均已保存
results/4090-ami-extra-*.json及jsonl。源审计旧limits文本误写Four，
实际records/files/sessions均为8；保持源审计字节和哈希不变，以补充审计明确纠正，
并修复生成器，使今后按实际会议数生成文字。

v29初始化JSON已实际比较，与v28完全一致；训练仍运行，尚无完整流式结果。
独立最终音频未评分，准备数据不作为识别质量或验收证据。


## v28：HiESP开发事件预算与配对复放（2026-11-10）

三套已审计相同PCM连续开发音频共12.789657865小时、50位说话人/会话。
官方HiESP为1次，v28原工作点也为1次；名义Poisson单侧上界同为
0.370914106/h，但反复使用的开发集不能作为独立验收或总体FAR相等证明。
在固定至多1次事件预算、原ambient0与混淆来源<=1%约束下，
v28仍选择0.994，代理来源宏命中69.6614583%。
同一诊断v19为79.2708333%、v26为65.3385417%、v27为72.34375%；
新增AMI背景尚未胜过v19，不按数据量或训练结束宣称进步。

对原72条配对开发WAV，实际重新运行官方WakeNet9 QEMU和v28 INT8，
72/72完成、CRC和同PCM及重置复现检查通过。初次长输出路径触发AF_UNIX长度
保护，保留失败目录；改用短/tmp/nyx-v28-paired-dev-v1后成功，未改模型/数据。
固定代理窗口Nyx8/12，HiESP1/12；若只看目标整段是否有事件，为8/12与5/12。
HiESP另4次在代理窗口前，之前已证实此代理边界不能当作真实末音素。
这些数不能证明Nyx真实召回或延迟优于HiESP。
共同负例Nyx1/48、HiESP0/48，误触为bf_alice速度1.1的Hey Next，2.99s；
交叉目标两引擎均0/12。不把该验证声线加入训练，不据此调整窗口。
完整结果results/4090-v28-paired-hiesp-budget-development-v1.json。

预备v30采样审计：等类别权重策略不变；新增会议使AMI的ambient抽样概率
从8.42050046%变为12.85455694%，不是仅增加数据而保持每个子来源概率不变。
这是既定类别内按时长采样的结果，应纳入解释；报告
results/4090-ami-extra-sampling-plan-v1.json。v30尚未启动，v29继续原进程。


## v28：Hey Next训练原片复放诊断（2026-11-10）

开发配对中的Hey Next误触不代表训练中缺少该词。使用v28实际输入
features-gpu-dialogue-ami-v1，按预定文本精确匹配且仅取training/label0/variant0，
共400个唯一训练源：Piper277、Kokoro100、Qwen23，不能解释成400位独立人类。
保持模型和开发预算阈值0.994、5帧平滑不变，完整序列流式复放，
每条reset后二次结果逐值一致。Piper2/277产生事件，Kokoro0/100、Qwen0/23；
两个训练误触峰均0.99609375。Piper中位峰0.03125，Kokoro最大0.071875、
Qwen最大0.1578125。逐源SHA/事件步/得分SHA见
results/4090-v28-training-hey-next-replay-v1.json。

最初从v29准备数据的未改动variant0前缀复放，后再用v28实际输入完整复跑；
400条记录和score哈希完全一致，正式报告固定实际v28输入的index/binary哈希，
首次诊断保留.local，未据此改标签。仅执行训练特征；没有导入开发声线。
这说明至少已有训练原片在流式触发器下也未完全拒绝，不是仅“从未见过该短语”。
但训练随机crop与完整流式复放任务不同、TTS意图不是独立发音核验，
不能仅凭这2条确定欠拟合、自动标签无误或应怎样加权；不得作为泛化验收成绩。
工具training/audit_training_phrase.py要求同模型验证校准、唯一训练原片，
固定原阈值，检查特征范围和逐片reset复现，无训练或阈值更新。


## v29：AMI关键词/混淆词对话混音完整结果（2026-11-10）

120000步、1319.1秒，选择70000步。与v28初始化JSON完全一致，
仅增加原四场AMI训练会议的关键词/混淆词混音变体8/9，不新增独立人群。
INT8 54440字节，SHA256
`0af1dc32031d661514bb9e4c5e263406c8cbbd1067d1e1ebb6839b635a1e4103`。

原密集验证工作点0.91015625：来源宏平均代理命中79.3229167%，
Piper87.3958333%/Kokoro71.25%，1849/2160，代理窗口前事件4次；
代理延迟中位120ms/P95 180ms，不是音素延迟。原ambient0/1.204811267h。
冻结原阈值评估，Mandarin/DiPCo/Libri-dev-other分别1/2/1次，
各上界0.686125930/8.344459947/0.926321677每小时。

扩展零事件工作点为0.99609375，三套语料0/0/0，
来源宏平均代理命中仅46.8229167%，Piper40.7291667%/Kokoro52.9166667%，
909/2160，提前0次，代理延迟150ms/P95 250ms。
这低于v27的70%，所以不能按原验证命中提高或零事件可行来宣布进步，
不启动独立最终测试。不同工作点不能混用召回和误触数据。

小样本probe8条reset复现通过，float/INT8事件步8/8一致；
平均漂移0.0012486488，最大0.2160888612；完整上下文平均0.0016896542、
最大相同。仅抽查，不能推广为全语料数值或事件等价。
全部完整报告results/4090-v29-*.json。

在v29全runner成功退出后启动v30：只追加前述8场独立训练会议负语音，
其余初始化、结构、混音、采样策略和导出不变，不修改v29输入。
既定类别内按时长采样使AMI概率变化已在前文披露；
v30用于受控检验独立会话覆盖是否改善，不预判成功。
最终集保持冻结；与HiESP匹配误触诊断仍待补充，不声称达到HiESP。


## v29：HiESP观测事件预算比较及float误触峰（2026-11-10）

全部三套same-PCM审计通过，原0.91015625下Nyx4次/12.789657865h，
HiESP1次；Nyx名义上界0.715696942/h，高于目标，且为重复开发数据。
固定HiESP至多1次预算后，v29选择0.92578125，负语料事件为Mandarin1、
DiPCo0、Libri0；代理来源宏命中78.0729167%，Piper85.3125%/Kokoro70.8333333%，
1808/2160、提前4次，原ambient仍0。此点优于v27/v28同预算72.34375%/69.6614583%，
但低于v19的79.2708333%，不能用零事件点的退化掩盖另一工作点的改善，
也不能把相同观测次数当作总体FAR相等或独立验收。

72条原配对开发音频重新实际复放完成。v29代理命中7/12，
HiESP1/12且另有4次在代理窗口前；关键词整片有事件分别7/12和5/12。
共同负例v29为1/48、HiESP0/48；交叉目标均0/12。
保留原代理边界，不把7vs1解释为真实召回或音素延迟优胜。

原阈值误触流的局部float与INT8复放峰：
Mandarin S0048为0.996486068/0.9953125，
DiPCo S05为0.948030353/0.9234375，
Libri1651为0.915861964/0.915625。
三处600帧预热、原INT8峰和float窗口/流式一致检查通过，
说明这些高分也存在于浮点，并不是纯量化误触；不是全语料float事件统计。
全部证据见results/4090-v29-{hiesp-*,paired-*,float-*}.json。

v30实际初始化JSON与v29完全一致，训练继续同一进程；
不因开发误触将任何验证音频/声线移入训练，不访问独立最终音频。


## v29中文误触原句与相位控制（2026-11-10）

对唯一Mandarin误触aishell_S0048的1207.16s，逐条核验该验证说话人的
原WAV/PCM哈希，依原200ms gap及对齐规则重建整流；
重建stream PCM SHA与既有frontend一致，原事件从已存score重新计数一致。
触发前1.7s全在BAC009S0048W0387.wav内部（原流1200.98..1208.2219375s），
没有跨原句拼接边界；1.7s仅保守观察区间，不是精确声学归因。
官方文本为“反观 秘鲁 得分 最 高 的 是 八 号 弗里 亚斯 十八 分”，
不能由文本定位实际误触音素或证明所有标注绝无问题。

将原句单独用相同模型、原阈值0.91015625、5分数平滑复放，
每次重置frontend/model，只在开头加0/160/320个零样本（0/10/20ms）：
0ms在相对原句6.17s误触、峰0.9921875；10ms无事件、峰0.66875；
20ms在6.18s误触、峰0.9953125。三种条件都通过二次reset逐值复现。
说明至少该句脱离拼接仍可触发，也存在明显的细小时间位移敏感性。
这些控制改变了frontend初始状态和模型相位，不当作原连续分数完全等价、
相位单因果证明或全语料普遍结论；更不能选择无事件相位重写原误触。

工具training/audit_read_speech_events.py，完整结果
results/4090-v29-mandarin-event-source-controls-v1.json；原事件仍计入，
不移动验证句到训练、不改标签或阈值，独立最终音频未访问。


## v29：量化五帧平均边界数值审计（2026-11-10）

旧smooth使用float64卷积与1/5权重，数学上等于阈值的量化均值
有时落在阈值下一浮点数。以模型实际output quantum1/256、window5，
先严格检查既有smoothed分数接近整数sum格点，再规范化为
round(score*5/quantum)*quantum/5，保持所有阈值、滞回和事件规则不变。
这是另列数值诊断，不修改原触发器、原报告或运行中的v30代码。

审计v29全部2160正例、3520混淆负例、50位说话人/会话连续负语料，
分别在原0.91015625、HiESP预算0.92578125、扩展零事件0.99609375。
正例在原阈值有2条、预算阈值有1条事件提前一步（30ms），
事件总数分别1853/1812/909均不变，固定代理窗口命中1849/1808/909也不变。
混淆负例与三套连续负语料的全部事件序列在三个工作点完全不变。
故该数值边界不能解释此前误触或召回退化，但足以影响个别30ms触发时间，
不能把延迟视作已经与MCU数值等价。原报告继续代表原策略，
后续若采用整数求和/统一平均规则须另立可复现版本并重标定，不偷偷回写历史。

源码training/audit_quantized_score_boundaries.py，报告
results/4090-v29-quantized-mean-boundaries-v1.json，包含输入SHA及全部差异源/步数。
没有新训练或最终音频评分。v30保持既有单变量实验，仍在原进程训练。


## v30：新增独立AMI背景，开发门槛通过（2026-11-10）

仅在v29基础追加已预定8场独立AMI训练负语音，实际2.975060729小时、
32位新AMI参与者；关键词、验证/回归输入和训练配置保持不变。
初始化JSON已实际比较与v29相同；类别内时长权重变化按前文披露。
120000步、1319.5秒，选择70000步；INT8 54440字节，SHA256
`04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c`。

原密集验证及扩展零事件校准均选择0.91796875，
来源宏平均代理命中82.5520833%，Piper86.3541667%/Kokoro78.75%，
1847/2160，代理窗口前触发0次。代理延迟中位120ms/P95 177ms，
不是末音素或硬件延迟。原ambient0/1.204811267h。
冻结同一阈值后Mandarin/DiPCo/Libri-dev-other分别0/0/0，
对应6.913985191/0.754487847/5.121184826小时，单侧95%名义上界
分别0.433285897/3.970550731/0.584968591每小时。

该点通过既定扩展开发零事件门槛；代理命中优于v27零事件70%与v29零事件46.8229167%。
反复使用的开发集与短远场曝光不能作独立最终验收，也不能直接声称达到HiESP。
还需同PCM/同预算与配对复放；独立最终集继续冻结，不因本轮通过就立即评分。
完整训练、初始化、密集验证grid、扩展校准、三套负语料和小样本probe报告
已存results/4090-v30-*.json。


## v30 HiESP开发审计与独立负例预注册（2026-11-10）

三套相同PCM审计通过。12.789657865h开发音频v30为0次、HiESP为1次，
名义上界分别0.234230838/h与0.370914106/h，均不是独立验收。
至多1次HiESP事件预算仍选择0.91796875，来源宏代理命中82.5520833%，
优于v19同预算79.2708333%；v30观察事件更少，不是总体FAR匹配证明。
72条配对复放完成，v30代理命中10/12、HiESP1/12且另4次在窗口前；
整片目标有事件为10/12和5/12。两引擎共同负例0/48、交叉目标0/12。
音素端点仍未核准，不以这些数字宣称真实召回/延迟优胜。

基于上述完整开发证据，冻结v30模型SHA、阈值0.91796875、原五帧平均/
滞回/重置规则，拟对先前封存的73位LibriSpeech test说话人10.745014097h
进行一次独立负例检验。预注册文件4090-v30-independent-negative-preregistration-v1.json
固定manifest SHA与统计规则，先提交并推送再评分。
所有73条完整运行，不因提前失败停止；不依据最终结果调整阈值或重新选择候选，
失败照实发布，后续候选需另备全新独立验收数据。最终负例永不进入训练。
本次不评分配对正例最终集；负例置信上界即使通过，也不代表完整目标达成，
官方HiESP同最终音频比较、独立正例/端点及部署分布证据仍需完成。


## v30 独立负例固定工作点结果：数值门槛通过（2026-11-10）

预注册和评估器已先提交并推送564adce，再开始读取最终分数。
用错模型的拒绝检查确认在任何最终推理/输出前失败；随后仅正确模型正式运行一次。
保持v30 SHA、阈值0.91796875、原五帧float64平均及滞回/67步refractory，
完整运行全部73位封存LibriSpeech test说话人；逐流feature SHA和3秒prefix
重置复现检查通过。测试模型/阈值未重选，原manifest字节和SHA保持不变。

**73/73完成，0次误触/10.745014097h；单侧95% Poisson上界0.278802079/h，
通过事先固定的合并负例数值门槛<=0.5/h。**
分语料保留：test-clean40人5.403466840h、0次、上界0.554409301/h；
test-other33人5.341547257h、0次、上界0.560836052/h。
两者单独曝光不足以各自证明<=0.5/h，不能将合并通过推广至每个子分布。
预注册判据为这套固定混合保留集的合并上界，不改变事先判据。

报告results/4090-v30-independent-negatives-v1.json，包括全部73条事件列表/
得分SHA；完整得分与progress保留.local/wakeword/v30-independent-negatives-v1。
这套负例现在已经正式评分，今后不得再称“从未评分”；不用于模型/阈值再选择，
也不成为训练数据。更换候选的独立验收需要新数据。

完整研究目标仍未达成：本次只有读语音负例；没有同最终音频HiESP结果，
正例最终集尚未评分，独立真实末音素端点/召回与延迟比较、远场/电视分布证据
仍不足。负例数值成功不等于整体达到HiESP。后续首先保持v30固定，
补齐官方HiESP对同负例PCM的比较及独立正例评测语义。


## 官方HiESP对独立负例的同PCM准备与预注册（2026-11-10）

从原固定官方test-clean/test-other归档校验SHA，并对已提取的每条FLAC
逐文件验证与归档成员字节相同。按原human_worker顺序添加200ms静音及
features()实际逐句160样本对齐补零，复用连续frontend；
全部73条流的每一帧重算特征与Nyx已评分的冻结缓存逐值一致。
这同时明确旧缓存的per-utterance隐式alignment，不能省略补零后声称同PCM。
重建WAV及源FLAC/PCM哈希见4090-hiesp-independent-negative-pcm-manifest-v1.json。
音频曝光仍仅原始10.745014097h，所有插入静音/补零排除。

官方wn9_hiesp保持既有开发请求阈值0.63/getter0.62496001，
QEMU/ELF/flash SHA固定，另立4090-hiesp-independent-negative-preregistration-v1.json，
先提交推送后再执行最终负例官方评分。此阶段没有重算或选择Nyx模型/阈值。

run_replay/run_corpus新增显式final-preregistration分支，
仅接受independent-negative-final/testing和固定PCM manifest/engine/threshold；
原development角色限制保留。实际开发样本复放事件时刻及CRC/stage104与旧报告一致，
未提供最终预注册及改阈值两种情况都在启动QEMU/创建输出前被拒绝。
完成后需对同PCM、全部流、事件与padding重新审计；
最终正例仍不评分，同负例结果不能证明总体FAR相等或真实延迟优胜。


## 官方HiESP同独立负例结果：73条完整审计通过（2026-11-10）

预注册dd642ac已先推送，随后官方未修改wn9_hiesp以既定0.63请求/
0.62496001 getter复放73条同PCM独立负例，全部完成，无运行失败。
逐流对照冻结Nyx结果、原feature SHA、重建WAV/PCM SHA，验证官方
CRC、stage104、调用次数、最终512样本填充、固定QEMU/ELF/flash SHA、
原阈值/getter及双方原音频曝光；Nyx原score SHA与全部事件重新计数通过。
未再次推理Nyx，模型/阈值未调整。

10.745014097h、73位保留说话人：
- Nyx v30：0次，观察0/h，单侧95%上界0.278802079/h。
- 官方HiESP：1次，观察0.0930664205/h，单侧95%上界0.441494490/h。
官方事件来自test-other说话人8188；全部事件保留，无文本过滤或事后调阈值。
两引擎在预定混合负例分布的名义数值上界均<=0.5/h。
不同子语料曝光与各自上界仍在完整比较报告中，不因合并通过而省略。

这证明固定工作点下同一独立负例PCM的观测结果，不证明总体FAR等价、
Nyx统计显著优于官方或同总体误触率的召回/时延排名。
开发采用的“至多1次事件预算”也不是此最终观测点的总体FAR匹配。
独立正例最终集仍未评分，可信端点/延迟比较、真实声学部署分布仍未完成，
完整研究目标保持未达成，不能写成已达到HiESP。

官方完整报告results/4090-hiesp-independent-negatives-v1.json，
逐流比较results/4090-v30-hiesp-independent-negative-comparison-v1.json，
审计工具training/compare_independent_negative_engines.py；
官方原始UART/每流report与输入PCM保留/tmp/nyx-hiesp-final-neg-v1。


## 独立端点盲审工具：准备完成，尚无人工标签（2026-11-10）

当前负例独立数值证据完整，但正例真实末音素延迟仍缺独立参考。
既有能量规则和MFA已存在实际反例，不能用自动输出自行验收。
为24条既有开发目标和2条官方控制准备独立盲审包：
所有WAV/PCM哈希核验、原始音频不变，匿名ID固定、随机顺序固定；
HTML内嵌音频，可直接离线打开，不包含模型分数、触发时刻或自动边界。
支持完整/慢速试听、波形放大、末端上下限（16kHz样本坐标）、
错读/无法判断与备注、浏览器进度保存、JSON部分或完整导出。
区间大于20ms时明确保留不确定性，不强迫缩窄；未完成部分不标作完整。
这只是获取独立标签的工具，不是已经验证的金标准。

生成器training/prepare_blind_endpoint_review.py、模板blind_endpoint_review.html，
实际包.local/wakeword/blind-endpoint-review-v3/index.html（全部音频内嵌），
审计results/4090-blind-endpoint-review-inputs-v1.json。
复现生成器仅需--alignment-inputs .local/wakeword/mfa-development-inputs-v1
及全新--output；旧评分窗口/标签不改。

内置浏览器因容器无display不可用，另建隔离review-browser-env安装Playwright1.65.0、
Chrome Headless Shell158.0.8127.18，补齐实际缺少的系统库和fontconfig后实测。
音频真实播放/解码、上下限颠倒校验、翻页/刷新保存、10ms区间精确导出8000..8160样本、
部分导出review_complete=false、无自动边界/分数、无pageerror均通过。
测试使用临时浏览器context和AUTOMATED_UI_TEST_NOT_HUMAN_LABELS身份，
导出fixture销毁、context销毁；没有生成或填充任何人工真值。
测试源码check_blind_endpoint_review_ui.py，证据
results/4090-blind-endpoint-review-ui-check-v1.json。

下一步仍需要本人或独立标注者实际试听与标注，并独立核验标注可靠性；
开发盲审通过也不能自动推定最终正例端点准确。已向用户请求这项缺失参考，
未将未答复当作人工标签。独立正例最终集仍未评分，v30模型/阈值不变。

## v30独立配对TTS诊断预注册（2026-11-10）

冻结468条原始TTS：26个保留国际声线×3语速×6意图，Nyx/HiESP各78目标、312共同负例。
原始PCM不变，统一添加1.8s前导和1s尾静音，双方消费相同PCM。
固定v30阈值0.91796875、5分数均值及原触发策略；官方请求0.63/getter0.62496001。
模型、输入manifest、前端/检测器/评分代码及QEMU/ELF/flash哈希均预注册，先推送后评分。
缺失预注册、错误阈值实际拒绝且不创建运行输出；开发样本复放stage104通过。

统计合成意图目标片段在原音频开始后是否有事件，单列前导事件，完整保留共同/交叉目标负例事件。
没有发音人工核验或可靠末音素边界，不统计音素延迟，不宣称真实召回或完整目标通过。
声线预设不是独立自然人，两种目标词也存在词汇难度差异。
任一运行失败不得生成质量汇总；不得从最终结果重新调阈值/筛选/重生成音频。
预注册results/4090-v30-independent-paired-preregistration-v1.json；
输入manifest results/4090-v30-independent-paired-pcm-manifest-v1.json。

## v30独立配对TTS诊断完成：未证明达到HiESP（2026-11-10）

fd7b1ed预注册推送后才首次评分；468/468官方QEMU完成，无运行失败。
逐条验证原PCM、CRC、stage104、512调用/尾填充、固定engine/代码/预注册哈希。
Nyx INT8全468条独立reset两遍分数完全一致，所有score/replay文件哈希核验。
固定阈值未改，未删除、改标或重新合成任何录音。

- Nyx意图目标片段原音频开始后触发42/78=53.8462%；HiESP45/78=57.6923%。
- 双方目标前导事件均0，交叉目标负例均0/78。
- 共同负例312条：Nyx4条/4事件，HiESP0条/0事件。
- Nyx共同负例事件按原意图文本：Hey Nick 1/78，Hey Mix 3/78；Hey Next和音量指令均0/78。

这是意图片段触发描述，不能直接称为正确发音召回率；26个国际TTS预设尚无独立发音审查。
不得从这些结果筛选合成声线、改最终标签或选择新工作点。
两目标词有词汇难度差异、预设不是独立自然人、没有可信音素端点；
本轮无延迟或显著性/总体FAR匹配声明，未达完整目标。
该最终集现已评分，后续候选的独立验收需要新预先冻结数据。
原始全事件报告results/4090-v30-independent-paired-v1.json；
按声线/原意图负例描述与文件核验results/4090-v30-independent-paired-audit-v1.json。
本地原始分数/UART/replay位于/tmp/nyx-v30-paired-final-v1。

下一阶段优先解决独立发音/末音素参考，并以训练来源扩展口音及近音词覆盖；
新训练不得使用本轮保留声线、最终录音或独立最终负例。
仅凭最终TTS触发比例无法区分错读与检测漏失，不能将其自动转成训练标签。

## v31训练侧窗口挖掘受控方案（2026-11-10）

v30已冻结，不修改其模型/阈值或已发布最终结论。
在原训练variant0、同冻结阈值下复放三个既有近音词：
Hey Mix Piper5/262、Hey Nick Piper1/312、Hey Next Piper5/277训练片段触发；
对应Kokoro/Qwen训练片段均无事件。每条独立reset两遍完全一致。
这只是重代入诊断，不是泛化证据；报告results/4090-v30-training-hey-*-v1.json。

新v31仅把negative_candidates1改4，保留v30全部训练特征/预训练初始化/
teacher60/context150/frozenBN/prefix16/stratified ambient/source exponent0/
seed726/120k步及每5k验证。每条抽中训练负例的原窗口保留，额外三个窗口
使用隔离RNG4726，以inference模式选最高分；训练正例/ambient基线抽样不变。
1000次endpoint检查通过：所有候选范围合法，原endpoint及主RNG状态与count1一致。
INT8代表校准仍关闭挖掘。shell语法检查通过。

已有v9的同原则实验失败（official64从零模型、v6训练数据、40k），
本次是对预训练/扩展数据组合的单变量再检验，不声称方法必然改善。
方案results/4090-v31-plan.json；脚本experiments/v31_negative_windows.sh。
先推送再运行，阈值只在既有开发验证集选择。
已评分最终正/负数据永不进入训练、调阈值或新候选独立验收；
后续验收必须新建预先冻结独立数据。当前没有新候选质量结果，目标未完成。

## 新独立远场负例来源选择：AMI官方测试划分（2026-11-10）

后续候选需要新的独立验收数据，不能重复使用已评分LibriSpeech/配对TTS最终集。
前瞻固定Kaldi AMI split_eval.orig全部16场会议、每场仅Array1-01原始远场通道；
共享元数据时长估计9.072156944h，实际曝光以下载后的PCM为准，不补齐/挑选短段。
16位参与者与官方Kaldi train/dev、原AMI seen_type training/development及本项目
实际12场训练会议参与者均需隔离。相同4人团队跨四场重复，不虚增自然人数。

先检查共享注释64份共88828词，无双方字面目标词模式；未读新音频或调用唤醒分类器。
转写筛查不是完整发音真值，未来需保留这一限制及会话/说话人聚类问题。
下载器training/prepare_ami_independent.py强制选择/代码/metadata/实际训练来源哈希，
只下载源WAV、核验格式/PCM/非静音并记录哈希，不导入任何唤醒引擎。
选择results/4090-ami-independent-source-selection-v1.json推送后才下载。
本记录不授权模型评分：未来先用开发数据固定候选，再单独推送评分预注册。

v31方案c734ef5推送后已在RTX4090启动。实际完成10500/120000步，
尚无运行错误；初始化全部JSON字段与v30完全相同，文件SHA均为
0997eae9819ad443e999bf8099ea4ce90192635386d3e44c19e1bd8f726b3c94。
审计results/4090-v31-initialization-audit.json。此时尚无训练完成或INT8质量结果。
新AMI来源选择e12ba31推送后已启动下载；未进行分类器评分。

## 新独立远场输入：完整源会议前端连续性实测（2026-11-10）

新增training/prepare_independent_dialogue_inputs.py；只接受固定SHA来源audit及全部
16场testing原始WAV，生成相同PCM的WAV和连续MicroFrontend特征，不加载模型。
原音频完整保留，仅最后0..159零样本用于10ms对齐，不插入中间间隔；
前端跨10s块保持同一实例，每场首次reset。原始时长与对齐时长分别保存。

在已完整下载的EN2002a实测：2142.709375秒原音频，末尾新增10个零样本，
214269帧。全部10s分块生成的特征与整段单次features(pcm)逐元素完全一致，
不是仅检查最初几帧；输出PCM往返读取也完全一致。无唤醒评分。
实证results/4090-ami-independent-input-continuity-v1.json。
该单会议检查只证明实现连续性，不替代全16场输入冻结、语音真值或识别验收。
下载和v31训练原进程均继续运行，尚无完整新模型评估结论。

## 新独立远场复放角色与拒绝保护（2026-11-10）

run_replay/run_corpus新增显式independent-dialogue-final角色，只接受testing及
单独final_role预注册；manifest/引擎/阈值和source audit/source manifest哈希固定。
检查连续录音、无插入中间间隔、原PCM保留、末尾0..159样本对齐及原始曝光。
旧development与原独立负例/配对角色保留；没有用新最终录音测试分类器。
原af_sky开发样本实际复放的事件、PCM/CRC、调用次数、stage104/getter
与既有报告一致。三个明确fixture分别缺预注册、改阈值、改来源audit，
均在创建输出/推理前被拒绝。证据results/4090-ami-independent-replay-guards-v1.json。

下载中核对到实际WAV时长不同于共享元数据：ES2004a元数据1186秒，
完整下载PCM为1049.3546875秒，公开转写末词1049.04秒。
不按元数据虚增曝光，也不裁剪/排除此流；最终输入以所有完整PCM样本数冻结。
截至本次核验8场已完成，全部末词时间未越过对应实际录音末端。

## 新AMI独立远场负例准备完成并冻结，尚未评分（2026-11-10）

16/16原WAV下载完成，源audit SHA
`ddd3c6d4a1d792965cc2fc8d154d3943f927db4bfabc69c38b7a62e35080447b`。
实际原始PCM曝光9.062306666666666h；16位自然参与者跨四组会议重复，
与官方train/dev及实际48位AMI训练参与者均不重叠。
逐场原WAV哈希/样本数核验，全部公开转写末词落在录音内；
这不等于完整语音真值或精确音素对齐。源转写/来源报告完整保留。

同PCM输入与连续缓存全部生成并复核：16/16原PCM前缀哈希与原始样本数一致，
仅新增总计1056个末尾对齐零样本，全部后缀为零，不计入曝光；
16/16输出WAV/PCM及feature SHA、uint16形状和10ms帧数均核验。
固定prepared manifest SHA
`588ff7fbf95a9aaca68d5e9895c975418e60a1f76fcf22862705b02d0adb5c84`。
无删除、改标、裁剪、会话拼接或分类器评分。

results/4090-ami-independent-source-v1.json记录来源/许可/完整原manifest；
4090-ami-independent-source-timing-v1.json记录元数据与实际PCM差异；
4090-ami-independent-input-manifest-v1.json冻结双方相同PCM及feature清单；
4090-ami-independent-input-audit-v1.json记录独立完整性复核。
本地原音频ami-independent-final-source-v1、输入ami-independent-final-inputs-v1。
输入准备并非识别验收；须在候选通过开发检验并固定后另推评分预注册。
v31原训练仍运行，已完成57000/120000步，未发布中途分数为最终识别质量。

## 独立远场Nyx固定工作点评分器准备（2026-11-10）

training/evaluate_independent_dialogue.py仅接受另行预注册的模型/阈值/代码/
完整16场PCM与feature哈希、源audit/manifest与固定原始曝光；不选择阈值。
检测器在全部输入核验后才构造，每场独立reset，前300帧重复检查后再全流评分。
检查全流分数长度与有限性，保存分数与原始事件；单流失败保留并继续剩余流，
有任何失败就不输出质量汇总。逐四个会议组报告时长/事件/名义Poisson95%上界，
明确重复参与者/会话相关性与总体分布失配不被名义区间覆盖。
此工具不测正例召回、音素延迟或硬件性能。

可复现拒绝检查check_independent_dialogue_guards.py使用明确无效的临时fixture，
改模型/非有限阈值/缺会议/缓存前端版本不符/路径越界五项均在输出创建和
模型构造前按预期分支拒绝；没有最终推理。
最终代码哈希与检查见results/4090-ami-independent-nyx-evaluator-guards-v3.json。
语法检查通过。早期开发检查留本地ami-evaluator-guard-development，不作为最终识别证据。
新的独立AMI最终集仍未评分，尚未选择用于该验收的候选工作点。
v31原训练已完成75500/120000步，继续既定计划；尚无完整新候选质量结论。

## 训练前缀截断的跨来源内容差异：诊断，不改标签（2026-11-10）

只抽既有training、variant0独立路径、standalone目标PCM，固定seed10726；
Piper64、Kokoro64、Qwen30条，检查原有150帧context及15/25/35帧提前量。
逐条源SHA与样本数核验；按MicroFrontend窗口支持计算源PCM截断范围，
统计原始源能量保留比例，不读取任何模型分数或最终录音。

提前150ms时源能量保留至少99.9%的样本：Kokoro43/64、Qwen12/30、Piper0/64；
中位保留比分别99.937%、99.825%、95.733%。提前250ms分别7/64、6/30、0/64；
提前350ms分别4/64、2/30、0/64。没有一个截断保留全部原始非零样本。
这只证明固定代理提前量的源声学内容不同，不证明完整关键词已被标成负例：
末尾/s/等辅音能量可能很低，失去很少能量仍可能失去关键音素。
同样，零“完整非零PCM保留”也不能证明前缀负标签全部正确。
源能量统计不包含渲染增益/削波/RIR；不将其当作实际训练特征能量统计。

工具training/audit_training_prefix_content.py；
完整样本与截断范围results/4090-v31-training-prefix-content-v1.json。
当前v31不改配置或标签；下一步标签改进仍需要独立发音/音素完整性参考，
不能从这些能量比例自动改标或声称找到了已证实的失败原因。

## v31训练/导出与首轮INT8开发验证完成（2026-11-10）

原定120k步全部完成，1761.0秒，既定验证规则选中5k检查点；
22,801参数、54,440字节，INT8 SHA
`9083f8139a481120146ffbe596b9317acd2e1739f34dd56091833401957b07fd`。
初始化/训练输入metadata/架构/步数/采样/前缀/损失等与v30逐字段相同，
仅既定negative_candidates1->4；trainer源SHA也一致。
训练results/4090-v31-training.json；配置核验4090-v31-training-configuration-audit.json。

密集INT8开发网格选择0.83203125，原环境0/1.204811267h，近音最差来源0.989583%。
代理窗口来源平均79.6614583%（Piper74.3229167%、Kokoro85%），
池化1631/2160，0提前事件。报告中的120ms中位/200ms P95仍是相对既有
代理端点，不能改称真实音素延迟或用来宣布优于HiESP。
低于v30密集/扩展零事件工作点82.5520833%，当前不支持替换v30。
完整原始report results/4090-v31-validation.json。

原pipeline继续既定Mandarin/DiPCo/Libri dev-other扩展验证及流式一致性检查，
尚无扩展最终结果；不因本次较低分数提前中止已承诺的开发验证。
新的9.0623h独立AMI最终集未评分，保留给有充分开发证据且预注册的候选。

## v31完整开发验证结束：拒绝替换v30（2026-11-10）

原pipeline正常exit0，所有预定训练/导出/开发校准/负语料/流式探针完成。
原INT8阈值0.83203125：Mandarin0/6.913985191h、DiPCo0/0.754487847h、
Libri dev-other2/5.121184826h。完整两次Libri事件保留，不改标签或隐藏。
既定扩展零事件规则选择0.95703125，三类额外负语料与原环境均零事件，
来源平均代理窗口命中率仅61.3802083%（Piper54.84375%、Kokoro67.91667%），
池化1216/2160、0提前事件；比v30同一规则82.5520833%低21.171875个百分点。
因此v31拒绝替换v30，不进行新的独立最终评分，也不继续为v31挑阈值。
这是该组合上的受控失败，不证明所有困难窗口挖掘方法都无效。

流式探针使用原始开发阈值0.83203125（不是新扩展阈值），重置重复性通过；
8条事件步骤7条完全相同，一条Piper目标float93/INT892，相差30ms。
全分数平均绝对误差0.0050656235、最大0.2214303613；完整上下文后平均
0.0012305081、最大0.0617836714。探针正例4/4双方均代理窗口命中。
这些是小开发子集的状态/量化证据，不是全域或MCU数值等价结论；
也不提供新扩展工作点音素延迟结论。

完整结果4090-v31-{mandarin,dipco,libri-dev-other}-validation.json、
4090-v31-expanded-calibration.json、4090-v31-streaming-probe.json。
新9.0623h AMI最终输入仍未评分；v30模型/阈值与已有最终结论全部保持冻结。
独立发音/音素端点参考仍未取得，盲审包尚无人工标签。
后续优先验证训练标签/前缀完整性及新的独立正例参考，不凭能量保留比例自动改标，
也不将v31训练结束或可复现导出当作整体目标达成。

## 训练前缀完整性盲审包：精确训练渲染，尚无人审标签（2026-11-10）

新增prepare_training_prefix_review.py，从既定训练内容探针中按seed47726，
各引擎随机4条原始来源，共12条；不按能量或分类器输出选择。
每条保存完整渲染及15/25/35帧提前量的三种截断，共48条匿名随机排序音频。
逐条重建variant0整段渲染，全部MicroFrontend数组与实际训练缓存逐元素一致。
截断取实际150个重叠特征帧的152×10ms声学支持区间，原始渲染PCM精确切片；
只供听辨，不将切片重置前端后的新特征当原训练特征。

页面不包含来源/原标记/提前量/能量/自动边界/模型分数，复用已测离线界面；
要求判断完整发音，缺音素时如实记发音不符与备注，完整时给末音素不确定区间。
原始12条及其36条相关变体不当作48独立人，也不作召回评测或自动改标。
生成结果.local/wakeword/training-prefix-blind-review-v1/index.html，音频全部内嵌；
pack_id b74e6bc59ced7ec8676960a2f94db8c38d0d71a6d8b0ae5355b32a67b4fdf03d。
完整映射/源码/PCM/缓存哈希results/4090-training-prefix-review-inputs-v1.json。

UI检查器增加可选expected-clips，默认26保留旧包检查；
真实离线file页面48条新包和26条原端点包均通过播放/解码、非法区间拒绝、
标注样本精确导出、翻页/刷新保存、部分结果不误报完整、无页面错误。
检查fixtures销毁，没有人工标签。证据4090-training-prefix-review-ui-check-v1.json、
4090-blind-endpoint-review-ui-check-v2.json。
新训练包与原开发端点包均仍等待实际独立试听；机器检查不代替人审。
没有启动新训练或修改v30/v31/最终集；完整目标保持未达成。

## 人审导出的完整性与分歧收集器（2026-11-10）

未发现实际试听JSON，48条训练包与26条开发端点包仍无人工标签。
新增training/collect_review_annotations.py：从冻结HTML/audit复核pack_id与全部匿名ID，
接受一个或多个原UI导出；核验协议、父manifest、盲审声明、音频SHA和样本数。
缺/多/重复音频、非法/反向/越界样本坐标、布尔值伪装整数、错误完整声明及
已知AUTOMATED_UI_TEST导出均拒绝；合法部分结果保留，未完成记录不作有效标注。
重复标注者名称不当作多人；名称不同也不证明真实身份独立，报告明确未核实。

逐条保留有效意见与备注，单人意见不当共识，发音分歧/无法判断不丢弃。
多人均认为短语完整时仅报告保守端点区间并集及是否重叠，
不通过求平均/取交集制造精确边界。文件有效或意见一致不自动成为金标准，
不改训练标签、模型、历史窗口或最终集，也不宣称真实人声来源已核实。

check_review_collection.py在两套真实冻结输入上使用纯内存synthetic fixtures：
各16类错误导出拒绝，以及合法部分、区间并集、不相交区间、发音分歧、单人
非共识、重复身份保护均通过。没有写出fixture标注文件或生成任何真人标签。
证据results/4090-blind-review-collection-check-v1.json。

收到实际导出后的调用示例（单人/部分结果也可先完整校验）：

```sh
python3 research/wakeword_p4/training/collect_review_annotations.py \
  --html .local/wakeword/training-prefix-blind-review-v1/index.html \
  --input-audit research/wakeword_p4/results/4090-training-prefix-review-inputs-v1.json \
  --annotations /path/to/actual-review-export.json \
  --output .local/wakeword/actual-prefix-review-validation.json
```

多位标注者可重复--annotations；开发端点包使用其对应HTML/audit。
当前缺少的仍是实际独立试听参考，不以工具完成替代这项数据。

## 冻结v30在新AMI远场域的独立双引擎检验预注册（2026-11-10）

当前没有独立试听标签，正例/音素延迟验收不能完成；继续推进不依赖端点的
独立真实对话误触域检验。v31已拒绝，选择此前由开发数据选定、完全不改的v30，
仅增加新域证据，不把本次最终结果用于新训练/模型/阈值选择。

模型04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c、
阈值0.91796875及原5分数均值/触发器不变；官方原请求0.63/getter0.62496001不变。
双方使用已冻结manifest588ff7fbf95a9aaca68d5e9895c975418e60a1f76fcf22862705b02d0adb5c84，
完整16场/16位反复出现的参与者、9.062306666666666h原始PCM，只有末尾对齐零样本。
先推送results/4090-v30-ami-independent-preregistration-v1.json及比较代码，再首次评分。
模型、全部engine、输入/代码SHA已核对，无推理。

主要负例数值标准为每引擎独立在本AMI曝光上Poisson单侧95%上界<=0.5/h，
不将旧LibriSpeech小时并入以挽救失败；四个重复参与者会议组也全部列出。
所有流与事件保留，不因提前出现误触停止或过滤；不完整就不报子集质量。
训练标签/音素端点未核验的问题、会议相关性、转写筛查局限和总体分布差异保留，
数值通过也不证明HiESP整体等价、真实正例召回/延迟或硬件性能。
新AMI从此次运行开始成为已评分最终集，未来候选需要新的独立验收来源。

比较工具training/compare_independent_dialogue_engines.py只读取原始结果，
核对PCM/CRC/stage104/全512调用与尾填充、缓存/score SHA、固定阈值/engine/
源代码及每流曝光，按原检测策略重计全部Nyx事件，不再次加载模型推理。
比较代码SHA也预注册；最终报告前必须完成该逐流审计。

## v30新独立AMI检验：Nyx完整完成，数值标准未通过（2026-11-10）

d2d5d32预注册推送后首次评分；Nyx16/16完整完成，无运行失败，原始scores保存。
原始曝光9.062306666666666h，1次事件，观察0.110347182/h；
单侧95%上界0.523472080/h，超过预注册0.5/h，因此本域数值标准失败。
不四舍五入通过、不合并旧LibriSpeech时长、不删流/事件/改标签或调阈值。
这与之前独立LibriSpeech0事件结果并列保留，不能把read-speech通过推广成远场通过。
结果results/4090-v30-ami-independent-negatives-v1.json，原始分数
.local/wakeword/v30-ami-independent-final-v1；每场开头reset重复性已核验。
官方HiESP同PCM原进程仍运行；此时尚无双方完整审计/质量排名结论。
全部16场的按会议组结果与原事件时间保留；相关性等名义Poisson限制不变。

## 新AMI独立双引擎检验完整审计：双方均未通过数值上界（2026-11-10）

官方16/16复放完成，无运行失败；预注册比较工具逐流审计全部通过。
双方原始曝光9.062306666666666h；Nyx1事件，HiESP1事件，观察均0.110347182/h，
名义Poisson单侧95%上界均0.523472080/h，均高于预注册0.5/h。
这表示本次有限曝光未证明所需上界，不证明总体实际FAR超过0.5/h。
Nyx事件IS1009c/1743.56s；官方事件IS1009d/1086.624s，会议不同；
相同有限样本计数不等于总体FAR相等，也不证明召回/延迟达到HiESP。
四个重复参与者会议组的曝光与各自数值上界均列出，不能视16场为16组独立人。

已核验全16输入WAV/原PCM/对齐后PCM SHA、原音频保留和零后缀、
QEMU CRC/stage104/全调用次数/最终512填充、固定代码/阈值/getter/engine SHA、
Nyx缓存/score SHA、全分数长度、原触发规则重计事件、逐流双方相同曝光。
没有再次加载模型或重跑Nyx，未筛选/重标/调阈值/追加曝光以改变结论。
结果4090-hiesp-ami-independent-negatives-v1.json、
4090-v30-hiesp-ami-independent-comparison-v1.json。
官方原始inputPCM/UART/per-stream报告另完整复制到
.local/wakeword/hiesp-ami-independent-final-v1，所有复制文件SHA逐个一致；
/tmp原始运行也保留，Nyx原始分数位于v30-ami-independent-final-v1。

新增4090-independent-data-usage-ledger-v1.json统一列出三组已评分最终数据：
LibriSpeech独立负例、Kokoro配对意图片段、AMI独立远场，均不可再称未触碰，
不可作为后续候选训练/阈值选择或新的独立验收。
原始输入manifest中的classifiers_scored:false等只描述准备时状态，
输入哈希冻结后不能用这些历史字段推定当前未评分；以评分报告/使用台账为准。
此前read-speech名义上界通过与本次远场上界失败完整并列，不合并以挽救失败。
实际独立发音/端点参考仍缺失，v31拒绝结论不变，整体目标未达成。

## v32前缀负标签消融预注册（2026-11-11）

用户要求重新设置目标继续推进。保留完整验收要求，先执行有明确假设的开发实验：
相对v30仅prefix_negatives16→0，原混淆负例槽恢复，不更改标签或最终集。
边界参考尚缺不阻止该消融，但仍阻止最终召回/末音素延迟结论。
配置见results/4090-v32-plan.json，执行experiments/v32_no_prefix_negatives.sh。
同v30开发评测与扩展负例标定；特别检查早触发，失败也保留推送。

v32原进程session80679确认存活；启动初始化JSON与v30全部字段相同，
审计results/4090-v32-initialization-audit.json。未因观察等待重启任务。
同时复核历史四次预训练embedding头实验，不能当作更强教师或未尝试路线；
摘要和报告哈希见4090-pretrained-reference-history-audit-v1.json。

### v32量化校准耦合检查

源码复核发现representative()调用batch(mine=False, augment=False)，
但batch中的prefix替换不受mine标志控制。因此prefix16→0也改变INT8代表输入，
虽然校准流程和主RNG不变。此前脚本“Original export calibration retained”
只能理解为流程保留，不能理解为代表tensor完全一致。
v32是端到端prefix策略消融，不能把INT8差异全部归因于训练梯度；
不修改正在运行的代码、不覆盖原导出，保留浮点与INT8探针。
若需严格训练因果隔离，须另外预声明并对双方冻结权重使用完全相同的代表输入序列重新导出。
审计results/4090-v32-export-calibration-coupling-audit.json。

### 同代表输入的补充量化对照预注册

为减少原prefix策略与INT8代表输入耦合，预注册补充导出；不改v32原任务。
common_calibration_export.py固定seed57726，对每个引擎/标签组选择20个variant0训练行，
均匀裁150帧，共120窗，保存实际float32输入；无模型分数选择/验证/最终数据。
inputs SHA52a484b1d6d3ee0375cd2b29bef144ab92ddfe0984fb97861d96e3457c4ff9ca。
双方冻结SavedModel分别使用同一输入顺序及原转换配置；保留所有原导出。
补充校准不含独立ambient行，不声称其为最优校准或MCU等价；
分别按原开发门槛评估，不根据结果修改代表输入，不取代原预注册结果。
预注册results/4090-v30-v32-common-calibration-preregistration-v1.json。

共享代表输入补充导出v30完成，TF2.21 CPU转换退出0；
模型54440B，SHA b744c466e6bde704eeb1e5cffe232a13505400f8c3f52953a9563b31dea90d85。
原v30模型未覆盖，完整SavedModel文件哈希/训练报告/输入哈希保存在
results/4090-v30-common-calibration-export-v1.json。
补充开发评估session2842已启动，v32原训练session80679继续运行。

共享校准v30的原英语开发评估完成（session2842退出0）：
阈值0.9375，代理macro命中率82.0833333%，1822/2160，早触发1，
Piper85%、Kokoro79.1666667%，ambient0/1.204811267h。
这是补充导出自己的开发工作点，不能替换原v30冻结阈值或宣称音素延迟；
早触发完整保留，不用改端点窗口消除。
结果results/4090-v30-common-calibration-validation-v1.json。
接着用experiments/common_calibration_negatives.sh v30跑原三组连续开发负例，
session54432；未使用最终集。v32原训练session80679继续运行。

共享校准v30补充连续负例全部完成（session54432退出0）：
Mandarin0/6.913985191h、DiPCo0/0.754487847h、Libri-dev-other0/5.121184826h。
扩展开发门槛仍选0.9375，代理macro82.0833333%、早触发1。
分别保存三组报告及4090-v30-common-calibration-expanded-validation-v1.json。
这些是重复开发评测，短时长和选择偏差不允许独立FAR验收解释；
同代表输入对照的v32一侧尚未完成，不能下双方差异结论。

## v32训练完成：原密集开发点未显示收益（2026-11-11）

120000步完整训练+导出1200.2s，选中15000步；RTX4090，22801参数，54440B。
原模型SHA be9130507b6f350c6bfe9209d3329d928d956b5565cbbad2fcefa8fa418870d3。
配置审计所列所有不变字段与v30相同，prefix16→0，模型哈希已核验。
结果4090-v32-training.json、4090-v32-training-configuration-audit.json。
原密集INT8开发点0.9609375：代理macro75.7552083%，1656/2160，早触发26；
Piper76.9270833%、Kokoro74.5833333%。并非音素真召回/延迟；早触发不删除。
尚待扩展开发负例与共享校准双方对照，当前不替换v30、不评分最终集。
原pipeline session80679继续负例/探针；共享校准v32补充pipeline session17186已启动。

v32原pipeline完整退出0，原冻结密集点连续开发负例：
Mandarin1/6.913985191h、DiPCo1/0.754487847h、Libri-dev-other2/5.121184826h。
原预声明扩展网格没有可用零事件工作点：最高0.99609375仍DiPCo1、Libri1，
代理macro49.21875%、早触发2；不扩大网格追求挽救，也不替换v30。
原流式探针reset重复通过，4/4正例代理命中；全分数MAE0.00286480518、
max0.139050603，完整上下文MAE0.00182335867。不是MCU完全等价。
五份负例/校准/探针报告保留；共享校准补充侧仍在完成原三组开发负例。

## v32补充共同量化输入对照完成：拒绝替换v30

session17186完整退出0，双方代表输入/manifest/导出代码SHA相同。
v32补充模型SHA34df915c60ddcb103f4c46f6147ddc755cdd0e88671fa233d8a7fa92341b5196，
密集点0.96875：代理macro72.8385417%，1586/2160，早触发25；
Mandarin0、DiPCo1、Libri-dev-other2，扩展网格仍无零事件可用工作点。
共同输入v30：代理macro82.0833333%、早触发1、三组负例0/0/0，有扩展开发点。
因此本配方取消prefix负例没有改善，不替换原冻结v30；原导出和补充导出均保留。
这不证明所有prefix标签都正确，也不否定其他有依据的标签处理。
共同输入减少量化输入混杂，不消除检查点选择/学习状态差异。
完整结果与来源哈希见4090-v30-v32-prefix-ablation-comparison-v1.json，
六份v32-common-calibration结果报告同时保存；均非独立验收/音素真召回延迟。
整体目标未完成，未评分任何既有独立最终集；原v30模型/阈值不变。

## v33不确定前缀损失屏蔽预注册（2026-11-11）

用户重新设置完整目标继续。v33保留v30全部16个prefix输入与抽样，仅将这些槽训练权重20→0；
不恢复其他混淆负例、不改其余权重或batch分母，不声称旧标签错误。
当前没有可验证的完整/不完整子集，所以先对全部16个未核验前缀保守屏蔽，
不是已经实现按真实音素挑选的精细屏蔽。保留frozenBN，防止无损失样本改统计。
120k/5k、teacher60/context150、原初始化/特征/开发协议均不变；最终集不评分。
实际trainer AST权重块已执行检查，baseline及屏蔽槽/其他权重/零梯度语义通过。
计划4090-v33-plan.json；检查4090-v33-loss-mask-check.json。

v33原训练session7030/PID18604存活，初始化全部字段与v30一致。
额外执行实际batch函数AST，使用确定性合成feature提供器验证控制流：
训练和representative两个调用模式中，mask开关不改变输入、标签、调用序列或main/prefix RNG状态，
仅16个槽的权重20→0。此处合成输入仅用于代码语义，不是识别质量或实际完整特征一致性证据。
审计4090-v33-batch-invariance-audit.json；不据此宣称转换器状态或量化数值等价。

## 后续混淆词辅助分类的训练集意图清点

仅清点现有训练feature index文本，不读取最终集/模型分数；
results/4090-training-phrase-inventory-v1.json保存实际输入SHA和分引擎逐文本计数。
现有训练已含Nick/Next/Mix/Six/Max/Nice/Nexus，不能把增加这些字符串当新数据。
三引擎词分布不同，variant0正例Piper6000/Kokoro300/Qwen30，增强行不是新人。
关键约束：当前负例裁剪可取静音或后缀，不能将整句意图类别强行赋给任意随机窗口。
若后续做多分类辅助头，需预声明有效短语crop或clip级聚合监督，保留二分类输出。
本次清点不证明实际发音/音素边界或说话人隔离；v33未完成前不启动另一训练。

## v33训练/导出完成（2026-11-11）

完整120000步，选中35000步，训练及导出1165.8s；54440B、22801参数。
模型SHA a38e15ad99082e17619a0ba7fdfa8ef16272c1701a18ee7227aa65a336f24e19。
配置审计所列字段与v30全部一致（包括prefix16），仅开启mask_prefix_loss。
报告4090-v33-training.json、4090-v33-training-configuration-audit.json；模型哈希核验通过。
原INT8代表采样策略保持，但原导出未保存实际代表tensor，不宣称bitwise一致。
原session7030继续开发评测；没有最终数据评分或达标结论。

## v33完整开发评测完成：拒绝替换v30

原session7030完整退出0，原密集点0.96875：代理macro63.515625%，1466/2160，早触发19。
Mandarin1/6.913985191h、DiPCo0/0.754487847h、Libri-dev-other2/5.121184826h。
扩展零事件开发点0.99：代理macro53.90625%，1139/2160，早触发3；
三组连续负例分别0，不等于独立FAR置信验收。
v30原扩展代理macro82.5520833%、早触发0；v32无可用扩展点。
v33不替换v30；全局屏蔽16个未核验前缀损失不改善本配方，不再重复此类实验。
不据此证明所有旧前缀标签正确；未来精细策略须有独立内容参考。
比较4090-v30-v32-v33-prefix-policy-comparison-v1.json，包含输入报告哈希。

原密集阈值下streaming probe重置重复通过，8/8事件步一致、正例代理命中float/INT8均3/4。
全score MAE0.00222747128，max0.489960879；完整上下文MAE0.00293968606，max同值。
最大漂移在Kokoro开发正例af_sky/001.wav的4.01s，事件步双方仍106；保留全部分数，
不能把小均值或此探针事件一致解释成MCU数值等价。未评分任何最终集。
训练/开发/负例/校准/探针结果完整保存，整体目标仍未达成。

## 混淆辅助分类的完整PCM窗口支持审计

training/audit_phrase_crop_support.py核验全部14598条training/variant0源文件SHA、
16k单声道/样本数与精确2秒lead；不读取开发/最终音频、不运行分类器。
150个重叠特征覆盖152hop=24320样本，按半开区间计算完整原PCM可覆盖的裁剪位置。
12729条可完整放入现有上下文；1869条明确不适用，不裁断后强行赋整句类别。
主要短词：Piper Nick312/312、Mix262/262、Next277/277；
Kokoro Nick99/100、Mix96/100、Next93/100；Qwen Nick21/24、Mix20/20、Next23/23。
检查24320和24321等边界，保留长片段和各引擎覆盖差异。
完整逐源位置manifest在.local/wakeword/training-phrase-crop-support-v1.json，
其SHA及逐文本计数在results/4090-training-phrase-crop-support-v1.json。
这是声学支持而非正确发音/音素边界证明；不推广为增强/RIR尾音覆盖。
后续辅助头可以仅对这些完整源crop用意图类别训练，保留原二分类随机窗口任务，
但必须预声明额外监督权重/类别映射/校准与检查点规则；本轮尚未启动新训练。

## v34短语辅助分类预注册与端到端检查

保留v30二分类batch128/prefix16、原权重/抽样/导出策略；额外24个完整源训练窗口，
四类target/Nick/Next/Mix各6条、类内等引擎、独立RNG67726，无增强，CE权重0.1。
使用共享binary Dense前的特征建立训练专用4类头，原binary BCE+0.1CE共同更新一次Adam。
只按原binary开发规则选检查点与阈值；保存best_auxiliary权重，但导出原binary模型。
这不是经核验的音素类别，也不是不增加训练计算的纯架构对照。
120k/5k、teacher60/context150/frozenBN、原初始化与开发语料不变；最终集不评分。

实际训练特征梯度检查通过：辅助loss更新13个共享变量、BN不变、binary原预测不变，
24窗类别各6；两步完整训练/检查点/INT8转换检查通过，导出仍单uint8概率、22801参数。
结果4090-phrase-auxiliary-smoke-v1.json、4090-phrase-auxiliary-export-check-v2.json。
原152hop审计仅为feature覆盖；teacher60忽略最后输入帧，所以辅助crop end+1，
仅保留24160样本内且实际149帧卷积支持覆盖全部PCM的源，防止错误完整性声明。
该修正规则已重跑梯度及两步导出检查，旧v1流程日志保留，不作为正式训练结果。
计划4090-v34-plan.json；正式脚本experiments/v34_phrase_auxiliary.sh。

v34正式任务session2782/PID26851存活；初始化所有字段与v30一致，
实际manifest/训练器/辅助模块SHA与预注册一致，weight0.1。
修正phase后目标组Piper2585/Kokoro285/Qwen27；Nick312/99/21，
Next277/93/22，Mix262/95/20；均按完整源有效支持筛选，未引入最终数据。
启动审计results/4090-v34-startup-audit.json；不作为发音正确或质量证据。

## v34训练/导出完成

完整120000步，选中80000步，训练及导出2056.4s；54440B、22801参数。
原INT8模型SHA ddb05cd3b83d1c6884aacbb290044ad4808b9a78c17dd4dddc6f236a69157e07。
基线固定字段全部与v30相同；辅助权重/manifest/代码SHA符合预注册，保存best辅助权重SHA。
报告4090-v34-training.json与4090-v34-training-configuration-audit.json。
原session2782继续完整开发评估，未评分任何最终数据，尚无候选优劣结论。

## v34完整评测：拒绝当前辅助短语配方替换v30

原session2782完整退出0，原密集开发阈值0.96484375：代理macro75.1822917%，
1865/2160，早触发14；Piper89.53125%、Kokoro60.8333333%，来源差异保留。
Mandarin3/6.913985191h、DiPCo1/0.754487847h、Libri-dev-other1/5.121184826h。
扩展预设网格无可用零事件工作点；最高0.99609375仍DiPCo1，
代理macro63.2552083%、早触发1，不扩大阈值网格挽救。
v30维持原冻结模型/阈值；比较4090-v30-v34-auxiliary-comparison-v1.json。
本结果不支持当前四类干净短语额外CE配方，不证明所有辅助监督方法无效；
未使用既有最终集，也没有独立发音/末音素验收结论。

正式INT8结构核验为[1,3,40]输入与单[1,1]uint8输出，无辅助头tensor名称；
报告4090-v34-export-structure-audit.json。部署参数22801不变。
原密集点float/INT8探针reset重复通过，7/8事件步完全一致，正例代理命中双方4/4；
全score MAE0.00185657583、max0.189404964，完整上下文MAE0.00194479253。
完整分数与偏差保留，不解释为MCU数值等价。
六份验证/负例/校准/探针报告完整推送；整体目标仍未达成。

## 跨家族训练语音审核预声明

复核历史tiny.en/small.en同200Piper WAV审核：small.en别名一致80/100正、79/100负，
不能由tiny.en大量Next转写推断普遍错读。新增不同家族wav2vec2-base-960h CTC诊断，
固定revision22aad52d435eb6dbaf354bdad9b0da84ce7d6156，CUDA FP32贪心、不用LM/VAD提示。
只用相同200条历史训练pilot，100正100负；保留原WAV与标签、全部转写和分歧，不筛除。
ASR无论一致或不一致均不是发音真值/音素边界，不作唤醒质量或HiESP验收。
代码training/audit_training_ctc.py，预声明results/4090-ctc-training-audit-plan-v1.json。
依赖在.local/wakeword/ctc-audit-env隔离安装，模型固定版本下载/哈希校验进行中。

## CTC跨家族审核完成：不同识别器分歧不足以证明错读

固定200训练WAV完整完成，CUDA4090 FP32、贪心CTC，无LM/VAD提示。
模型revision/LFS SHA全部核验，依赖版本与来源报告保留。
原声明严格别名一致：CTC正4/100、负26/100；同WAV Whisper small.en正80、负79。
CTC经常输出HEYNIX等连写。事后仅删除空格的敏感性诊断为正20、负28；
原始严格结果不改，不增加HAY/PAY等事后词汇别名或编辑距离容差。
两种归一化都不表示发音正确率；该CTC在这些短合成词上的词法/分词偏差明显，
不能把其低一致率当作训练语料大规模错读证据，也不能自动筛除/重标。
所有200条转写及跨家族对照在4090-piper-training-ctc-details-v1.jsonl；
主报告4090-piper-training-ctc-audit-v1.json，空格诊断4090-ctc-training-spacing-sensitivity-v1.json。
模型来源4090-ctc-model-source-v1.json，运行版本4090-ctc-audit-environment-v1.json。
审核未读取任何独立最终音频或唤醒分数；没有发音/末音素金标准与达标结论。

## CTC自然语音机械正控制预声明

固定现有train-clean-100按目录字典序前24位说话人，各取首章首条原始FLAC，完整PCM。
清单SHA256 0f9593ac0e554c575686bee172bb2a1ec1348cd3e62b38d47d06f55785087815。
复用固定wav2vec2模型、CUDA FP32贪心CTC；不调整解码器，报告全24条WER/CER。
归一化仅小写ASCII字母/空格；CER再移除空格。选样先于推理。
此语料可能属于该ASR的960小时训练集，只检查运行/解码机制，绝不解释为独立泛化表现。
不用最终集、不评分唤醒模型、不改标签；代码ctc_natural_control.py。

正控制24/24完成，24位说话人，737参考词中4词编辑错误，WER0.54274084%；
3327参考字符中6字符编辑错误，CER0.18034265%。全转写/来源/哈希均保留。
这支持当前GPU FP32贪心解码运行正常，不能把合成Nyx短词低一致率简单归因于运行损坏；
也不能排除模型对训练内自然语音有优势。不是独立ASR性能或短词发音真值。
加载警告masked_spec_embed为训练掩码嵌入，当前eval不进行该掩码增强；未重新训练或改解码。
下一研究决策：停止使用无参考ASR一致率自动清洗；优先取得盲听发音/末音素标注，
再在训练/开发范围核验音素监督或数据修复假设。没有参考时，不启动无明确依据的新参数轮训。
v30仍冻结，最终达标目标继续active；现有最终集不可重用作为新独立验收。

## 冻结v30交接Mac/P4（2026-11-11）

按用户要求原样打包hey_nyx.tflite、原始training.json、原始expanded-calibration报告及manifest。
模型54440 bytes，SHA256 04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c。
两份报告逐字节核对原始发布commit f212a40fdae2f077572383187937a4c4d3a525fa；
模型为未跟踪原始产物，以同commit训练报告记录的模型哈希认证，不声称模型blob已在Git内。
threshold=0.91796875，smoothing_steps=5；未重新导出、校准或训练。
已上传现有research-hey-nyx-20261110 Release，并无认证重新下载核验归档及全部4成员。
归档SHA256 60000e37f1b01c408f48f6310bfa7c6d0e837fe6d808127aab9de9a88e1e3cb5。
下载地址与验证记录：results/gpu-v30-p4-handoff-v1-download.json；
清单：results/gpu-v30-p4-handoff-v1-manifest.json。
只交接模型/研究元数据；PCM自测、P4输出/资源与同条件Hi ESP对照由Mac侧开展。

## 显式音素、未裁切合成pilot预声明

Kokoro 0.4.9支持is_phonemes=True、trim=False。固定Hey Nyx输入hˈeɪ nˈɪks，
并生成Nick/Next/Mix混淆词；4个已有training声音×3速度×4短语=48条。
只用训练声音，不声称新独立留出数据，不读取最终集，不评分唤醒模型。
原生24k float波形npy与16k PCM16 WAV同时保留，核对token/duration与样本数。
自动门槛在推理前固定：有限值、0.2–5秒、RMS≥0.001、零abs(sample)≥1；
完整原生输出不裁切。生成末端不冒充末音素真值。见synthetic_phone_pilot.py。

显式音素pilot完成48/48，全48条通过预声明机械质量门槛；原生波形与duration映射核验通过。
另逐条读回WAV确认16k单声道PCM16、样本数/哈希一致且无int16饱和值；
原生npy逐条读回确认单声道有限值和样本数。
报告4090-synthetic-phone-pilot-v1.json与全清单4090-synthetic-phone-pilot-manifest-v1.jsonl。
本轮未训练、未评分，因此没有模型提升结论；下一步需在训练用途内比较完整短语合成输入，
并制定声音来源分区与新开发协议，不能把本pilot复用的训练声音声明为独立测试。

## 显式音素pilot冻结v30诊断预声明

48条训练来源pilot全部复放，不选阈值：v30原模型，threshold .91796875，原float64五分数均值。
每条只添加1.8秒前静音/1秒后静音，保留全波形，重置重复核验输出。
报告所有事件/分数及相对整个生成末端的偏移，不把any-event计数当完整短语召回。
训练来源诊断不用于阈值/最终验收；不读取既有最终集；见score_phone_pilot.py。

48条冻结v30复放已完成，48/48重置重复输出完全相同。Nyx任意事件8/12；
Nick/Next/Mix各0/12。am_onyx三语速均无事件（峰值.040625/.01171875/.28203125），
bm_fable1.1无事件（.29609375），其余8条触发。
触发相对全生成末端偏移有负值（最早-350ms）；原生输出含尾部时长，不能解释为末音素提前。
完整scores/times/事件已保存于4090-synthetic-phone-pilot-v30-details-v1.json。
结果支持下一假设：显式音素本身不足以消除声音/速度覆盖缺口；扩大训练来源覆盖需同时
防止把这些已观察的训练声音当成未消费验证，并需以单独开发分区控制误触发回归。
未改变v30、阈值或校准；没有HiESP胜出/独立召回/FAR通过结论。

## 全训练声音显式音素扩充预声明

继承48条pilot的固定音素、未裁切生成和机械质量门槛；扩展到原Kokoro分区全部20个
training声音×0.8/0.9/1.0/1.1/1.2五语速×Nyx/Nick/Next/Mix=400条。
读取历史generation.json的training分区并断言与validation/testing无交集。
所有新样本明确标注training用途，不改变历史声音分区，不声称生成独立性。
相同声音/速度/音素可能复现pilot字节，允许训练内部重复但后续去重统计，
绝不把重复算作新增独立数据。生成全部后检查质量及长度，未通过者记录而不静默丢弃。
没有读取最终集或选阈值；本阶段不启动没有完整数据接入审计的GPU训练。

扩充生成同时新增序列化与上下文覆盖审计audit_phone_training.py：
逐条读回原始npy/WAV核验SHA、格式、样本数和token/duration对应关系，统计pilot重复哈希。
teacher60/context150按149个实际使用特征帧、151 hops=24160样本支持范围，
用2秒前置历史的对齐相位逐条判断整个生成波形是否能被一个训练crop覆盖。
未覆盖不等于漏掉音素，但不得将其标称为完整源波形crop；不裁切、不重标。

400/400合成完成并通过质量与序列化核验，20个training声音与旧validation/testing无交集。
WAV长度12400–34000样本；318/400整个生成源可被当前模型有效上下文覆盖，82条不可。
按类别完整源覆盖：Nyx82/100、Nick86/100、Next71/100、Mix79/100。
这是包含生成静音的全PCM覆盖统计，不是发音完整率；不可覆盖者保留原始数据，不静默裁切。
400个WAV哈希互异，与pilot无字节相同WAV；但其中48条复用完全相同voice/speed/phones配置，
故字节差异不能证明新增独立语音。报告与全部清单4090-synthetic-phone-training-*。
下一步数据接入必须保留这些覆盖标记，增强时不得用固定crop误标完整短语。
本轮尚无GPU候选或独立留出结果，v30未改；未读取三个已评分最终集。

## 显式音素训练特征包预声明

全部400条quality-pass training源，各作clean、25dB白噪声、15dB白噪声+固定合成RIR三种。
RIR延迟[0,160,480,960]样本、幅值[1,.25,-.12,.06]，full卷积保留960样本尾部。
语音RMS .08，混合后仅整体降增益至peak≤.95防削波；每条种子由路径/variant固定哈希。
2秒前置/1秒后置，teacher60有效支持151hops；完整signal及回声尾部不适配则报告并不入pack。
只生成training二分类特征，不新增辅助头，不评分或选阈值；原始400条全部保留。
实现prepare_phone_features.py，完整记录每variant参数和pack索引。

训练特征包完成：1200个尝试variant中930个完整signal适配、270个因上下文不适配而报告排除。
保留全部400源；pack为930×150×40 uint16，Nyx240条、混淆词690条，覆盖20个training声音。
读回npz核验shape/dtype/二分类标签与全部930索引对应、完整支持区间及pack SHA。
pack SHA256 c2fd2212d9c8c648d0444e5fccce27d8c572c206280b1c70153f325e0f9e3c8a。
完整报告4090-synthetic-phone-features-v1.json；尚未训练或评分，不宣称模型改善。
后续可用独立RNG平衡二分类额外样本，与原v30主batch保持可审计的损失比例，
同时需要单独声明候选选择与新开发数据协议，不能把训练包当验证集选阈值。

## v35 显式音素二分类额外监督预注册

假设：完整源PCM且具声音/速度覆盖、噪声/RIR增强的额外二分类样本，
可改善v30在phone pilot的声音漏触发，不重用v34失败的四类辅助头CE配方。
原v30初始化/teacher60/context150/frozenBN、128主batch、prefix16全部保留。
每步独立RNG87726额外取16条：8Nyx、8混淆词（Nick/Next/Mix等概率），类内等声音再等行抽样。
总目标 main128 weightedBCE + .1*extra16 weightedBCE，负例仍weight20；
拼接144时以144/128与.1*144/16权重补偿Keras分母，代数核验通过。
原主batch样本/随机数生成与representative输入函数不改，不添加辅助头；独立额外采样不用于校准。
120k步，每5k用原历史开发协议选checkpoint（明确为已消费开发，不冒称新独立验收）。
该训练结果仅候选研究；之后仍需新开发阈值协议、新冻结合成集和负曝光门槛才能升级v30。
原三个最终集不访问；两步GPU全流程smoke通过后才启动正式任务，模型/报告hash完整保存。
代码phone_binary.py、train_stream.py；命令experiments/v35_phone_binary.sh。

v35两步CUDA全流水线smoke已终结exit0：初始权重与classifier初始化SHA匹配v30，
BN初始/选中snapshot完全相同，H5全部数据集有限值；22801参数，单[1,3,40]输入、[1,1]uint8输出。
smoke模型54312bytes（不要求新导出字节与v30相同）；原v30及其交接包未修改。
保存4090-v35-smoke-training.json与4090-v35-smoke-audit.json；仅管线检查非性能证据。
按已预注册命令启动正式120k训练，额外batch16，总代数loss与主batch保留检查通过。

## 合成新开发波形协议预声明（评分前）

v35训练仍运行同一会话，不重启。为后续比较准备80条新开发波形：
历史validation声音af_sky/am_liam/bf_alice/bm_george，速度.85/.95/1.05/1.15，
固定音素Nyx/Nick/Next/Mix与Hi ESP（hˈaɪ ˈiː ˈɛs pˈiː），4×4×5。
声音与training/testing分区均无交集；这些validation声音过去用于开发，因此新波形
不声称新来源独立或未使用过的声音。用途仅开发，不作为最终验收。
生成无模型评分，机械质量门槛与原生波形/token/duration保留同training。
生成完先冻结全部哈希，随后方可评分；阈值仅按预定开发网格与连续开发负音频约束选择。
比较需完整保留所有事件，并以整个生成片段末端作为明确的合成参考；
原生输出中的静音使该参考不等于末音素，不据负偏移推断提前音素触发。
已有三个最终集不读取。本新开发协议不是已完成的独立最终门槛。

新增v35_historical_regression.sh仅复现原开发密集网格、Mandarin/DiPCo/Libri-dev-other连续负例、
扩展校准与float/INT8探针，依赖正式training.json完成后才运行。
其阈值与分数明确为历史开发回归，不替代上述新合成开发或最终验收，未自动启动。

80/80新开发音频机械质量门槛通过并已读回核验；各Nyx/HiESP/Nick/Next/Mix16条。
清单冻结SHA256 10b5e060163f2eaf896ff8219cd9c1ff341c569f4e585b3a68828fb71bef2b94，
4090-synthetic-phone-development-v1.json及对应manifest保留完整配置、音素、时长、来源与哈希。
训练/开发声音分区无交集；没有最终集访问、唤醒评分或阈值选择。
v35同一训练进程已超过6500步；会话79588仍活跃，继续原任务不重启。

## 新合成开发双引擎基线复放预声明

在冻结80条生成音基础上，仅加1.8秒前静音和1秒后静音，逐样本确认源PCM未改。
reference=前静音+完整未裁切生成长度；固定[-100,+800]ms窗口仅诊断，
所有窗口外事件照常记录，不丢弃；生成尾部静音可能使有效事件落在窗口前。
v30固定.91796875/five-score原策略，HiESP固定.63，不选阈值、不声称匹配FAR。
原官方QEMU运行/CRC/全PCM哈希验证流程不改，输出短/tmp目录避免AF_UNIX路径限制。
准备器prepare_phone_replay.py；compare_paired_validation报告改为披露manifest的endpoint协议，
移除旧硬编码“12条/能量端点”描述，保持评分算法不变。

80条复放首次运行terminal失败于record27，UART只落盘NYX_PCM_DONE前缀，
旧runner以子串判断完成并立即quit QEMU，截断了计数行；stage104已达到，不等于计数完整。
失败报告4090-phone-dev-replay-partial-done-failure-v1.json，原/tmp日志保留，无整体分数。
修正为等待完整newline终止的DONE计数行；实存部分行不匹配、完整模拟行匹配检查通过。
按相同冻结输入/阈值重跑到新目录，保证80条runner来源SHA统一；不是观察超时重启。
v35训练会话不变、未重启。任何最终验收仍未进行。

修正runner后同80条开发复放完整终结exit0，80条官方stage104/CRC/计数与同PCM证据通过。
Nyx目标any-event9/16、固定生成末端窗口内5/16、窗口前4事件；HiESP any-event8/16、
窗口内0/16、窗口前8事件。全生成末端含尾静音，不能把0窗口命中当作HiESP零召回，
也不能把Nyx窗口5对0解释为超过HiESP。此处any-event仍仅合成意图诊断。
48条共同混淆负例Nyx2事件/2条、HiESP0；交叉目标负例各0/16。
原阈值不变，非匹配FAR；没有连续负例曝光上界或最终验收结论。
完整比较、逐条官方报告与manifest、any-event摘要已发布4090-*phone-development*。
v35同训练进程已19000步以上，继续原会话79588；本次基线未改变其训练或选择规则。

## v35后处理排队与官方缓存复用核验

score_phone_development.py逐条核验80条官方report的manifest/PCM/模型来源、阈值getter、CRC/stage，
不再重复运行HiESP；重算Nyx并复核reset。用v30基线验证80条双方事件逐条完全一致。
核验报告4090-phone-dev-cache-baseline-audit-v1.json；这是评分程序验证，不是新性能提升。
v35_after_training.py绑定当前训练PID及/proc启动时间，原进程终结后核验120k产物和模型SHA，
再运行historical_regression及新开发音历史阈值诊断。无完整产物即失败，不启动新训练。
后处理仅排队；训练未终结时不会读取半成品评估，不自动发布或升级模型。

## 新合成来源VCTK与声音保留预声明

检查本项目研究代码/报告未发现VCTK/lessac已用记录。固定rhasspy/piper-voices
revision ae9d8352b73c1b1750748843e753a504b00e05b0的en_GB-vctk-medium，
HF树声明ONNX 76952753 bytes，LFS SHA 4e9fc85ab9009385319fc6bae7f55577f8a2d7ee77fd9159a5500eb6531f41e6。
模型卡说明109个VCTK声音、22.05k、由Lessac medium微调。新权重/语料，不是新模型架构。
下载固定模型/配置/MODEL_CARD并记录全部SHA，按排序声音名+Python Random88726预分区：
20个development、89个reserved_final；不用于训练。声音ID不能证明所有物理身份或预训练无交集。
本轮只下载/冻结来源及分区，不生成最终音频、不评分或选阈值。
后续先用development生成质量流程和阈值协议；最终89声音须另行冻结协议后才生成/评分。
不以新来源替换既有最终集记录；原三个最终集继续封存。v35继续原训练任务。

VCTK模型与元数据固定版本下载完成，ONNX大小/LFS SHA匹配；三文件重新读回SHA均通过。
ONNX CPU成功加载，输入input/input_lengths/scales/sid，109声音分区20/89无交集。
来源报告4090-piper-vctk-source-v1.json，原模型卡4090-piper-vctk-model-card-v1.md，
接口报告4090-piper-vctk-onnx-schema-v1.json；未执行语音生成或唤醒评分。
保留最终声音仅为预先隔离，不表示最终协议或独立验收已经完成。

## VCTK开发生成预声明

固定20development声音×length multiplier .9/1/1.1×Nyx/HiESP/Nick/Next/Mix=300条。
Nyx文本固定Hey Nicks.，eSpeak en-gb-x-rp音素/token完整保存，未知音素直接拒绝。
scales继承配置noise .333/noise_w .333，length_scale=1.4×multiplier；不按唤醒评分调整。
ONNX只输出waveform，无duration/显式随机种子接口，明确记录不可得，不伪造token时长或字节复现保证。
保存全长22.05k原生float npy与16k PCM16，零裁切；有限值、0.2–5秒、RMS≥.001、
浮点abs≥1和PCM16饱和值均为0作为质量门槛，失败也保留清单。
只生成开发部分，不访问89个reserved_final声音。生成后冻结全部哈希，未评分前推送元数据。

VCTK开发300/300全部生成并通过机械门槛；另逐条读回npy/WAV核验SHA、采样格式、
ceil(native_samples×320/441)重采样长度、有限值及PCM16无饱和。
清单SHA256 dcf1bbd2ee183bd1bdb893f4662e94d74cc6554b2cc15651834bdc5e59f030da。
20个开发声音与89个保留声音无交集；后者未生成/评分。
4090-vctk-development-v1.json及全清单已冻结发布；尚无唤醒评分或阈值选择。
v35原训练进程已45000步，后处理仍等待同PID；未重启或使用VCTK做训练。

## VCTK开发基线双引擎复放预声明

固定已冻结300条VCTK开发源，prepare_phone_replay增加显式vctk协议，仅接受清单与分区固定SHA。
全PCM加1.8秒前静音/1秒后静音，读回确认原PCM逐样本相同；生成末端为参考。
长度倍率与速度分开记录，不把length_scale当speaking-speed。
v30.91796875/原five-score策略，官方HiESP.63，冻结阈值基线不选阈值。
全300条QEMU计数/CRC/stage及同PCM核验后才产整体报告；所有事件保留。
固定[-100,+800]ms窗口只是生成参考诊断，同时报告any-event，不能误读为末音素召回。
仅20个development声音，无89个保留声音或历史最终集访问。

VCTK加静音复放manifest冻结SHA779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be。
缓存候选评分器按两个明确manifest SHA分别支持80条Kokoro与300条VCTK，
仍逐条核验官方report/同PCM/阈值getter与来源；不放宽到任意来源数据。
本变更只扩展协议身份白名单，不改变分数/事件/端点窗口算法。

VCTK300条同PCM双引擎复放完整终结exit0，300条官方计数/CRC/stage全部通过。
v30目标any-event24/60、HiESP54/60；共同混淆负例v30为4事件/180条（4条），HiESP0/180；
交叉目标双方0/60。固定生成末端窗口v30仅4/60、窗口前20事件；HiESP8/60、窗口前46事件。
生成尾部使窗口内命中严重低估any-event，不将窗口值称为真实召回，全部事件保留。
这批新来源开发诊断清楚暴露v30的来源泛化弱项，但不是独立最终或匹配FAR结论。
完整比较、官方逐条reports、manifest与any-event摘要已推送4090-*vctk-development*。
v35原训练继续，尚无其VCTK候选结果，不能由训练loss推断此差距已改善。

VCTK官方缓存评分程序以v30再核验通过：300/300重置输出相同、双方事件逐条与原复放完全一致。
记录4090-vctk-dev-cache-baseline-audit-v1.json，候选评分可复用已核验官方报告而不重跑QEMU。
v35最新59000步，仍原进程；候选训练和后处理均未宣称完成。

## VoxConverse连续负音频来源准备预声明

研究记录未发现VoxConverse已用。主库joonson/voxconverse固定commit
24bf60be297701cd7e4ef18550c6d390c1b87365，README注明研究用途CC BY4.0。
官方test ZIP请求HTTP401，故明确使用diarizers-community/voxconverse镜像
revision3acfa1b45ca4b7419aee999d67d94c617f9c9d47，固定parquet LFS SHA/size与模型卡。
在音频/模型评分前固定候选分区：dev前2分片、test前4分片；本轮只下载dev0检查结构。
镜像声明diarizers预处理，不能声称官方原始PCM逐字节等同。说话人标签未必跨录音全局唯一，
没有跨语料物理人物隔离保证；无转写，目标短语不存在尚未认证。连续性/时长需另行审计。
不评分、不用训练、不下载所保留test；已有最终集不访问，原验收门槛不放宽。

新增audit_voxconverse_shard.py在完整下载及LFS SHA核验后，逐嵌入录音检查WAV格式/长度、
原编码字节与解码PCM16哈希、标注时间合法性和重复哈希，统计原样嵌入录音时长。
不重采样、不裁切、不做唤醒评分；时长只是候选曝光，不冒充已评分FAR证据。
标注时间合法性不证明原视频未剪辑，也不证明不存在目标短语。

VoxConverse dev0分片下载完整终结exit0，485283393 bytes/LFS SHA核验通过。
逐嵌入录音审计44/44均16k单声道PCM16、44个编码哈希互异、标注时间合法，
完整镜像录音总时长4.240854947916667h，不裁切/重采样，不做唤醒评分。
仅此时长即使未来0事件也不足以满足0.5/h单侧95%上界；不会把未评分候选时长当FAR通过。
下一步仍按预先选择补dev1及另行冻结最终负例协议；test候选分片目前未下载。
来源计划、主库README/镜像模型卡、下载证明与审计报告已发布4090-voxconverse-*。
这只是新的连续开发背景数据准备，非独立最终验收；未知目标短语与人物身份限制继续披露。

按已冻结VoxConverse来源计划补dev1，允许清单仅dev0/dev1，不接受reserved test分片。
449464339bytes，固定LFS SHA58fc6ff493c18018c97a7c7a0466db763dfef6a5f6367a3b9aec74355254b8eb。
下载器拒绝覆盖任何已存在文件，不因等待/超时重复启动；审计器增加显式开发分片参数。
两片均完成后将按编码/PCM哈希与录音路径查重复，再统计总原生时长；无评分前不报FAR。

新增prepare_voxconverse_features.py：在两片完整审计/容器哈希通过后，先核对跨分片录音路径、
编码SHA和PCM SHA均无重复，才抽取完整嵌入WAV并生成MicroFrontend缓存。
原WAV编码字节保持不变，PCM不归一化/重采样/裁切，不拼接额外静音；frontend内部末尾对齐不算曝光。
输出官方复放manifest与Nyx cache；共享schema中的speaker字段明确仅为recording ID，不冒充全局人物。
无转写的连续背景中任何事件都保留计数，不以可能包含目标短语为由事后删除。
新特征准备不评分或选阈值、不访问保留test分片；源码语法检查通过。

dev1完整下载/审计终结exit0：43录音3.9328177777777777h，容器LFS SHA通过。
两片合计87录音8.173672725694447h，路径/编码SHA/PCM SHA均无重复。
完整镜像WAV编码字节不变，87条前端cache生成完毕并逐条重读核验dtype/shape/feature SHA、
PCM SHA、采样数及原生曝光秒数。源音频不切分、无音量归一化，输出连续复放manifest已冻结。
报告4090-voxconverse-development-{combined-audit,cache,feature-index,replay-manifest}-v1.json。
仍无唤醒评分，8.17h不是FAR通过证明；保留test分片未访问。
v35最新100000/120000步，后处理等待同进程，未更改训练数据或候选选择。

## VoxConverse开发连续背景v30评分预声明

87条8.173672725694447h，v30原始SHA/threshold .91796875/sliding_window5，
不选阈值、不改音频；每录音重置，计入所有事件，原生秒数为曝光分母。
背景无转写，不能认证完全无目标短语；不事后剔除任何事件，报告全背景激活计数与名义Poisson上界。
仅开发，不解释为独立最终或新来源验收。比较可用原始scores后续开发网格分析，不用test。
共享evaluate_frozen_human增加显式identity-unit=recording，并披露cache来源限制，
只改标识解释和报告文案，事件/评分算法未改；旧使用默认speaker保持兼容。

v30开发背景完整87条评分终结exit0，全部score trace SHA与事件数重算通过。
8.173672725694447h中2事件，rate .2446880450/h，单侧95%上界.7702527166/h，
不满足.5/h门槛；不是证明真实总体FAR>.5/h。全部事件原样保留，不因无转写剔除。
完整报告4090-v30-voxconverse-development-v1.json，阈值未重新选择。
下一步官方HiESP同87条完整PCM固定.63复放，4并发，每条600秒host截止，
完整DONE计数/CRC/stage核验，任何失败不产整体summary。开发用途，不访问最终集。
实现replay_voxconverse_official.py，短/tmp输出；本次仅排队该新评估，不重启训练。

## 新开发阈值规则（v35结果前固定）

新开发输入：固定80条Kokoro、300条VCTK及87条VoxConverse8.173672725694447h。
网格为旧THRESHOLDS并[.5,.99609375]间1/256固定步长，不扩网格挽救。
合成主指标为Nyx片段呈现区间[前静音结束,完整生成末端+800ms]内有事件的比例，
按两来源等权macro；不是完整音素召回。此前窄末端窗口已显示生成尾静音偏差，
故评分前明确更改主指标，旧窗口/全部事件仍保留作诊断，不隐去差异。
可用点须：Vox背景全事件Poisson upper95≤.5/h；两合成来源全部负片段（含HiESP交叉词）零触发；
目标片段生成前零事件；macro≥冻结v30原阈值两来源均值.48125。
在可用点中先最高macro、再较低背景upper、再较高threshold；无可用点则明确失败，无回退冒充通过。
这是开发选阈值而非最终验收；旧v30固定模型/阈值不改，89保留声音/test负分片均不访问。
实现calibrate_phone_development.py，输入报告/score哈希严格核验。尚未对v35执行该选择。

新开发选择器以v30缓存全流程执行通过，原阈值.91796875点复现两来源trigger9/16、24/60、
负片段2/64和4/240及Vox2事件；并非评分代码引入的计数差异。
固定网格无可用点，selected/threshold=null；报告4090-v30-new-development-calibration-v1.json。
这是v30开发阈值研究失败，不修改冻结v30，也不扩网格或放宽条件来挽救。
v35仍原进程115000步以上，尚未用其分数选新开发阈值；官方背景复放同会话继续。

新增v35_new_development.py后续评估排队：绑定现有historical evaluator的PID/启动时间，
其终结后须完整120k训练/模型SHA及Kokoro新开发报告通过核验，才执行VCTK与Vox评分，
随后应用已预注册新开发阈值规则。任何缺产物或失败不重启训练/评估。
不改原训练、checkpoint选择或旧阈值，所有阈值选择仅使用已声明development数据；
未自动发布、未自动访问最终数据、未自动升级v30。

v35正式训练完整终结exit0，120000步1849.5秒，选中35000步；54440bytes/22801参数。
模型SHA897b5e0bccf1ed23e38d3c0492144c2c1a52dd0e1094ded78600b87e2993569b。
全部H5数据集有限值，frozenBN快照不变，单[1,3,40]输入/[1,1]uint8输出；
训练和导出审计4090-v35-{training,export-audit}.json。后处理原会话已启动开发回归，未升级。

HiESP新背景87条同PCM复放完整终结exit0，8.173672725694447h中1事件，
upper95 .5803834675/h；v30同曝光2事件upper95 .7702527166/h，双方均未过.5/h。
逐录音官方CRC/stage/manifest/PCM与Nyx特征来源/曝光关联全87条核验。
报告4090-hiesp-voxconverse-development-v1.json及4090-v30-hiesp-voxconverse-development-comparison-v1.json。
不以1对2事件解释匹配总体FAR或显著胜负，不访问最终数据，不额外加曝光挽救此次开发结果。

v35历史开发回归完整终结exit0：密集点.8984375，代理macro85.390625%、1830/2160、5早触发。
同点Mandarin0、DiPCo1、Libri-dev-other4事件；扩展网格无可用点。
最大.99609375仍Libri-dev-other1事件，代理macro62.5260417%，不扩网格挽救。
float/INT8探针reset通过，7/8事件步完全相同、正例4/4双方；MAE.0022927613、max.2273514271、
完整上下文MAE.0019446225。不是MCU数值等价。
新合成开发在历史密集阈值.8984375：Kokoro any-event12/16、共同负例0/48；
VCTK any-event22/60、共同负例5/180；各交叉词0。v30对应9/16、24/60，负例2/48、4/180。
阈值分别历史选定，不能将这些计数解释为匹配FAR公平胜负。VCTK泛化没有获得一致改善。
以上8份报告已保存4090-v35-*；新背景评分及新规则校准同一后处理会话仍在运行。
旧报告完成不代表v35验收或升级；原v30及Mac交接模型未改。

v35新开发后处理完整终结exit0：Vox87条8.173672725694447h在历史阈值.8984375中4事件，
upper95 1.1198783379/h；全部trace SHA与事件数重算通过。
两新合成来源+Vox的预注册138点新开发网格无可用点，threshold/selected均null。
不改网格、曝光或门槛，不访问89个保留声音/test背景分片。拒绝升级v35，继续保留冻结v30。
完整新背景/校准及比较4090-v35-*、4090-v30-v35-synthetic-binary-comparison-v1.json。
旧ACCEPTANCE_BLOCKERS增加当前合成目标说明：人工听审不再阻塞继续研究，旧描述保留为历史。
本结果否定当前额外二分类样本配方可直接升级，非所有合成或音素监督方案无效。
下一步先诊断VCTK来源的声音/速度/音量/长度失败分布，再设计新数据或训练变更，不重复此配方。

## VCTK来源描述性诊断与固定增益对照预声明

先用既有报告/源manifest作描述统计，不重评分：v30三length倍率均8/20；
漏触发/触发RMS中位数.04153/.04023，v35 .04041/.04163，未见明显单纯电平分界。
60条Nyx中仅1条全生成波形超过24160样本，不能由长度直接解释大多数漏触发，
且全生成长度含静音，不等于唤醒音素跨度。表4090-vctk-source-diagnosis-v1.json。
进一步固定增益.5/1/1.5，对全300条目标/混淆词复放v30/v35，原各自历史阈值不改。
所有源峰值×1.5<1，保证无削波，保留量化舍入与每变换PCM/分数SHA。
1×须逐事件完全复现旧基线，否则整个对照失败；不据结果筛音、选阈值或改生产预处理。
只用20开发声音，不访问保留最终声音；实现probe_vctk_gain.py。

固定增益对照全1800条model×source×gain复放终结exit0；1×逐事件复现基线，
所有变换无削波，1800 score SHA重新核验通过。完整报告4090-vctk-gain-probe-v1.json。
v30在.5/1/1.5增益：目标26/24/24（分母60），共同负例4/4/5（分母180）；
v35目标23/22/23，负例6/5/5。各交叉词均0/60。未测增益变化背景FAR，不部署任何增益修正。
在所测范围内无主要差距修复，不能无限扩大增益搜索挽救；也不声称所有录音电平因素均排除。
同1×误触发涉及Next/Mix及v35的Nick，完整误例保留，无自动重标/筛除。
下一研究方向优先训练合成来源覆盖，而非单纯加长上下文或调增益；
VCTK开发20声音不挪作训练、89保留声音不动，可研究不同权重/语料的英国英语Piper训练源，
以明确来源记录和小规模生成检查后再决定新训练，不重复v35同一数据配方。

## Alba 英国英语训练来源与机械小样

核对固定 Piper revision ae9d8352b73c1b1750748843e753a504b00e05b0 的模型卡：
Alba 为爱丁堡数据集10283/3270，CC BY4.0，从hifi_tts_92/Cori Samuel初始化；
Cori为LibriVox单声音，Alan许可需要进一步追溯。先仅固定Alba候选训练源。
模型63221984bytes，SHA1f9da809b6da1ef4190bb5dd6cce4f4af803274539e9cf9e0693085150bac3d7。
模型卡未标注VCTK来源，但不声称物理说话人或预训练全局独立；同Piper架构，单声音。
来源文件与ONNX schema保存4090-piper-alba-*，模型CPU载入及所有源哈希复核通过。
首次pilot-v1错误传入多声音sid，exit1，未产出音频；修正为schema实际三个输入后，
新目录pilot-v2完整exit0，4词×三速度共12条，12/12机械质量通过；全部PCM/native哈希复核。
保存完整未裁切22.05k float原音与16k PCM16，音素/token/scales/版本及生成边界，
ONNX无可用音素时长/随机种子，不宣称位级重生成或声学音素真值。
报告及清单4090-alba-training-pilot-*-v2；仅training用途，不评分、不移动VCTK声音。
这是不同命名语料的新训练来源小样，不是候选性能改善证据；尚未启动新模型训练。
下一步设计覆盖多个独立训练声音的来源组合及预注册采样配方，避免重复v35。
冻结v30及Mac交接归档未改，最终保留声音/背景分片未访问。

## v36来源平衡数据干预预注册与GPU冒烟

固定Cori high checkpoint，SHA470b4dd634c98f8a4850d7626ffc3dfc90774628eeef6605a6dd8f88f30a5903，
114219352bytes；模型卡称24h LibriVox、从头训练500epochs、public domain。
不假定Cori与Alba初始化Cori Samuel物理独立；它们是checkpoint来源组。
Cori训练小样12/12机械质量通过，PCM/native哈希复核，未使用任何VCTK声音。
Alba所有12条原音能完整装入24160样本支持区间；不据此推断音素边界。
新training-only pack固定400Kokoro+12Alba+12Cori，三增强共1272尝试，
1002纳入、270完整保留但不训练的超上下文变体；新增72裁片全纳入。
原Kokoro930裁片与标签逐值完全不变；pack SHA c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096。
新假设是checkpoint来源覆盖而非单纯加大v35同来源样本：先均衡三来源，再均衡声音。
原main128、extra16、.1附加BCE、negative20、seed87726、frozenBN及checkpoint选择不变。
训练逻辑默认路径保持原v35随机抽样顺序；新--phone-source-balanced显式启用。
完整预注册experiments/v36_british_source_balanced.md。
4090两步冒烟exit0，权重全部有限、BN前后一致，流式int8[1,3,40]到uint8[1,1]，
模型54312bytes（两步冒烟模型，不是冻结v30）；审计4090-v36-smoke-audit.json。
正式120k仅在冒烟审计通过后启动；现有已消费开发集只可作探索回归，不冒充新独立验收。
不得以已有开发集结果直接升级，三旧最终集封存、89保留声音和test背景不访问。

v36原训练会话持续运行（PID90180，/proc starttime1516616），已观察8500步；
不是训练完成或评估结论。新后处理v36_after_training.py绑定该PID及启动时间等待，
不启动/重启训练。训练完整终结后核对120000步、model SHA、固定pack、BN，
再执行可复用audit_stream_export.py，随后历史回归及Kokoro/VCTK/Vox开发回归。
对这些已消费开发数据的校准输出显式命名reused-development-exploratory，
只作探索筛选，不自动发布/升级模型，不访问最终数据。后处理日志v36-after-training.log。
导出审计脚本已在真实两步smoke模型运行通过，包含所有H5有限值和量化接口检查。
另独立审计默认v35抽样10批完全一致；v36实抽300批4800条得到
kokoro1542/cori1654/alba1604，原128输入逐值保留、8正8负及BCE缩放逐值检查通过。
审计4090-v36-sampler-audit.json。原训练不因观察超时重启。

## 新开发数据预留（v36结果之前）

在v36训练仍运行且未产生导出/新来源评分时，先提交270ad79固定SLR83新开发协议，
再生成音频。不改变已失败v30/v35的旧开发门槛或报告；旧开发回归不是新独立验收。
固定Piper southern_english_female low/native16k和northern_english_male medium/native22050，
两模型卡及OpenSLR83原始页面均指向UK/Ireland crowdsourced corpus，CC BY-SA4.0。
不是VCTK、不是当前训练源；仍不能保证物理人/全局预训练独立，只覆盖两个preset。
来源manifest SHA10f1c7b0351c6349fb200245b4ccb8a5cd68d929bd0759cfbc911838d51431d8。
原始页面保存在.local源目录，哈希入报告；许可和引用保留在协议。
30条=2声音×.85/1/1.15×5词，30/30机械质量通过；保留native float与全长PCM，
输入音素/token/scales/软件版本均保存，无可用音素duration，不声明声学音素真值。
源哈希/原生和PCM长度复核，30个PCM唯一；与804个既有训练/开发PCM无字节重复。
报告4090-slr83-development-{v1,manifest-v1,audit-v1}。当前未作唤醒模型评分。
同时固定VoxConverse此前未用dev2/dev3元数据（仅元数据，无下载/评分），
first2 remaining dev indices，报告4090-voxconverse-new-development-plan-v1。
新stage不把新曝光合并旧dev0/1来挽救旧FAR；背景完整性/重复审计是评分前条件。
新阈值规则、零混淆词误触发、FAR95<=.5/h等在生成前固定；最终集仍不访问。
v36原训练观察至24000/120000步；原后处理PID91294仍绑定等待，不重启。

## 新开发复放/连续背景准备完毕（尚未评分）

SLR83固定30原音制作90条（三条件）完整复放，干净30条原PCM逐样本保留，
25dB白噪/15dB白噪+完整960样本RIR，噪声以原音RMS为参考，不做RMS标准化。
预备replay-v1发现RIR噪声曾以混响RMS计算，评分前纠正并另存v2，v1保留、不评分。
v2 manifest SHA88cfa446880e8a562927221d552a544bb72f204af071cd6867c7e1284db80415，
全部WAV/PCM哈希、无饱和、源长度、完整尾音/参考边界与干净字节复核通过。
冻结清单与审计4090-slr83-development-replay-{manifest,audit}-v2.json。
评分工具仅增加这一精确manifest白名单，不放开最终音频或改变既有事件/阈值规则。

Vox新dev2/dev3下载原会话exit0，LFS大小/SHA全部匹配；原等待准备进程exit0。
86条完整native16kmonoPCM16 WAV，8.487218975694445h；无裁切/拼静音/重采样/标准化。
与旧dev0/dev1的路径、编码SHA、PCM SHA均无重复，保留镜像预处理/物理人重叠未知限制。
每条完整编码字节保存，全部86个特征SHA/shape/dtype和原WAV SHA二次核验。
新index SHA865960c9f88fc1d306ab3cfaed0cf3ce861df475375a84d25335263968f24305；
replay manifest SHAc65b0e566384bc3c54d36d4766d6e7abed731e01263354f5d05a924a454b2170。
所有来源/下载/单分片/跨stage/合并/前端/复放清单保存4090-voxconverse-new-development-*。
目前无唤醒模型评分或FAR结果；新曝光不得并入旧开发来挽救旧门槛。
v36原训练仍在运行，原后处理等待保留；先完成探索回归再考虑消费新开发评分。
最终保留集未下载、未生成、未评分；冻结v30及Mac交接不变。

## 跨端通过 Issues 沟通

用户指定GPU端与Mac/P4端改用仓库Issues沟通。检索现有open/closed列表无对应议题后，
建立 https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13 作为统一交接与研究同步线程。
已发布冻结v30下载地址、归档/模型SHA、manifest/验证记录、原触发参数、Mac回报项目，
以及v35拒绝/v36训练中和新开发数据未评分状态。没有声称P4已通过或v36可替换。
后续重要研究产物/拒绝或升级决策在该Issue同步；Mac/P4实测结果在该Issue回报。
用户明确授权通过Issues沟通；无需再次请求发布研究更新许可。

## 新开发校准与生成参考报告工具

实现synthetic_presentation.py：核对manifest原记录、每条完整INT8平滑trace哈希、
长度（MicroFrontend30ms/10ms、模型3帧、平滑5）、逐事件重算与原报告一致。
按呈现区间lead..generation/RIR end+800ms报告触发、共同/交叉负例、提前/晚事件，
按声音/增强条件macro，分别列原生生成终点与完整混响终点延迟，不称音素真值。
已用旧VCTK300条×v30/v35两模型各完整核验；重算24/60、22/60，官方均54/60。
此为已消费开发数据重算，不是新性能实验或匹配FAR结论。
审计4090-synthetic-presentation-metrics-audit-v1.json保存来源/模型/报告SHA及摘要。

实现calibrate_slr83_development.py，严格绑定SLR83新manifest及dev2/dev3 index，
90合成/86背景完整trace，frozen-v30 SHA及.91796875，固定138点与新baseline最低macro。
FAR95<=.5/h、零共同/交叉负例误触发、零pre-generation，以及无可用点输出null，
排序macro、lower upper、higher threshold与生成前预注册一致，不加入旧曝光。
5个数值边界/门槛测试通过，涵盖起止包含、晚事件排除、两种参考相差60ms、
早事件/两种负例/FAR或baseline macro任一失败均不可通过。
新校准尚未执行：v36原训练仍在进行，先完成原后处理与探索回归再决定消费新开发。
Issue13尚无Mac回复；当前没有P4通过结论。

新开发运行入口experiments/v36_unseen_development.py已实现，但未启动。
它要求完整120k、对应导出审计/reset探针和已有开发有可用工作点，再允许消费新数据；
已有开发明确失败时先保留新数据并评审拒绝，不自动拿新集救旧失败。
入口覆盖同PCM官方合成基线、v30重复一致性核验、v36合成、双方完整新背景、
新门槛校准以及官方新背景复放，不自动访问最终集/发布/升级。
官方背景runner仅增加新manifest精确白名单（86条），旧87条协议不变。
生成参考报告支持同模型选定校准阈值，从认证完整trace重算，不把历史阈值冒充新阈值。
用现有v30冻结校准覆盖此路径，全部300条两引擎摘要与既有报告逐值相同。
扩展为6项关键测试：新增trace改变/截短/NaN拒绝；全部通过。
这些检查未消费SLR83或dev2/dev3模型分数，也未启动第二个训练/评估实例。
v36原训练已观察80000/120000步；原后处理保持等待，最终结论尚待完成。

v36原训练会话完整终结exit0，120000步1713.2秒，选择100000步checkpoint。
模型54440bytes/22801参数，SHA73b3ee6ea081fa23884ab99c6254fb5174c0a7dbc35561b198349f13cdae143b。
全部保存权重有限、冻结BN前后一致，int8[1,3,40]到uint8[1,1]导出审计通过。
原后处理PID91294已进入历史开发回归，未启动重复评估。
原始training.json与导出审计保存4090-v36-{training,export-audit}.json。
这仅是训练/结构审计通过，不是性能验收或升级；SLR83/dev2/dev3新评分仍未消费。
再次核对冻结v30为54440bytes及指定04aa4564... SHA，无修改。

## v36完整开发回归：拒绝升级

原后处理会话exit0、PID91294消失，全部既定回归完成；没有重启。
历史密集阈值.95703125，代理macro87.2135417%、1886/2160、3提前事件。
Mandarin3、DiPCo0、Libri-dev-other4事件；历史扩展138点无可用点，
最大.99609375仍Mandarin1和Libri1，不能靠扩网格挽救。
流式探针reset通过、7/8逐事件步完全一致、正例双方4/4，
MAE.0016182827、max.5523370653，完整上下文MAE.0023402058；不等于MCU数值等价。
合成历史阈值计数：Kokoro目标any-event13/16、共同负例1/48；
VCTK目标40/60、共同负例20/180（Next11/Nick5/Mix4），双方交叉HiESP均0。
与v30/v35分别历史阈值下计数相比仅是描述，不能称匹配FAR的提升。
Vox旧开发87条8.173672725694447h，6事件，upper95 1.4488463203564517/h。
固定两合成+Vox138点无合格点，selected/threshold均null；网格最小upper95仍.7702527166/h。
全部380合成+87背景=467个trace SHA与历史阈值事件重算通过。
完整原始回归/校准/探针报告及拒绝决定保存4090-v36-*，不丢弃误例或自动重标。
因此拒绝v36，继续冻结v30；SLR83/dev2-dev3未评分，89最终声音/test分片未访问。
来源均衡补充配方增加目标触发的同时混淆词也多，未满足判别/FAR门槛，
不证明所有新来源无效。下一步优先审查主训练正/混淆标签与新完整源监督之间的冲突，
不能重复v31窗口挖掘、v32只删除prefix、v33只屏蔽prefix或v34辅助四分类作为新方案。

## v37完整源主监督假设与训练侧审计

仅检查训练features index9db652...所有正例及原25个jitter的机械完整源覆盖：
Piper509618/1500000抽样、Kokoro56397/75000、Qwen5130/7500，等engine比例59.19018%。
该审计只以lead_frame和原samples核对完整PCM，不含RIR额外尾音；
生成静音使其不等于音素完整性，不能据此声称旧标签错误或翻转标签。
报告4090-training-full-source-support-audit-v1.json保留公式参数/分engine计数/例子。
v36主正例BCE质量.25与完整源补充.05相比，后者只有前者五分之一；
这提示可以检验整个phrase监督分布替换，不能当作已证明的失败原因。

预注册v37_full_source.md：直接用已固定1002全源裁片替换主32正/48混淆负例，
保持48连续ambient和negative20、teacher60/context150/frozenBN/初始化/优化器/120k。
独立RNG97726，三checkpoint来源均衡，来源内声音均衡，三混淆词均衡。
原main RNG仍完整抽样再替换，prefix0不再给生成正例的proxy前缀另贴负标签；
不加补充BCE/辅助头，不挖掘窗口。与v32只删prefix但保留旧phrase监督不同。
它同时改变整个phrase训练分布，训练源覆盖较旧主集少，不能归因于端点单因素；
仅机械完整源/发音意图监督，不声明声学音素真值。新开发/最终数据不参与训练。
50批确定性、所有裁片属于固定pack、32/48标签以及48ambient槽位保留检查通过。
4090-v37-sampler-audit.json；4090两步完整导出冒烟已启动，待完成审计后才正式训练。

v37两步4090冒烟完整exit0、26.5秒，全部权重有限、BN不变，流式接口int8[1,3,40]到uint8[1,1]。
配方核对实际batch128、无phone补充BCE、full_source seed97726、prefix0。
导出审计4090-v37-smoke-export-audit.json。smoke不是性能结论。
只有该审计通过后启动正式120k；仍不消费新开发/最终数据，v30保持冻结。

v37正式原训练PID103771/starttime1736905观察至9000/120000步。
已启动唯一后处理PID104582，绑定该PID/启动时间等待，不启动或重启训练。
v37_after_training.py校验完整120k、batch128、full_source seed97726/固定pack、
phone补充为空、prefix0、BN不变及model SHA后才执行导出审计和同旧开发回归。
继承原Kokoro/VCTK官方同PCM缓存及Vox完整流；固定网格，不自动使用新开发/最终数据。
脚本语法检查通过。低训练loss不是性能或泛化证据，正式结果仍需完整回归。

## 收到Mac/P4实测与Mac350已消费开发回归（Issue13）

Mac已在Issue13回报d8e24475...的P4单fixture/idle证据，以及1234fb1e...的
350条既有Mac合成回归；GPU端匿名下载公开包并校验全部成员：
probe包159107bytes、SHAd58df89b1090e5976fcce9d3bed51ee9ecc45a93c9bc8ec9528976d3a1cd38b2、43成员；
Mac regression包235455bytes、SHAc0abb4e489b2be3ea752b2ec64efb764bb91720190971d7d167b09c47c420a61、24成员。
两包模型原字节54440/04aa4564...一致，CONTENTS完整覆盖、逐成员大小/SHA通过。
接收记录4090-mac-evidence-receipt-v1.json；原始Mac紧凑报告/审计按原字节引入本分支。

Mac报告单P4 fixture179输出字节与事件step144一致、两120秒idle/capture通过，
但accepted=false，不涵盖整库P4数值一致、识别验收或active-call资源。
Mac350回归v30 host11/50目标触发 vs retained HiESP P4 48/50；
普通句子双方0/100、Nyx混淆v30 3/90 vs HiESP0/90；ESP混淆的发音未确认，不能合并作公平胜负。
GPU独立核验NPZ所有350条94940个uint8输出SHA/数量，再按五分数整数sum>=1175、
2*sum<1175重武装、67步refractory重算全部事件，host与firmware保存event steps完全一致。
复现声音计数Daniel0/10、Karen4/10、Moira4/10、Samantha0/10、Tessa3/10。
审计4090-mac-saved-output-audit-v1.json明确GPU未重跑PCM/前端推理或硬件。

将Mac350登记为已消费描述性开发回归：不得训练或选阈值，不是新的独立最终集，
不改v30阈值、不删失败声音、不以单fixture说明350条v30都已实机测过。
其prime56160及480/512块协议与GPU简短padding不同，后续候选必须沿用其冻结原PCM/prime，
并按各自已选阈值报告来源/协议差异，而不是把不同协议摘要直接合并。
v37原训练不改、不重启，现有新开发和最终分区仍保持未评分。

### Mac350 cross-host CPU diagnostic (2026-11-11)

Issue #13 requested actual native execution beyond saved-output auditing. Independent
CPU replay uses frozen v30 and all 350 authenticated original WAVs, 56160-zero
prime, 480-block tail alignment, per-stream frontend/model reset, BUILTIN_REF
one-thread LiteRT, unchanged speed and threshold. All 94940 outputs retained
locally at `.local/wakeword/mac-cpu-replay-v1/scores.npz` with SHA in report.
101/350 frontend hashes differ from Mac; only one output stream differs, with
two output bytes different. First output difference is Tessa-1-3.wav index271,
Linux5 vs Mac8. All fixed integer-policy event steps agree. Mac frontend arrays
are absent from its archive, so first differing frontend element is not yet
located. This is consumed diagnostic evidence, not new acceptance; actual
runtime versions and per-source hashes are in results/4090-mac-cpu-replay-v1.json.
No v37 restart, calibration, retraining, or new dataset consumption.

### v37 full-source training completed (2026-11-11)

Original training session exited0 after120000 steps; selected historical float
checkpoint85000. Original training/export audit copied unchanged to results.
Model54440B SHA09c5191e9b8a1ebe5ebfdfd62e22ea49f520d0eae4fe826910ec614187705d06,
22801parameters,1459.0s report runtime. Export audit confirms finite saved weights,
frozen BN unchanged, int8[1,3,40] to uint8[1,1]. Original queued postprocessor
PID104582 authenticated recipe/batch128/pack/seed/model and began historical
regression. This is export evidence only, not accuracy/FAR/MCU equivalence;
no promotion or unused development/final exposure.

### v37 rejected after complete exploratory regression (2026-11-11)

Original postprocessor session2676 exited0; no duplicate jobs. Original reports
copied unchanged. Historical INT8 threshold.99609375 was explicitly nonviable:
544/2160 proxy positives, macro.285677; hard-negative false clip1.5625% and max
source2.1354%. Mandarin/DiPCo/Libri-dev-other zero events at that threshold.
Kokoro target any7/16 with3/48 common-negative false clips; VCTK14/60 with9/180.
Old Vox87 recordings/8.173672725694447h had zero events, upper95.3665099367309777/h.
The fixed138-point two-source/old-Vox grid selectednull: max synthetic macro.4
below.48125 and minimum12 negative false clips, despite passing background.
Calibrator authenticated all467 saved score traces. No new SLR83/dev2-dev3/final
exposure. Streaming probe reset passed, MAE.00079089/max.241607; float1/4 vs INT8
0/4 probe hits, not a general quantization-loss conclusion. Reject v37 and retain
v30; no threshold extension, voice removal or endpoint/phonetic causal claim.
Full phrase-distribution replacement alone did not meet the synthetic goal.
Further work should diagnose score/representation and same-checkpoint quantization
before another long training run, preserving all failures and source splits.

### Same-checkpoint v30 Mac350 float/INT8 diagnostic (2026-11-11)

Completed all350 consumed original Mac PCM streams on CPU, same authenticated
native frontend as prior Linux replay, original saved_model, explicit float
state-zero reset,56160prime/480tail and unchanged.91796875 threshold/smoothing5.
All float outputs retained locally, hash3b1a18df275afca363ddfc908bec05041f6fa0b2ae7f17c0d6bbe837948c4a0d;
full per-file SavedModel hashes, provenance and per-source events/maxima/errors
in results/4090-mac-float-int8-v1.json. First stream reset-repeat exact.
Float12/50 vs INT811/50 target intent clips; same3/250 nontarget false clips
(Tessa HeyNext variant3, HeyNexus variants0/3); zero on50 HiESP cross-target.
Only float-only target is Moira-3-0 HeyNicks?: float maximum five-mean.9440673,
INT8.91328125, below threshold. No INT8-only target. Most misses and all these
false clips persist in float, so quantization is not the main explanation on
this consumed fixed-threshold corpus. This does not prove phonetic correctness,
float-optimal threshold, general FAR, new acceptance or hardware equivalence.
No retraining, threshold selection, new development or finals used.

### v37 training-window/stream-support diagnostic (2026-11-11)

Pinned1002 training-only complete-source crops: offline float window at both.5
and historical.99609375 classifies258/258 positives and744/744 negatives correctly.
INT8 isolated150-frame feature replay from zero state: last score above historical
threshold203/258positive and2/744negative; maximum individual score221/258 and25/744;
maximum five-mean172/258 and2/744. At.5 five-mean219/258 and27/744.
Original per-crop results/scores retained; report records hashes. This suggests
single-window fit does not ensure sustained streaming support even on training
crops. However window float versus stream INT8 confounds state/phase/quantization;
feature crops start with zero model buffers rather than original stream history.
It is not full PCM accuracy or isolated causality, and neither new acceptance nor
threshold selection. Next diagnostic should separate float streaming/window and
INT8 on identical crops before changing training supervision or context.

### v37 float-stream support decomposition (2026-11-11)

Same1002 authenticated training crops, original saved_model, per-crop zero float
state, first reset-repeat exact. Historical.99609375 threshold: float-stream last
203/258positive,2/744negative, identical aggregate INT8 last counts; float max222
vs INT8221 positives; float max five-mean178/258 vs INT8172/258, both2/744negative.
At.5 float/INT8 five-mean both219/258 positives,27/744negative. All offline window
float remains258/258positive,0/744negative at both thresholds. Max window versus
float-stream-last error1.0. Thus isolated-crop support gap already exists before
quantization. However this comparison intentionally does not shift the known
window/stream phase1 or restore original frontend/model prehistory. Five-means
include incomplete-context startup. Do not call this a newly proven export bug,
isolated endpoint cause, or complete-PCM accuracy. Preserve frozen model and all
scores. Next recover original full training-feature streams and compare aligned
windows/stream phase and temporal support before changing training.

### v37 complete training-history reconstruction and phase audit (2026-11-11)

Reconstructed all1002 included original augmented training waveforms/features;
all1002 crops exactly equal frozen pack, including noise seeds/gain/full RIR.
Full float streams from original history and phase1-aligned offline windows:
max score difference.010274052619934082 (not byte parity). Full-history target
maximum score232/258, maximum five-mean218/258 at historical.99609375; negatives
five-mean Mix1/247,Next2/224,Nick1/273. This raises positive five-support from
isolated-crop178 to218, demonstrating that omitted history materially changes
that diagnostic; remaining40 positives still fail five-support. Maxima over whole
streams are descriptive, not event-policy acceptance. Prior unaligned max1.0
cannot be called an export defect; aligned residual is not zero and retained.
All full raw/float/aligned arrays saved locally with report SHA; no new development,
finals, threshold selection or model change. A next training hypothesis may expand
end-position/phase coverage using only whole-signal-valid windows from these exact
reconstructed streams, rather than another identical fixed-end crop repetition.
Preregister counts/exclusions and keep original independent gates unchanged.

### v38 temporal-source coverage started (2026-11-11)

Preregistered before formal training at752dd20. Offsets0..14frames after original
end, keep entire generated signal/full RIR inside effective150frame support.
15030attempts→13448valid/1582excluded; all1002 offset0 crops identical; prior270
unsupported variants remain excluded. Base-example sampling unchanged, then uniform
valid offsets via independent107726 RNG, preserving long-source mass.50batch
repeat exact and original v37 source/base RNG states identical. Ambient code and
weights unchanged. Two-step4090 smoke exit0,38.7s,finite weights/BN unchanged,
INT8[1,3,40]→UINT8[1,1],54312B SHA93de6f90dd4c8a129eb6e2653ca8d47b00e65ebb7a27ceaacc646d7f99ca1744.
Formal120k runner session84703/PID118009 started once; same old-stage gate,
no new/final data and no automatic promotion. Unique postprocessor verifies temporal
pack/seed plus original full-source recipe before any historical evaluation.

### Mac shared-sequence helper on actual v37 (2026-11-11)

Issue13 Mac3baa61e provided shared-variable continuous scorer, imported unchanged
helper SHA1df37cc65e14a81c9b41e7e52c262e4d32cc514859a1459344c2401dbb425f55.
CPU actual v37 weights and all1002 previously saved training streams: normalized
raw/25.6 clipped0..26, phase1, stream output49 onward.90115 mature outputs compared;
sequence vs saved GPU float stream max4.649162292480469e-6; sequence vs original
batched aligned windows max.010274291038513184. All22801parameters/shared variables
and weights unchanged, optimizer0. Does not erase prior real-weight residual or
prove backend cause, gradients, INT8/MCU or recognition improvement. No new/final
sets or audio regeneration. v38 continues unmodified; future five-score-loss
experiment must be separately preregistered after current experiment completes.

### v38 training completed and export authenticated (2026-11-11)

Original session84703 exit0 after120000 steps; report1445.3s; historical float
selection15000.54440B SHA5bf6c0900ab6a05846f752bdd9e3d52a9b18d4be698e0a7e7e918a6985c41aca.
All saved weights finite/BN unchanged, INT8[1,3,40]→UINT8[1,1]. Original training
and export audit copied unchanged. Original postprocessor PID118594/session79006
verified temporal pack/seed107726/base recipe and started historical regression.
No new/final data, threshold extension or automatic promotion. Export evidence
only, not accuracy/FAR/MCU acceptance; retain v30 until full gates pass.

### v38 temporal coverage rejected (2026-11-11)

Original postprocessor session79006 exited0. Original reports copied unchanged.
Historical INT8 .99609375 nonviable:606/2160proxy positives/macro.367448,
hard-negative false1.9886%/maxsource2.7083%. Kokoro any11/16,commonfalse4/48;
VCTK any16/60,commonfalse11/180,cross0. Old Vox87fullstreams/8.173672725694447h
zero events,upper95.3665099367309777/h. All138points reject/selectednull;maximum
syntheticmacro.6854167 but minimum15 negative false clips, so no usable threshold.
All467 saved traces authenticated by calibrator. Probe reset passes,MAE.001607912,
max.177049,float3/4 vsINT83/4 probe hits; not MCU parity. Expanded temporal coverage
alone failed confusion gates. No newSLR83/dev2-dev3/finals or promotion. Preserve
v30. A separately preregistered sequence-loss experiment can now follow actual
shared-sequence mapping evidence; don't merge it into v38 or relabel failures.

### v39 preregistration and stable five-score gradient validation (2026-11-11)

Independent v39 plan uses v37 baseline128 BCE plus weight1 sequence16 supplement
(8positive/8confusing,negativeweight20), not v38 jitter. Source/voice/phrase balanced
eligible162frame segments yielding five scores; original whole generated signal
fits all five.817included/185excluded from supplement only; all1002 remain main
BCE. No Mac350/new/final training, no score-based inclusion. Pack SHA
 e5e731ff9fe390100b58d153002648cc238edf6e7daefcc58ea39929d256efc7.
Mac e32f0d6 flagged probability-BCE saturation before any v39 optimization;
sequence_loss.py imported unchanged SHA d023acc12ba808df4273d134bec33dbbdfd19dcf17b8a0485fd3fd7fb8944dda.
Stable logsumexp(logsigmoid) computes exact BCE(mean probability), not mean logits
or mean BCE. +/-20 wrong-label tests loss20/eachgradient+/-0.2. Actual v37 first16
training segments: all15 variable gradients match independent overlapping-window
logits, max1.1368683772161603e-13; gradientL1.000107626, sigmoid logits agree with
probability scorer; all original variables unchanged,optimizer0. Saturated examples
make small gradients; no general training/recognition claim. Training integration,
GPU two-step export and formal120k remain to run before any candidate gate.

### v39 integrated and formal training started (2026-11-11)

Main128 v37 weighted BCE plus sequence16 stable BCE(mean five probabilities),
same Adam once per step; new graph shares original variables. Separate original
window copy used for export to avoid upstream mode mutation of shared graph.
50 sampler batches exact; discarded CPU plumbing update confirms optimizer delta1,
finite loss/original weights change/BN exact. Two-step4090 smoke session66219 exit0,
37.8s,finite saved weights/frozenBN/int8[1,3,40]→uint8[1,1]. Smoke54560B SHA
 e7114a82e8d66f850809629771d252d8c46d6167389bf9e2ad8bed77c88ba85d.
Formal session77297/PID129206 started once after audited integration e491229.
Unique postprocessor authenticates seed117726/sequence pack and no temporal jitter
before original historical/synthetic/oldVox138-point exploration. No new/final data,
threshold extension or automatic promotion; frozenv30 retained pending full gates.

### Mac independent v39 update evidence received (2026-11-11)

Issue13 Mac68c210b independently checks e491229 compiled update on random CPU
teacher60/frozen nondefault BN with non-saturated gradients. GPU anonymously
retrieved3340201B archive SHAea112859514fcf54027157a594cca03705c2e165416fe53f25d3b6c1a75d6814,
verified exact9member CONTENTS and every payload size/SHA; training and two sequence
helpers byte-identical locally. Original report imported unchanged and receipt
added. Reported combinedloss15.84808445,sequencegradientL1264.9831,15combined-gradient
maxdelta9.5367431640625e-7,31updated arrays exact under same incoming gradients,
iterationdelta1/frozenBN unchanged. GPU did not rerun this Mac random experiment;
not actual-GPU/recognition evidence or independent-gradient-exactness claim.
Current formal v39 remains unchanged; no new/final data.

### v39 formal training and export completed (2026-11-11)

Original session77297 exited0 after120000 steps,2733.1s onRTX4090; selected120000
under unchanged historical checkpoint rule. Original training report copied unchanged.
INT8 model54688B SHA eaa826469fe4669a8b0c7ff355a1a26f07589b691478ad9a2955c4f266a19096.
Export audit confirms finite saved weights,frozenBN unchanged,int8[1,3,40] to
uint8[1,1]. Unique original postprocessor authenticated recipe and started historical
regression. Export contract only; no accuracy/FAR/MCU parity or promotion claim.
No new/final data consumed; frozenv30 retained pending full gates.

### v39 rejected after completed fixed regression (2026-11-11)

Original postprocessor session55928 exited0. Historical threshold.8984375:
Kokoro target any9/16,common false2/48;VCTK8/60,false6/180,cross0.
OldVox87 full streams8.173672725694447h,0events,upper95.3665099367309777/h.
Authenticated467traces;fixed138point grid has0eligible points,max synthetic macro
0.3645833333333333,minimum negative false clips8.
Sequence loss alone failed joint positive/confusion gate. Preserve original reports
and rejection;no newSLR83/newVox/final scored,no promotion,frozenv30 retained.
Mac is separately building six-engine complete-container training-only candidate
features under6f2cea3;not imported or mixed into v39. Future intervention needs
verified data and a separately frozen hypothesis/recipe,not blind weight changes.

### Mac complete-container candidate received (2026-11-11)

Anonymously downloaded975895a archive91611008B SHA
26d58635fa21c022245a1ccdadee44299ce29a02e815bf1701a4c7ef160cb396.
All7795 payload size/SHA and safe unique regular paths verified. Independently
checked7782saved raw arrays,labels,crops and38910overlapping windows/sample bounds.
2594canonical sources retain2632alias rows. Actual included engines are five:
Kokoro,macOS,OpenAI,Piper,Qwen; earlier six-engine shorthand was inaccurate for
this included pack. Original Mac report/audit preserved unchanged.
Original WAV binding and cross-host30witnesses still pending; no model scoring
or training/new/final access. This receipt alone does not authorize promotion
or establish source independence,pronunciation truth or frontend parity.

### Original training WAV binding and cross-host witnesses (2026-11-11)

Anonymous original synthetic archive395739341B SHA
47bb80ce97b271997864ae531640f65d437c685259de223f0b5154016348549c
verified;only2594canonical training WAVs extracted. All file/header/PCM hashes
match. Original Mac auditor aborted on frontend element mismatch;failure log
preserved. Separate fixed30witness pass retains all differences:30/30rendered
PCM exact,16/30full frontend arrays exact,14differ. No byte-exact frontend parity
claim; no model inference/training or validation/new/final WAV access.
Future training must bind one explicit feature environment/data hash and retain
this cross-host limitation rather than silently replacing saved arrays.

### v40 data-coverage hypothesis frozen before integration (2026-11-11)

Preregistered v37-baseline original128BCE,half phrase slots retained British,
half replaced with five-engine complete-source candidate;no v39sequence loss.
Independent127726sampler preserves base RNG/ambient. Equal engine/conditioning
identity/canonical-source/variant,negative Nick/Next/Mix balanced;aliases once.
Pinned Mac saved features retained with measured Linux frontend limitation.
No optimization or candidate scoring yet;sampler audits and4090smoke required
before120k. Original138point gate,new/final isolation unchanged.

### v40 sampler integrated and GPU smoke passed (2026-11-11)

Independent50batch/2000draw audit reproduces phrase/engine/identity/canonical
source/variant indices exactly;labels exact,retained British and48ambient exact,
base full-source RNG unchanged. Pinned Mac saved features normalized once.
Original128BCE/representative export now uses preregistered half-slot replacement.
Two-step4090smoke session88884 exited0;export audit confirms finite weights,
frozenBN unchanged,int8[1,3,40] to uint8[1,1]. No recognition/FAR claim.
Formal120k runner ready after these checks;new/final untouched.

Formalv40 started once after911937b:session44285/PID147224,starttime2531106.
Unique postprocessor session66707 authenticates exact training identity,120k,
coverage pack/seed127726,original base seed97726,no sequence/jitter,frozenBN
and modelSHA before unchanged historical/synthetic/oldVox138point evaluation.
No automatic promotion/publication or new/final data access.

### Standalone Linux/Mac frontend coefficients compared (2026-11-11)

Mac4e7ad51 driver copied unchanged;authenticated PyPI2.0.2sdist compiled in
isolated CPU directory without changing installed wheel orv40. All original
header/translation-unit hashes match Mac. Linux tableSHA
1336813ab90ed846b095af1d840b6b4f77ad299be0cd127b342abdddbd28876f.
Against defaultMac:only weight2112471vs2470/unweight2111625vs1626 differ;
index211belongs block33(start208,width12). Window/all other integer tables exact.
22diagnostic mel float bit entries differ. No-contractMac additionally differs
at293. This is a concrete coefficient discrepancy near observed channel32/33,
not proof of installed-wheel/P4/runtime causality. v40unchanged;no model/audioIO.

### Installed Linux frontend confirms all30 Mac intervention predictions

Mac29d56cd single-pair211 intervention predictions received and bound to old
GPU report hash. Re-rendered fixed30 training inputs with actual installed
pymicro-features2.0.2:30/30 complete raw arraySHA match predictions,no residual.
Installed micro_features_cpp.abi3.so SHA
7c88ffb22e174fcd1fad58d4e2a850452a192f9ca32448e47d13fd3ebadb32f4.
Supports the single-pair explanation for these30witness differences only,
not P4 or recognition causality/universal parity. No v40/wheel/data changes.

### v40 training and export completed (2026-11-11)

Original session44285 exit0,120000steps,1909.3s,selected45000under unchanged
historical checkpoint rule. Model54440B SHA
0b48cd96e8837b27c2df73c709c3360da9f06079f61eb9464922bae438bb98e4.
Original training report copied unchanged;export audit verifies finite weights,
frozenBN exact,int8[1,3,40] to uint8[1,1]. Unique original postprocessor
authenticated recipe and started historical regression. No accuracy/FAR claim
or promotion;no new/final data consumed,frozenv30 retained.

### v40 rejected after fixed joint gate (2026-11-11)

Original postprocessor66707 exit0;467saved score traces authenticated.
Fixed138points:0eligible,maxmacro.7270833333333333,min14negative false clips.
Historical threshold0.99609375:Kokoro{'target_any': [8, 16], 'common_false': [4, 48], 'cross_false': 0},VCTK{'target_any': [30, 60], 'common_false': [8, 180], 'cross_false': 2}.
OldVox0events/8.173672725694447h,upper950.3665099367309777/h.
Data coverage alone failed confusion gate. Original reports retained;no new/final
scored,no promotion,freezev30. Next work must diagnose actual training support/
confusion mechanism before another intervention;do not blind-tune mix/loss weights.

### v40 actual old-source training support diagnostic (2026-11-11)

All6597sampler-eligible old-source variants scored,not a mined subset.
Windowfloat positives4216/4602>=.5;all1995negatives<.5. Complete INT8
full-history five-mean after complete-context warmup:4224/4602positive,
115/1995negative>=.5. At historical.99609375,35negative false clips
(allPiper:Next22/Mix11/Nick2). Original rawSHA verified and all scores saved.
First reset exact;not event accuracy or independent acceptance. Offlinefloat
vsfullINT8 confounds phase/state/quantization;next same-weight floatfull-history
comparison required before attributing cause. No new/final or training changes.

### v40 same-checkpoint float history diagnostic (2026-11-11)

All6597 eligible variants evaluated with shared float sequence scorer; zero
optimizer updates and all window variables unchanged. Among1995negatives,
windowfloat0>=.5, full-sequence float103>=.5 versus INT8115; at.99609375,
float43 versus INT835. Positive counts at.5:window4216,float4192,INT84224
of4602. Thus high negative scores exist before quantization in the full-sequence
calculation; this does not establish a sole temporal or quantization cause.
Actual exported float streaming checked on20 first engine/task witnesses only;
maximum absolute residual0.0017456412315368652, retained in original report.
Do not claim exact equivalence or all6597 actual streaming replays. Next diagnostic
must locate high-score temporal support and resolve/measure streaming residuals
before a new training intervention. No new/final scored; v40 remains rejected,
frozenv30 retained. Coordination continues in GitHub Issue13.

### v40 actual float negative-stream localization (2026-11-11)

Replayed actual exported float streaming for all118training negatives in the
union of prior floatsequence/INT8 fullhistory maxfive>=.5. Actual float103>=.5,
43>=.99609375, matching aggregate sequence counts; max individual output
sequence/actual residual.008029580116271973, so no exact-equivalence claim.
For103high clips peak-five final feature end minus training crop end ranged
-59..69frames,median32:19earlier,1equal,83later. For43at historical threshold:
-17..69,median36:2earlier,1equal,40later (Next23/Mix17/Nick3).
Processing frame positions only, not phonetic boundaries or event latency.
This supports investigating missing temporal negative support, not proof of a
sole cause. Diagnostic selected union only, not all1995actual float streams or
independent acceptance. No optimizer updates, all window variables unchanged;
new/final untouched and frozenv30 retained. Next intervention must explicitly
address full-source temporal support rather than repeat center-only coverage.

### v41 negative complete-timeline intervention prechecks (2026-11-11)

Preregistered38207a3. Only24candidate negative windows change relativev40;
source/variant draws preserved,independent137726 uniform raw endpoints150..N.
All1995negative raw arrays hash/shape authenticated.50batch audit1200crops
matches independent source RNG/crops; retainedslots exact. v40default sampler
regression also passes50batches. Two-step4090smoke exit0,35.4s,selected1,
54312B SHA730ccac17567dc1941d745628f993097d2ac94da9dfea458a619307bd2f1c9c2.
Finite weights,frozenBN exact,int8[1,3,40] to uint8[1,1]. Not accuracy evidence.
Formal120k is authorized after these passed prechecks; same historical gate,
new/final untouched. No promotion; frozenv30 retained.

### Unseen development executor generalized while v41 trains (2026-11-11)

Original v36-only runner preserved. New experiments/unseen_development.py takes
explicit --candidate and --check-only. Authenticates model/training/export/probe,
old synthetic/background report hashes, fixed138point scope and eligible point
before any unseen data access. Fresh output guard prevents silent rescore;
frozenv30 hash pinned. Existing SLR83/newVox threshold/official comparison protocol
unchanged. Negative guard checks on v37/v38/v39/v40 all reject at old-stage gate.
No passing candidate yet; actual unseen scoring path not exercised. New/final
remain unscored. v41 original training/session58922 and waiter/session42935 live;
first5000step internal validation completed. No new training or threshold changes.

### Retrospective original training PCM mechanical audit (2026-11-11)

All2594canonical original training WAV/filePCM hashes reverified. MonoPCM16/16k,
nonempty/nonzero,.2..5s,RMS>=.001 pass2594/2594. Zero-fullscale check fails40:
allPiper,exactly1fullscale sample each (HiESP29,Nexus3,Nyx6,Nick1,Next1).
Eight sources are in v40/v41 task coverage. Fullscale code is not proof of
flattened waveform clipping; native float/quantization provenance needed.
No filtering,attenuation,relabeling,data replacement or training restart.
This is a failed retrospective mechanical check, not a claim that inherited
training sources meet the new SLR83 quality specification. v41 remains a
controlled diagnostic and cannot establish all quality gates from these data.
Original containers do not prove uncropped phrase/phonetic truth or complete
TTS provenance. Preserve failures and coordinate native-generation review via
Issue13. New/final PCM not accessed. All details and audit code retained.

### Old Piper PCM endpoint mechanism diagnostic (2026-11-11)

Pinned local upstream converter b6068e38ca986fc3c1cd55b631878aeb619ebe5a65b32fd9f8a870c7e8e694d8
scales peak to32767 then casts. Isolated AST function on bounded float peak.4
produces a32767code without preexisting clipping; negative mirror produces-32767
and passes asymmetric endpoint check. All2039original Piper training WAV hashes
rechecked:abspeak32767in1667,32766in372;min-32767in1627,max32767in40.
This is consistent with peak normalization, not proof of its exact use on Mac
sources. None of40earlier failures matches local GPU generation manifest hashes.
Native float/provenance still missing for this claim; failed endpoint gate retained,
no retrospective relaxation or data changes. Do not call40files proven clipped
or the other2554proven distortion-free. Issue13 coordination continues.

### v41 training completed and authenticated (2026-11-11)

Original session58922 exit0,120000steps,2204.7s,selected25000 by unchanged
historical selection. Model54440B SHA
c53d37787a202da6de33c5452e4c53060864b9313ac30adcc5e0d602f95529bd.
Original training.json/export audit retained unchanged. Finite weights,frozenBN
exact,original int8[1,3,40] to uint8[1,1]. Original waiter session42935 authenticated
recipe/model and started historical regression. No accuracy/FAR/promotion claim.
Old-source PCM quality limitations remain; no unseen/final scored,freezev30.

### v41 rejected after complete historical regression (2026-11-11)

Original session42935 exit0;467score traces reauthenticated. Fixed138points,
zeroeligible,maxmacro.725,min13negative false clips. Historical.99609375:
Kokoro targetany9/16,commonfalse3/48,cross0;VCTK26/60,common10/180,cross0.
OldVox87fullrecordings/8.173672725694447h:1event,upper95.5803834674561834/h,
failing.5/h. Do not remove event or merge newhours. Complete-timeline negative
sampling alone failed joint gate;no new/final scored or promotion,freezev30.
All original reports retained. Quality source limitations remain. Next work must
measure whether the intervention reduced training-history false peaks before
another change, rather than repeat sampler/loss changes without evidence.

### v41 matched training-support diagnostic (2026-11-11)

Original session30464 exit0,all6597variants and prior saved-score hashes verified.
Matched1995negatives:fullINT8maxfive>=.5 v40115 ->v4129;at.99609375 35->0.
Original centerfloat negative>=.5 0->23;at.99609375 0->1. Positive4602 fullINT8
>=.5 4224->4254,at.99609375 3256->2423. Actual temporal sampling changes training
support,but does not fix old-development confusion/FAR. Selected checkpoints differ,
no isolated causal/generalization proof. All scores/reports preserved. Next same
v41weight fullfloat comparison required before quantization attribution;no new/final,
no promotion,freezev30. Avoid repeating temporal sampling alone as solved approach.

### v41 same-weight float full-history training diagnostic (2026-11-11)

Original session9393 exit0,all6597variants.1995negatives fullsequencefloat26>=.5
versusINT829;both0at.99609375.4602positives float4214>=.5,2528at highthreshold,
versusINT84254/2423. Highthreshold positive-support reduction also exists before
quantization.20actual floatstream witnesses,maxsequence residual.00702512264251709;
no exact/all6597actualstream claim. All variables unchanged,optimizer0.
Next actual exported float replay on all380already-consumed olddev clips started,
fixed.99609375,INT8fulltrace exact-reproduction required. Diagnostic only,no
new/final,FARrescue,thresholdselection or promotion;frozenv30 retained.

### v41 actual float old-development replay (2026-11-11)

Original session43081 exit0;all380olddev fullINT8smoothed traces reproduce exactly.
Actual exportedfloat streams at fixed.99609375:Kokoro targetany9/16,negative3/64;
VCTK26/60,negative10/240,identical INT8 counts. Clip-level event-present decisions
agree380/380;8clips have differing event indices,so no exact numerical/timing parity.
Thus these13confusion clips already fail in float;quantization is not necessary
for these observed failures. Not proof of sole cause or universalfloat/INT8parity.
No threshold selection,newFAR,new/final,training or promotion. Frozenv30 retained.
Next examine positive full-source temporal support alongside the proven negative
support change before another intervention;do not repeat negative-only sampling
or local five-score supplement as solved approaches.

### Positive complete-container temporal support census (2026-11-11)

All4602eligible training positives per v40/v41,all savedfloatscore hashes checked.
Conservative original Mac sample ledger yields13..100valid integer windowends,
103663phase-aligned five positions;2variants have no valid five due to phase.
Initial code incorrectly asserted every variant had one;casepackindex2/end338..350
exposed the phase gap(firststreamend339,last351). Corrected census retains2zero
cases,no source excluded. Under this ledger fullsupport highthreshold variants
v402468->v411374;support positions24917->7844. At.5 variants3524->3306.
Global peakinside support1017->409. These are savedsequence diagnostics with
known streaming residuals,not phonetic/receptive-field truth or acceptance.
Pre-register v42positive candidate full-interval sampling,keeping v41negative RNG
exact;no optimization started. Oldquality limitations retained,freezev30.

### v42 integration and prechecks completed (2026-11-11)

Preregistered bdb6eb1. Independent147726positive-end sampler uses complete Mac
container ledger;source and v41negative RNG exact. All4602positive rawSHA/shape/
originalcrop authenticated.50batch independent800positive/1200negative crops,
retainedslots and v41negative RNG/crops exact. Two-step4090smoke exit0,42.0s,
54312B SHA1f204ab8b8bb48ce9e9843545d4dc6e375376c9d7734671c7f3be3ec9927bda1.
Finite weights,frozenBN exact,originalstreamINT8contract valid. Not accuracy/FAR.
Formal120k authorized after these passed prechecks;unchanged historical gate.
Quality provenance/endpoint failures retained;research diagnostic,no promotion,
new/final untouched,frozenv30 retained.

### Mac positive interval cross-check received (2026-11-11)

Issue13 Mac52607a4 independently enumerated every endpoint for4602positives.
Original source6f60547efb8a23e9af810e698835818430cac0f925f38e01ad821338eaea4857
reexecuted unchanged on Linux metadata;report byte-exact SHA
887598121cb18123199f4ad99b1afc88ff1d4a80fb75f7c43fb2a7baea9bc733.
All intervals equal proposedcrop_end..350;103663supported five positions;
packindices2/371have zero,retained. No rawfeatures/scores/model inference.
Agreement limited to sample-container/phase arithmetic,not physicalRF or phonetics.
No v42training change;original session85259 and waiter8208 continue.

### 2026-11-12 UTC — v42 formal training and export completed

Original 4090 run completed 120000 steps in 2197.7 s; unchanged historical checkpoint rule selected step 15000. Original export is 54440 bytes, SHA256 `ff15ebf95001433135514babe0929bb75afa79d0e2a9755e0d61d9a6c7d81272`. Training metadata and export audit copied unchanged to results. All saved weights finite; frozen BN unchanged; int8 [1,3,40] → uint8 [1,1] export contract passed. This is not accuracy/FAR/P4 acceptance. Existing postprocessor authenticated recipe and started the preregistered historical regressions. No promotion, untouched development/final access, or change to frozen v30.

### 2026-11-12 UTC — v42 rejected at fixed old-development gate

Complete original postprocessor finished. Fixed 138 thresholds: zero eligible. Summary: {"candidate": "v42", "status": "rejected", "grid_points": 138, "eligible_points": 0, "max_macro": 0.7333333333333334, "minimum_negative_false_clips": 20, "frozen_v30_retained": true, "untouched_development_used": false, "final_used": false, "limits": "Historical reused development rejection; not independent acceptance. Score-trace hashes require separate receipt."}. At historical threshold 0.992: Kokoro target-any 10/16, common-negative false clips 6/48, cross 0/16; VCTK target-any 38/60, common false 15/180, cross 0/60. Old Vox 8.173672725694447 h, zero events, upper95 0.3665099367309777/h. Background pass does not override synthetic failures. Original reports copied unchanged. No new-development/final access; v30 retained.

### 2026-11-12 UTC — v42 saved-score authentication and unseen guard

Authenticated all 467 saved old-development NPY files (80 Kokoro, 300 VCTK, 87 Vox) against their original report SHA256 values; all match. Reproduction script and detailed receipt saved. `unseen_development.py --candidate v42 --check-only` rejected with the expected no-eligible-point error before unseen data access. The rejection remains authoritative; no candidate promotion or new training started.

### 2026-11-12 UTC — v42 training support diagnostic started

After old-development rejection, replay all 6597 eligible old-training variants using the selected v42 checkpoint, matching the prior v41 support probe. Retain every eligible variant and original raw feature hashes; compare center-window float and complete-history INT8 maxima/five means. Fixed reference threshold 0.99609375 is inherited solely for v41 comparison, not v42's historical 0.992 and not new threshold selection. Differences confound checkpoint selection and sampling; they cannot alone prove causality. No optimizer updates, new/final data, filtering, promotion or loss change. Original run log: `.local/wakeword/v42-training-support.log`.

V42 support diagnostic completed exit0 after fixing a summary-key rename KeyError (initial full replay retained separately). All6597 variants scored. {"0": {"count": 1995, "full_five_half": 12, "full_five_reference": 0}, "1": {"count": 4602, "full_five_half": 4391, "full_five_reference": 3852}}. V41 negatives at the same fixed references were 29 at .5 and 0 at .99609375; v42 has 12 and 0. This training-source improvement does not remove old-development false activations; no causal or generalization claim.

### 2026-11-12 UTC — v42 actual exported float old-development diagnostic

Run all380 already-consumed Kokoro/VCTK clips through the actual v42 SavedModel streaming signature, resetting states per clip, and reauthenticate/reproduce every saved INT8 trace exactly. Hold v42 historical threshold0.992 and detector policy unchanged. This tests whether observed v42 errors also occur before INT8 quantization; it does not measure new FAR or establish a sole cause. Original manifest/WAV/PCM hashes required. No optimizer, threshold selection, new/final data or promotion. Log `.local/wakeword/v42-old-development-float-probe.log`.

V42 actual-float diagnostic finished exit0. All380 INT8 traces reproduced exactly. At fixed0.992, float vs INT8 target-any: Kokoro9/16 vs10/16, VCTK39/60 vs38/60. Both modes have6/64 Kokoro and15/240 VCTK negative false clips, but individual negative decisions differ2/304:20 failures shared, one float-only and one INT8-only. All380 event-present decisions differ4. Initial attempted assertion of equal negative decisions failed before report publication; equal totals do not prove equal decisions. Quantization is not necessary for the20 shared failures; no sole-cause, timing parity or acceptance claim.

### 2026-11-12 UTC — matched v40/v41/v42 INT8 temporal census

Authenticated all three original score archives/reports and each per-variant score hash; grouped by engine/phrase/original conditioning-speaker list. All4602 positives retain103663 conservative complete-container five positions, including2 zero-support variants. At common0.99609375, support-hit variants v40/v41/v42=2399/1304/3835; positions24266/7549/73022. These are INT8 values, distinct from earlier shared-float census. V42 .992 original-history threshold:3961 positive variants with support hit,77600 hit positions,1 full-history negative above threshold but0 center-window float negatives. At.5,12 full-history negative variants remain. V42 improved container-supported positive coverage but did not solve old-development confusion. Neither fully separable training nor a sole generalization cause is proven; checkpoint choices differ. No data/loss/threshold changes or new/final access.

### 2026-11-12 UTC — v42 residual training negative window witnesses

Scored every legal150-frame window plus actual exported float stream for all12 training negatives selected by INT8 max-five>=.5. Original raw hashes verified. Seven of12 remain>=.5 in actual float; none>=.992. Pack812 (macOS hey nick variant2), the sole INT8 historical-threshold training failure, has actual float max-five0.9915394425392152 vsINT8 0.9953125, window float max0.9948824644088745. Only3/256 legal windows exceed.992 (35/256 exceed.5); this is uniform-draw probability, not realized optimization exposure. Other witnesses retained, including windows with zero>=.5. Window/stream phases and contexts differ, no exact equivalence claimed. Sparse window errors and quantization-boundary effects coexist in this training witness; this does not explain the20 shared float/INT8 old-development failures or establish a new intervention yet. No optimizer/data/threshold/promotion change.

### 2026-11-12 UTC — v42 deterministic negative-window exposure replay

Preserved v1 witness report; v2 adds exact high-window endpoints. Replayed original CompleteCoverage.draw_indices seed127726 and independent negative-end seed137726 for15000 batches, matching the selected checkpoint. Positive-end RNG is independent; no pre-loop batch calls exist, representative batches occur after training. Pack812 selected306 times across178 distinct ends;41 draws in final-weight>=.5 window set,4 in final-weight>=.992 set. Therefore never-drawn high windows is not supported for this witness. Other eleven witnesses retain complete end counts. Final-weight high sets do not establish scores/gradients at historical draw times; no causal optimization conclusion. No optimizer updates, model/data changes, threshold selection or new/final access. Both diagnostic runs exit0.

### 2026-11-12 UTC — Mac negative-mass audit receipt and v43 preregistration

Original bb5854a audit script run unchanged against immutable metadata; Linux JSON byte-identical to Mac, SHA cb8e12038c897bcf75af5a007552ab2eca72d5e57743834c9c42f53fd736c2f2. Complete-container probability .2483580266015314, expected5.960592638436753/24 slots. This motivates a falsifiable distribution experiment, not causal proof: preregister v43 with12 complete-container negative endpoints plus12 retained full-timeline endpoints, same source mass and all other v42 settings. Original timeline RNG remains consumed identically; independent seed157726 for changed slots. Implementation/sampler checks/smoke required before any formal run. Quality provenance remains unresolved; v30/newdev/finals unchanged.

### 2026-11-12 UTC — v43 integration prechecks passed

All1995eligible negative raw arrays authenticated; every conservative legal endpoint set enumerated independently. 50batches exact independent crop reconstruction, unchanged source RNG, all v42 positive endpoints and retained12 negative endpoints, original full negative RNG state, unaffected slots. 4090two-step smoke exit0 in43.6s; step1selected;54312B SHA21b1bb60bc1a545f3bd1b39228124dc1e3dee69c3f82dc24191f4fafa99f8b6b. All saved weights finite, frozenBN unchanged, int8[1,3,40]→uint8[1,1]. Formal launch and authenticated postprocessor prepared; no promotion or new/final access.

### 2026-11-12 UTC — original Mac Piper provenance cross-check

Imported original e56073a metadata report unchanged (SHAc4ff78084ad508451bedd9a86c354d3563e3626a39512116a28ba5fa5aa6ebd2), verified referenced Linux quality/mechanism report hashes and original Mac audit script identity. All40 failure SHA sets, training split, label and sample counts match prior Linux verified WAV/PCM audit; converter function text exactly matches local mechanism report. Mac reports2039/2039 original manifest matches; this receipt does not independently possess/replay the full original manifest. Original Mac dataset directory has no native-float artifacts; historical package/source/config integrity remains unproven, current lock cannot substitute. Original40 PCM endpoint check failures remain, no posthoc repair/filtering or gate relaxation. Source correspondence improved, historical synthesis quality not established. V43 original training continues unchanged.

### 2026-11-12 UTC — v43 formal training/export complete

Original one120k run finished in2183.5s; unchanged checkpoint rule selected step10000.54440B export SHAfe8e46206069e841eb8171beaca727b147602814e4130394ffd16fc8c8fff2ba. Original training/export audit copied unchanged; finite saved weights/frozenBN/int8[1,3,40]→uint8[1,1] contract passed. Original postprocessor authenticated mixture seed/slots and all recipe invariants, started historical regressions. No accuracy/FAR/HiESP/P4 acceptance claim or promotion; v30 and untouched development/finals preserved.

### 2026-11-12 UTC — v43 complete rejection and trace receipt

Original postprocessor completed. {"candidate": "v43", "status": "rejected", "grid_points": 138, "eligible_points": 0, "max_macro": 0.8854166666666667, "minimum_negative_false_clips": 22, "historical": {"phone": {"threshold": 0.99609375, "nyx": {"positive_clips": 16, "hits": 9, "recall": 0.5625, "premature_events": 3, "late_events": 0, "latency_ms_median": 9.999999999999787, "latency_ms_p95": 169.0, "common_negative": {"clips": 48, "false_events": 3, "false_activation_clips": 3}, "cross-target_negative": {"clips": 16, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 12}}, "vctk": {"threshold": 0.99609375, "nyx": {"positive_clips": 60, "hits": 29, "recall": 0.48333333333333334, "premature_events": 15, "late_events": 0, "latency_ms_median": -44.68749999999977, "latency_ms_p95": 129.53749999999994, "common_negative": {"clips": 180, "false_events": 19, "false_activation_clips": 19}, "cross-target_negative": {"clips": 60, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 44}}, "vox": {"source_hours": 8.173672725694447, "false_events": 1, "false_per_hour_upper_95": 0.5803834674561834, "threshold": 0.99609375}}, "authenticated_score_traces": 467, "frozen_v30_retained": true, "untouched_development_used": false, "final_used": false, "limits": "Reused historical development rejection, not independent acceptance or causal isolation; checkpoints differ."}. All467 saved score file SHA values match original reports. Original reports copied unchanged. Negative-container mixture did not yield an eligible old-stage point; no automatic further sampling change or promotion. Original synthesis-quality gaps persist; new development/finals untouched.

`unseen_development.py --candidate v43 --check-only` also rejected at no-eligible-point guard before unseen access, as required.

### 2026-11-12 UTC — v43 all-training support and matched census

All6597 eligible old-training variants replayed exit0. Full INT8 max-five negative variants>=.5: v42=12,v43=8; at.992:v42=1,v43=0; at common.99609375 both0. V43 center float negatives at.5=6,at.992/common threshold=0. Positive container-support hits atcommon.99609375:v42=3835,v43=3730; positions73022→70868 of103663,2zero-support variants retained. Atv43 historical.99609375,0/1995 training negative high peaks coexist with22minimum old-development false clips andfailedVoxFAR. Stronger training separation did not solve development confusion; checkpoint selection differs, so no isolated causal attribution. Source/variant/identity groups and trace hashes retained in v3 census. Copied v43 probe's inherited scope sentence incorrectly denied coincident historical threshold; corrected descriptive sentence+script digest only before publication, original pre-correction local report preserved, all numerical arrays unchanged. No optimizer/new/final access or promotion; future work should address coverage/generalization and unresolved synthesis-quality provenance rather than assume another temporal sampler alone suffices.

### 2026-11-12 UTC — provenance-complete US training pilot audited

Preregistered 3 presets ×3 speeds ×4 texts ×3 draws produced108/108 mechanical passes, all native/resampled float and PCM retained. Independent read-only audit verified every source/output/PCM hash, exact grid/order/training split, config phonemes/tokens/scales, exact resample arrays, full lengths and quality gates. PCM conversion replay is exact using pinned libsndfile; an initial ideal floor(float64(x)*32768) assertion differed by one LSB at one encountered sample and was rejected, not used to alter audio. This replay shares conversion library and is not independent rounding-implementation proof. Original report/run/manifest copied byte-for-byte; no wake-model scoring, new-development/final access, training or promotion. This does not repair inherited corpus quality or establish phonetic truth/speaker independence. Issue13 Mac request for v43 actual-float/INT8 failure identity correspondence received; complete that consumed-data diagnostic before deciding any new fitting intervention. V30 remains frozen.

### 2026-11-12 UTC — v43 actual-float old-development correspondence

Actual exported float streaming replay completed for all380 already-consumed old-development clips on4090; all380 INT8 traces exactly reproduce authenticated historical scores. Historical threshold.99609375/smoothing5 unchanged. Of304negative clips:281neither,21both,1float-only,1INT8-only (23union failures). Kokoro float/INT8 negatives4/3;VCTK18/19. Target-any float/INT8 both12/16 Kokoro,44/60 VCTK. Shared failures: PiperMix12,Nick3,Next3;KokoroMix2,Nick1. Two one-sided decisions mean quantization affects specific clips;21shared failures show quantization is not necessary for those failures, not a sole-cause attribution.

Correspondence script authenticated all6597 training histories and source/record hashes; retained per-engine/phrase/original-conditioning-identity counts plus every development failure identity. Training fullINT8 at historical threshold has0/1995negative high peaks and centerfloat0, while held-out-identity development float still fails22negative clips. This supports testing coverage/generalization rather than another temporal-only intervention. Engine/phrase correspondence is aggregate, not matched utterances; no exact mapping of Piper original speaker arrays to VCTK identifiers or physical-person independence asserted. No threshold/model selection, optimization, new-development/final access or promotion. V30 retained; new108-source pilot remains training-only and has not been scored.

### 2026-11-12 UTC — original108 pilot Mac archive and324 training augmentations

Mac Issue13 requested original float/PCM/source bytes for independent audit. Added explicit float32 dtype checks to independent pilot audit (original run/manifest/report/audio unchanged). Packaged all108 native/resampled arrays and WAVs, original run/manifest/report plus all nine pinned source files/source manifest and audit. Release gpu-us-training-coverage-v1.tar.gz:188022124B SHAb257f7c10b7eac4558b6c414f6ecc20fea7186f3b42ee04b2e637511471889b6. Anonymous curl with cache-busting URL succeeded; all340archive members checked,339payload sizes/SHA exact against manifest (SHAe0cac3c8bf4a1a89829b1ca91ecdafa7073303ec18925d4681f3609cf5e4576b). Canonical URL initially cached404, recorded rather than claiming it worked then.

Before rendering, db14f3e froze US training feature plan:108×3 variants, complete waveform plus25dB noise/room15dB, fixed level/gain/seeds, serializedPCM frontend, all original and augmented wave/raw artifacts retained. Completed324/324 mechanical passes and full-container crops,0unsupported; training.npz SHA4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da. This is feature preparation, no wake-model scoring/fitting or promotion. Original inherited quality gaps remain.

Independent read-only324-attempt augmentation replay completed exit0: every full float64 array, pinned-library PCM16 roundtrip, regenerated frontend raw uint16 array and150-frame crop/label exactly matches saved bytes/arrays. All source/output/PCM hashes and full lengths/ledger gates pass; shared libraries explicitly limit implementation independence. No wake model inference.

### 2026-11-12 UTC — v44 US coverage preregistration and integration

0c4c915 fixed replacement of8positive/12negative slots with independently seeded US coverage (167726), all v43 draws first, other slots unchanged. Fifty independently replayed batches/1000draws match source/variant labels/raw full crops exactly; untouched60phrase slots and original British/source/positive/negative/container RNG states exact. Two-step4090smoke exit0 in39.5s,selected1,54312B SHA2352aa9d9048eb451e5237acc849c43eaa9d09997ad488ad6fec8ac0f1764902. Finite saved weights/frozenBN/int8[1,3,40]→uint8[1,1] export contract passes. Original historical postprocessor prepared with added US seed/pack/slot checks. New mixture is coverage experiment, no promise of acceptance; inherited quality gaps remain, v30/newdev/finals preserved. Formal one120k run authorized after these checks.

### 2026-11-12 UTC — US training exact-isolation audit during unchanged v44 run

Pinned metadata confirms108unique original WAV/PCM hashes,0exact matches against2594inherited canonical source WAV/PCM hashes;0source-file/rendered-PCM/literal-preset matches against380consumed old-development metadata rows. Three new preset labels have no literal old-training identity overlap. Published reserved SLR83 source metadata only has0exactONNX matches with three US models. No reserved generation/replay manifest, waveform, feature or score accessed. This is exact-byte/preset identity evidence only, not physical-person/global-pretraining/perceptual independence; original clean development PCM unavailable in those compared fields, so no such hash claim. Ryan rdh ancestry remains documented.

Same v44 PID217915/start3767651 remains live, no restart or recipe changes. First5000step float checkpoint hasmax-sourcefalse.0225, above original.01criterion; no premature acceptance/rejection or adaptation. Continue fixed120k and existing postprocessor.

### 2026-11-12 UTC — Mac strict resampling discrepancy preserved and replayed

Received original Mac7936daa report SHA12ff0b6b3cac3cf70e6743f9e16719fc3f741d82e086278eb744aec9f68cc07f and predeclared054d7e0 audit/source/plan. Original archive340members/339payloads authenticated byMac, all108 other mechanical gates pass and original saved-float→PCM conversion has0sample differences. MacARM64 native→16k strict bit-identity0/108:194115changed samples,max8.940696716308594e-8=0.0029296875PCM LSB. Mac all_process_checks_pass:false retained without tolerance relaxation.

Ran exact original Mac audit script SHA8f6232deeb1ff97fc8f880671e646547d645020a9acc6a378143d809ffc86afd under same frozen plan against original Linux artifacts:108/108strict pass,0resample/PCM differences. Report inputs/package versions exactly equal; host differences not causally isolated. Fixed-host exact replay established, cross-platform bit identity not established. Copied original Mac report/receipt and Linux replay unchanged; receipt independently recomputes108-row discrepancy summary. No audio rewriting/regeneration, gate changes, model inference, training restart or reserved data access. Original v44 PID217915/start3767651 continues using certified original Linux floats/PCM.

### 2026-11-12 UTC — original v44 formal training/export completed

Original PID217915/start3767651 exited after one120k run,2298.0s,unchanged checkpoint rule selected35000. Export54440B SHA88de16e71b5f49b5e85d226b74a07aa036324dcaab48e6f58c9c8a9343608652. Original training/export audit copied unchanged; all saved weights finite,frozenBN unchanged,int8[1,3,40]→uint8[1,1] contract passed. Original postprocessorPID218483 authenticated US167726/pack/slots and original v43 recipe seeds/BN before starting fixed historical regression. No accuracy/FAR/HiESP acceptance or promotion claim; v30/newdevelopment/finals preserved.

### 2026-11-12 UTC — v44 complete old-stage rejection

{"candidate": "v44", "status": "rejected", "grid_points": 138, "eligible_points": 0, "max_macro": 0.6895833333333333, "minimum_negative_false_clips": 11, "historical": {"phone": {"positive_clips": 16, "hits": 4, "recall": 0.25, "premature_events": 0, "late_events": 0, "latency_ms_median": -19.999999999999794, "latency_ms_p95": 8.499999999999883, "common_negative": {"clips": 48, "false_events": 2, "false_activation_clips": 2}, "cross-target_negative": {"clips": 16, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 4}, "vctk": {"positive_clips": 60, "hits": 26, "recall": 0.43333333333333335, "premature_events": 8, "late_events": 0, "latency_ms_median": 11.968750000000083, "latency_ms_p95": 105.07812500000014, "common_negative": {"clips": 180, "false_events": 9, "false_activation_clips": 9}, "cross-target_negative": {"clips": 60, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 34}, "vox": {"threshold": 0.99609375, "source_hours": 8.173672725694447, "false_events": 0, "false_per_hour_upper_95": 0.3665099367309777}}, "authenticated_score_traces": 467, "frozen_v30_retained": true, "untouched_development_used": false, "final_used": false, "limits": "Reused old development rejection only. Different selected checkpoints confound causal comparisons; no promotion or gate relaxation."}. Original postprocessor exited0. All467saved score files authenticate against original reports. Original reports copied unchanged; unseen --candidate v44 --check-only rejected at old-stage guard before unseen access. US coverage mixture did not yield an eligible fixed old-stage point. No automatic follow-up training or temporal/data-mass escalation; inherited quality limits persist. Next evaluate training support for both old and new training cohorts before choosing another intervention; v30 retained.

### 2026-11-12 UTC — v44 full training cohorts and actual-float development diagnosed

All6597inherited training variants authenticated and replayed. Historical.99609375 negative centerfloat/fullINT8 highs0/1995; at.5 fullINT8high9. Old positive conservative support: v43→v44 hitvariants3730→3263, positions70868→59833/103663;2zero-support variants retained. Piper positives account460of467net lost hitvariants (3083→2623); checkpoints differ, no isolated source-mass causality.

All324US training variants retained,648actual exported float/INT8 trace arrays authenticated. Exact supervised crop float at.99609375:negative0/243high,positive77/81high. Full-stream negative highs float1/243,INT8 4/243 (allMix, identities retained); positive full-container supporthitvariants float72/81,INT8 69/81; anyevents78/77. Thus exact crop fit does not guarantee full-stream rejection; do not label stream mismatch solely as failure to fit supervised negatives. Center/window vs streaming confounds phase/state; quantization adds differences. No optimization or source selection.

All380consumed old-development actual exportedfloat/INT8 replay completed;380INT8 traces exactly reproduce original outputs. Of304negative:291neither,11both,2float-only,0INT8-only. Float-only VCTK p248-1.1-Nick .9994240165vs.9953125,p361-.9-Mix .9995241046vs.984375. Kokoro negative2/2,VCTK11/9;bothmodes targetany4/16and34/60. All11INT8failures also actualfloat failures, so quantization not necessary for those11; cannot claim sole cause. Frozen model/report identities preserved.

Next narrow hypothesis to investigate before any new fitting: newUS examples are supervised only at one fixed complete crop, unlike inherited full-container/timeline coverage. Enumerate legal150-frame window float witnesses for all four US negative stream-failure variants and compare exact trained crop to actualfloat/INT8 stream, retaining all endpoints/failures. This diagnostic must precede deciding whether targeted new-cohort temporal coverage is justified; do not automatically increaseUSmass or launchv45. Newdev/finals untouched,v30retained.

### 2026-11-12 UTC — newUS fixed-crop holes witnessed; v45 time-coverage preregistration

Alllegal150frame float windows for four frozenv44newUStraining stream witnesses evaluated and arrays retained. Indices28/65/210/211:220/232/206/206windows,complete81/69/95/95. Exacttrained float scores2.736e-6/8.071e-8/1.733e-7/1.669e-7;window peaks.998833/.999831/.999879/.999984. Historical highendcounts6/15/8/8,completehigh6/13/8/8;onevariant has2noncomplete highends. This establishes fixed-crop time-position gaps, not equivalence of window/stateful inference or sole cause of olddev failure. Allwindow arrays/SHA/counts authenticated; no optimization/newdata.

Preregisterv45 changes only existingUS20slots time sampling:8positivecomplete,6negativecomplete+6negativefulltimeline,independent177726;source167726/alloriginaldraws/sourceprobabilities/other108slots unchanged. All324eligible, no witness mining/reweighting or source increase. Everyraw/endset audit,50batchindependent replay and4090smoke required beforeone120k. Originalgates,qualitylimits,v30/newdev/finals retained. No fitting started yet.

### 2026-11-12 UTC — v45 all-endpoint and sampler audit,4090 smoke passed

Implemented9f16671 US-only temporal plan. All324raw/file/data/shape/lowcrop/endsets independently exact. Fiftybatches/1000USsource+temporal draws match independent enumerations/seeds167726/177726;originalBritish/complete/source/positive/negative/container RNG states and other60phrase slots unchanged. Original48ambient generation code unmodified. Two-step4090smoke exit0 in43.7s,selected1,54312B SHAe206977a71777335871cf1a9619e8f88656c85626bbc622725cfd896a6cc7f37. All saved weights finite,frozenBN unchanged,int8[1,3,40]→uint8[1,1] contract passes. Original training/audit reports copied unchanged;formal120k plus original historical-only postprocessor prepared with177726/slots validation. No accuracy/FAR/promotion claim,oldqualitylimits/v30/newdev/finals preserved.

### 2026-11-12 UTC — Mac v44 diagnostic metadata audit reproduced

Imported original Mac1066df5 report unchanged, verified audit source SHA, ran exact original script on pinned9f16671 git objects; parsed JSON identical (byte status inreceipt). Original380identity/event correspondences,13witnesses,6597old/324UScohort counts/groups and4window ledgers corroborated. This is metadata arithmetic/hash-link evidence only, not independent model/trace-array replay. No change to originalv45 PID235075/start4081425 or newdev/finals.

### 2026-11-12 UTC — original v45 training and export completed

Original PID235075/start4081425 completed120000steps in2464.8s, original rule selected35000. Model54440B SHAf5d175a0172d4fde70dce746fa358331c3fcdf590139c2e76113fe14c45e30af. Original training SHA77d09bedd336c41f49c9c4cb5cf949b1c248f2727c76b417ced49bcd134bc7f0; original training/export audit copied byte-for-byte. All saved weights finite, frozenBN unchanged, int8[1,3,40]→uint8[1,1] contract passes. Original postprocessor PID235635 authenticated US seeds167726/177726, fixed slots/pack and inherited recipe before starting original historical regression. This proves export contract only, not accuracy/FAR/acceptance; no promotion or newdev/final access, v30 retained.

### 2026-11-12 UTC — v45 complete old-stage rejection

{"candidate": "v45", "fixed_grid_points": 138, "eligible_points": 0, "max_macro": 0.86875, "min_negative_false_clips": 34, "authenticated_traces": 467, "new_development_used": false, "final_used": false, "v30_retained": true}. Original postprocessor finished; all467saved score files match original report hashes. Original reports copied unchanged. Historical.99609375/smoothing5:Kokoro target-any8/16,commonfalse5/48,cross0;VCTK41/60,commonfalse29/180,cross0. OldVox8.173672725694447h,1event,upper95 .5803834674561834/h fails. unseen --candidate v45 --check-only rejects at old-stage guard before unseen access. V45 newUS temporal coverage did not produce an eligible candidate; no threshold/grid/exposure rescue or automatic v46. Different selected checkpoints/time-mixture calibration confound causal comparisons. Preserve v30 and untouched development/finals. Next inspect newUS and inherited training support to test the preregistered hypothesis before choosing any further intervention.

### 2026-11-12 UTC — v45 newUS and inherited training support diagnosed

All324US training variants replayed using actual exported float/fullINT8;648saved trace arrays authenticated. All243negative variants have0full max-five highs even at.5 in both modes (v44historical highfloat1/INT8 4). Positive81variants at.99609375:80support-hit and80event clips in both modes;v44supportfloat72/INT8 69. This supports improved new-cohort training fit, not generalization or temporal-only causal isolation.

All6597old eligible training variants replayed and score archive/everyhistory hash authenticated in matchedv40-v45 comparison. Historical.99609375:1995negative centerfloat/fullINT8 highs0;at.5 fullINT8high9 unchanged fromv44. Positive full-container supportv44→v45 hitvariants3263→3695,positions59833→70550/103663;2zero-support variants retained. Old training support improved while old-development has34falseclips minimum andVoxFARfails; training fit does not establish held-out-identity separation. Different selected checkpoints and calibration mixtures confound causal claims. Next determine actualfloat/INT8 correspondence of consumed old-development failures before selecting a new data/architecture intervention. No training, threshold selection, newdev/final access or promotion;v30retained.

### 2026-11-12 UTC — v45 actualfloat development failure correspondence

All380consumed old-development clips replayed with actual exported float; everyINT8 trace exactly reproduces original saved output. Historical.99609375/smoothing5 fixed:Kokoro target-any8/16 andfalse5bothmodes;VCTK target-any40/60float,41/60INT8 andfalse26/29. Of304negative:270neither,31both,3INT8-only,0float-only. INT8-only p314-1.1-Next,p362-.9-Mix,p238-1.0-Next havefloatpeaks.9898619652/.9915636539/.9806310177,INT8all.99609375. All witnesses and originalidentity/source hashes retained; aggregateengine/phrase correspondence is not paired conditioningidentity proof.

Quantization affects three boundary decisions but is not necessary for31shared failures. Alongside zero historical training-negative highs and improvednewUSfit, this rules out treating more time coverage or quantization repair alone as established solution. No threshold selection or new/final access. Before next formal model, review prior failed interventions and original synthetic quality/source coverage, and define a materially distinct data/generalization hypothesis; no automaticv46. V30retained.

### 2026-11-12 UTC — balanced training-only pilot384 generated and audited

Executed23aa605 frozen first32original-training IDs×3speeds×4texts×1draw, sameID pairs/no inter-ID blending,torchseed187726 on4090. All384attempts retained,384/384mechanical passes,0failures,9.98s generation. Complete native/resampledfloat32,PCM16 and exposed per-token native-sample duration arrays saved; full duration sums exactly equal every native waveform length. No normalization/trimming/retry/score selection. Source model/config,actual Python helpers/local modifications,packages/run/order recorded. Local model identity pinned; no new remote-source byte-retrieval proof claimed.

Independent read-only384-row audit authenticated every file/PCM hash, token composition/grid/allowedIDs,all originalqualitychecks;native→resampled arrays and savedfloat→PCM exact under pinned sharedlibraries. Duration arrays and lengths exact; no acoustic phoneme truth or CUDA generation reproducibility claim. Originalrun SHA0019be7bcef881c904c0ad0a331d8781ad2554c5bbf746442b05a5efd62efeea,manifestSHA25b15c743f70eeb5e80a38173d4b6166224b6d199b6521943b3f519a8fc17119,originalreportSHA7cbdc622c22a1ad7e02a6829e259462a386435f3ab6948bfd95a01789306bd88. Metadata originals copied byte-for-byte toresults;waveforms staylocal. No wake scoring/feature preparation/model fitting/newdev/final access. Next freeze full672-ID balanced matrix and independent feature/audit requirements before any larger generation or training. Original inherited quality gaps persist;v30retained.

### 2026-11-12 UTC — full672-ID balanced synthesis started

cfd83b8 froze all672original-training conditioning IDs×3speeds×4texts×1draw=8064attempts,torchseed197726 and one-exampleCUDA order. Pilot384originals untouched/not reused. Complete native/resampledfloat/PCM/duration artifacts retained; all failures must be reported, no retries/normalization/trimming/score selection. Original process PID252043, session13921, confirmed live and2352/8064complete at66.0s. Full script/audit adapted from auditedpilot with exact original model/config/source checks and all672partition list authentication. No quality pass claim until originalreport plus independent audit finish; no features/wake scoring/training/newdev/final access or promotion. V30retained.

### 2026-11-12 UTC — full8064 balanced training synthesis and audit completed

Original PID252043/session13921 exited0 after201.625s;all8064attempts retained and8064mechanical passes,0failures. All672allowed training IDs have3speeds×4exactphrases, no pilot reuse, failed-row replacement or scoreselection. Complete native/resampledfloat32/PCM/duration arrays retained. Independent read-only8064-row audit exited0:allfile/PCM hashes,ID/grid/token composition,originalqualitychecks,duration arrays/sums/full lengths exact;native→16k and savedfloat→PCM arrays exact under sharedpinnedlibraries. No acoustic phonetic truth/personindependence/CUDA-bitreplay claim.

Originalrun SHAdfd4313ff36fa6099dd4660230062463a4ad1c7feda06fadf78c9c0aa89da652,manifestSHAe3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0,reportSHA611cbe534996f26025ff35bafc04078ce170fed14ee07dce657c57f540085c97. Metadata copied byte-for-byte toresults. Complete lengths.39475..1.0216875s;metadata-only existingcontext150 preflight with0/960RIRtail finds0unsupported among24192prospective variants. This is only lengthledger,not frontend/feature verification or a new dataset. No wake scoring/model fitting/newdev/final access. Next separately freeze full-source augmentation/featureplan before execution;v30retained,oldqualitygaps unchanged.

### 2026-11-12 UTC — balanced24192 training feature variants fully replayed

cbf0ca1 froze full8064source×3augmentation protocol before execution. Original process completed24192/24192mechanical passes,allincluded,0unsupportedcontexts. Each672conditioningID×4phrases has9variants (3speeds×3augmentations),no source/quality/score-based selection. Complete originalPCM,specifiednoise/RIRfulltail,explicitRMS.08 andheadroom.95 augmentation;saveallfloat64/WAV/rawuint16. Originalsources unchanged.

Independent read-only audit completed24192attempts:alloriginal/augmentedfilehashes,sourcegrid/identitymetadata/gains/seeds,lengths,wholeaugmentationfloat arrays,pinnedPCM conversion,serializedPCMfrontenduint16 arrays andcomplete150framecrop/label exact. Reportcounts allmatch,0unsupported. Sharednumerical/conversion/frontendlibraries limit implementationindependence;notphonetic truth or CUDAgenerationreplay. PackSHA37ea6ff5ffc8c85973afec3a63f231527e493ddeb6cee77c08d594e96bc15546;originalreportSHA93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3. Originalreport copiedbyte-for-byte, auditreceipt retained. No wake scoring/optimizer,newdev/finalaccess orpromotion. Next freeze balanced-cohort fitting slots/mass/seeds with independent sampler/export smoke;oldqualitygaps remain explicit,v30retained.

### 2026-11-12 UTC — v46 balanced replacement sampler and4090smoke passed

Implemented7c0fa8c,onlyremaininginheritedslots8..15/44..55 replacedafterallv45draws,balancedsource207726/end217726. All24192rawfile/data/shape/lowcrop/legalendsets authenticated. Independent50batch/1000source+end draws exactlymatch;untouched60phraseoutputs andoriginalBritish/complete/timeline/container/US RNGstates exact;48ambientcode unmodified. SourceIDs672eachphrase9variants,allretained. Two-step4090smoke exit0,39.6s,selected1,54312B SHAf87e7014d8e234a7293e545b625b15b1faf80fe1cf1a86262e290673d82123b3. Finiteweights/frozenBN/int8[1,3,40]→uint8[1,1] passed;originaltraining/exportaudit copiedunchanged. No accuracy/FAR/promotion claim. Originalhistoricalpostprocessor extendedonlytoauthenticatebalancedseed/slots/pack;one120k formalrun next. Newdev/finals/v30preserved;originalqualitylimitations unchanged.

### 2026-11-12 UTC — balanced corpus exact isolation during unchanged v46

Training-only metadata audit authenticated originalfullmanifest and all672IDs against originalseed726 trainingpartition. All8064native/WAV/PCM hashes unique,0cross-label exactduplicates;0exactfile/PCM matches to2594inheritedcanonical sources. Suppliedsourcefile/renderedPCM metadata for380consumed olddevelopment clips has0exactmatches. IntentionaloldtrainingLibriTTS identity ancestry persists;not672newphysicalpeople,globalpretrainingdisjointness,cleanoriginaldevPCM or perceptualuniqueness proof. Existingfullartifactaudit supplies fileintegrity;this is metadata-onlycomparison. No reservedgeneration/replaymanifest,waveform,features orscores accessed; no modelinference or originalv46 PID257013/start4501271 restart/recipechange. Originalwaiter257576/start4503383 continues.


## Original US feature handoff and comparison corrections (2026-11-12)

Mac pending Issue13 request fulfilled: original training.npz/report.json and all324 original raw NPY copied without regeneration. Release gpu-us-training-features-v1.tar.gz,2706061 bytes,SHA256 c8c5ecc906306f360423a9635545032f35fa6db4687259c16853dd289a2fa9ce. Anonymous curl redownload and all326 payload sizes/SHA verified;327 total members including manifest. URL: https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-us-training-features-v1.tar.gz

Correction: v44 and v45 both selected35000; different weights/training trajectories remain confounded, but selected-step difference is not an explanation for that pair. v46 replaces negative44..55 previously all fulltimeline with six complete+six fulltimeline, as well as replacing source cohort. It is a combined source/time-window intervention, not isolated source coverage causality. Original preregistered run continues unchanged.


## v46 original training/export completed (2026-11-12)

Original RTX4090 run completed120000 steps in2629.9s,selected10000 by original rule. Original model54440 bytes SHA256 5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417. Original training SHA bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858; training/export audit copied byte-for-byte into results. All saved weights finite,frozenBN unchanged,int8[1,3,40] to uint8[1,1] contract passed. Original waiter authenticated balanced207726/end217726/slots and inherited recipe then started historical regression. Export only,not acceptance; full historical/138-point gates pending,no new development/finals accessed,no promotion,v30 retained.


## v46 rejected at old-stage gates (2026-11-12)

Original postprocessor finished. Fixed138points:0eligible,maxmacro 0.7458333333333333,minimum negativefalseclips 4. Historical .99609375/smoothing5:Kokoro any7/16,commonfalse1/48,cross0;VCTK any32/60,commonfalse3/180,cross0. Vox8.173672725694447h,3events,upper95 .9486135288434813/h fails .5. All467scorefiles SHA authenticated. Original reports copied byte-for-byte;unseen check-only rejects at old gate before unseen data access. V30retained,no newdevelopment/finals used. Balanced source plus temporal mixture did not pass;different selected checkpoint and trajectory confound isolated causal claims. No automatic v47.


## v46 balanced training fit diagnostic started

After formal rejection,run full24192variant balanced training-only INT8 census with original frozen model and authenticated pack/raw bytes. Save every fullstream trace;fixed descriptive thresholds .5/.992/.99609375,positive container support and negative maxima/events,all conditioning IDs/phrases/variants included. No threshold selection,optimizer,heldout/final access or promotion. Script v46_balanced_training_support_probe.py adapted from existing US census,INT8 only;cannot separate float/quantization causes. PID271429/session83665 observed live. Results pending;no automatic next training.


## v46 complete balanced training fit census

All24192 original variants replayed,all saved score hashes and maxima/counts independently authenticated. Original23MBreport preserved byte-for-byte. At .99609375:18144negative variants8fullstream eventclips;6048positive variants5120any-event,5119container-support hits. At .5:118negativeeventclips,5916positiveany,5913support. No zero-support variants. Fit remains imperfect;not purely development-only failure. Started identical pack census with frozenv45 for matched descriptive comparison;v45 never trained on these fresh sources,so this compares prior coverage/generalization to v46 training fit,not independent evaluation or isolated causal architecture effect. No optimizer/newdev/final access/threshold selection/promotion.


## Matched balanced training source census v45/v46

All48384saved traces authenticated and maxima/fullhigh flags independently recomputed. At historical .99609375:negative eventclips151→8/18144;positive completecontainer support3797→5119/6048. At.5 negatives567→118;positive support4960→5913. This pack trainedv46 only;comparison is v45transfer versus v46fit,not independent heldout improvement or isolated causal effect. V46still rejected olddevelopment/FAR;no automatic training. Next diagnostic replays all380 already consumed development clips using actual v46 exported float and exact prior INT8 reproduction at fixed historical threshold,following existing v45 method,without optimization/newdev/finals.


## v46 actual float and warm-count audit; Mac receipt

All380 consumed development INT8 traces exactly reproduced. Actual exported float has identical clip decisions:Kokoro target7/16,negative1/64;VCTK target32/60,negative3/240. All304negatives:300neither,4both,0float-only,0INT8-only. Thus quantization is not necessary for these four residual failures;does not establish background float behavior. All380float arrays/maxima/events and48384training arrays audited. Fulltrace eventclip counts equal warmed-five-high/restarted-eventclip counts for bothmodels at allthree fixedthresholds;container support separately reported,not equated to events/phonetic truth.

Mac6814637400 requested missing original calibration report. Copied exact958ba828dac95143d924c2b620bf4d846d5f8a4cd6b0271c3f47bc62f242ff8b bytes to results/4090-v46-validation-v1.json,without regeneration. Mac reports US324-array/1000draw replay passed and matched dense/mean continuation rejected (6814357739/correction6814453665);these are Mac-reported evidence,not Linux independent replay. V46rejected,v30retained,no newdev/final or v47.


## Balanced rendering concentration and next diagnostic

All24192variants grouped by original speed/phrase/variant and all8064sources paired across3renders. V46 historical negativehighs8allMixvariant2;positive929supportmisses include653variant2. Same-source positive509fail onlyvariant2;1293sources succeedallthree. Variant2 combinesRIR/noise15 whilevariant1noise25,so cannot isolate RIR/noise. Four olddev residual failures includeNick/Next/Mix and remain inactualfloat;do not explain them by this concentration. Predeclared full8064source rendering factorial quiet/noise15/RIR/RIRnoise15 using frozenv46;exactoriginalquiet/combined reproduction required before scoring. No training,newdev/finals or promotion.


## Rendering factorial implementation and original-arm verification

Implemented phase-separated no-model pipeline. First all8064sources×quiet/combined reproduce float64/WAV/frontend and gains exactly;controls forbidden until full reference receipt. Source bytes/manifests pinned. Clarified before controls/scoring: speech normalized to .08 before noise,so original plan noise-RIR RMS wording incorrect. Noise-only uses exact normalized combined noise vector prefix,with length/realized-SNR difference disclosed;all intermediate arrays retained for newcontrols. Reference session97108 running;no new controls or scores yet.


## Factorial reference verification complete

Original8064source quiet+combined arms all16128 exact float64/WAV/frontend/gains reproduced,original bytes untouched. Original reference receipt copied byte-for-byte toresults,scriptc36a726bdaea5859f40ce0728689cfa7383678be8aece89c459e2f3d4a779175. Then launched controls phase session32274,all8064sources noise15-only and RIR-only;no wake model scoring until independent control array/PCM/frontend/ledger checks. No training/newdev/final or promotion.


## All rendering controls generated; independent audit running

All8064sources×noise-only/RIR-only16128controls generated with intermediates,original controls report copied byte-for-byte. Independent audit script checks everyfilehash/array/PCM/frontend andcompletecontainer,source/armgrid,seeds,gains andnoise realizedSNR. Original audit PID276154/session27312 observed live,7001/16128 verified. No model scores/optimizer/newdev/finals yet;do not infer quality pass from generator all_exact field,which in controls means only generation completion until independent audit finishes.


## All factorial controls audited; frozen v46 scoring started

Independent16128control array/file/PCM/frontend/sourcegrid/container replay all passed. Noise-only realizedSNR14.988397332051138..15.011287016865243dB;headroomgain1all. Audit report records allpass and originalcontrol/ref hashes;shared pinnedlibraries limitation retained. Started all8064source×4arm frozenv46 INT8 census(session output retained);quiet/combined use authenticated prior traces,noise/RIR controls newly inferred. Fixed.5/.992/.99609375,fulltrace/warmed/container metrics separately;no training/newdev/final/selection/promotion.


## All-source rendering factorial completed

All8064sources×4arms32256traces complete;every scorehash/event/support/group and8064sourcepairs independently recomputed,allmatch. Historical .99609375:positive completecontainer supportquiet1849,noise-only1382,RIR-only1846,combined1363 /2016;negative fulltrace/warmed events0,6,1,8 /6048. Combined positiveany1364vscontainer1363 explicitlydistinguished. Under thisfixedrecipe noise15 sensitivitylargerthan RIR-only aggregate;not global RIR/noise causal conclusion or explanation of unaugmenteddevelopment/backgrounderrors. Noise-onlyprefixSNR14.988..15.011,headroom1all;RIRnormalization/tail remaininterventions. Original33MBreport copiedbyte-for-byte,audit+pairedpatterns saved. No threshold selection/newdev/final/training/promotion;v46rejected,v30retained. Need separate justified training hypothesis,not removal of difficult noisy variants or scoring heldout to rescuev46.


## Full factorial frontend-range census and historical BN check

All32256original/new raw feature arrays authenticated;fulltrace and generatedsignal-frame-endpoint regions grouped byarm. No raw/25.6 values exceed26 inanyarm (quietmax658,noiseonlyfullmax649,RIRmax658,combinedmax649),so input upper clipping not explanation inthiscorpus. Signalregion zero fractionquiet.520432,noiseonly.688943,RIR.558987,combined.708591. Descriptive frontend representation change,not evidence of phonetic deletion or automatic causal attribution. All40bandmeans/counts retained. No alteredfrontend/modelinference/training/newdev/final.

Historicalv22 already tried fixedBNstatistics/trainableaffine and failed:40kselected10k,macro.725521vs v19 .816146,sameoldgrid. Do not blindly repeat affineBN asnewidea. This priorrecipe differs fromv46,so notuniversalproof, butmere noise sensitivitydoesnotjustifyrepeatingit. Mac50001d2 reports completed159source/24checkpoint/264floatgrid/467receipt/138point metadataaudit,originalcalibrationgapclosed;explicitlynotindependentinference.


## v47 factorized pack audit and sampler implementation

All40320rawlinks/crops/labels/metadata authenticated;original24192subsetexact,new16128controlsexact,672seed726trainingIDs exact,2688groups each15variants,all5arms8064. PackSHA1a6cbf491c624c58cf96cfdac882966a7863927f546ac553075248caacfaaf5f;reportSHA7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44. Originalreport copiedbyte-for-byte. Independent dataaudit saved. Implemented separate FactorizedCoverage and mutuallyexclusive trainer flag,oldBalancedCoverage code untouched;v47samplerprecheckPID279551/session33014 running. No GPUtrain/smoke yet. Formalrequiresprecheck+smoke+postprocessor/exacthashrecipepinning,oldgates/newdataisolationunchanged.


## v47 smoke passed and execution pinned

Independent40320pack audit and50batch/1000draw/legalendpoint precheck passed. RTX4090two-step smoke45.8s,selected1,54312bytes SHAbbca7175b1f7e928c0e992f63c095c3cf0f2601877662de6d6925259c8c9263c. Finiteweights/frozenBN/int8[1,3,40]→uint8[1,1] exportpassed;not performanceacceptance. Originalsmoketraining/audit copiedbyte-for-byte. Formal recipe/postprocessor/historical shell/trainer/helper exacthashes pinned before formal startup. Postprocessor verifies15groups,newpack/report,allinheritedseeds/slots,then oldregression/phone/VCTK/Vox/138points only,no automatic unseen/promotion. Ready for one120k run;v30retained.


## v47 original formal startup verified

OriginalformalPID281170/start4934229/session44843 alive;waiterPID281771/start4936506/session17410 waits on exacttrainingidentity. Initialrandomhead/frozenBN match,pinnedexecsources unchanged. Actualtraining1000/120000observed;no restart or recipechange. Issue6815011788 records formalstartup. Completeexport/historicalgates stillpending;no acceptance/newdev/final access.


## v47 formal training and export complete

Original run completed 120000 steps in 2692.3 seconds on RTX 4090; selected step 5000 under unchanged checkpoint selection. Original training and export audit copied byte-for-byte. Model 54440 bytes SHA256 c6c9037c50c66230a783788678328d12a73e011dec5c97b94f74deba5e58cdd9; training SHA256 8a6b35c3b4e619660a751133176532f7d8a8535d8df30dfdfec09bc936d6b06f. Saved weights finite, frozen BN unchanged, int8 [1,3,40] to uint8 [1,1] contract passed. Original waiter authenticated the factorized pack, 2688 groups of 15, and inherited recipe before starting historical regression. Historical gates still pending; export contract is not accuracy/FAR acceptance. V30 retained, no new development/final access or promotion.

Original v47 validation calibration also preserved byte-for-byte in 4090-v47-validation-v1.json: threshold 0.9765625, smoothing 5, macro recall 0.32447916666666665 on historical training validation; 0 events / 1.2048112673611107 hours gives upper95 2.4864743173555475/h. The report viable_operating_point flag uses its historical calibration rule and does not satisfy the full research acceptance gate. Original historical evaluation waiter remains live; no change to its input calibration.


## v47 rejected on complete historical gate

{"candidate": "v47", "status": "rejected", "fixed_grid_points": 138, "eligible_points": 0, "max_macro": 0.6666666666666667, "min_negative_false_clips": 3, "historical_threshold": 0.9765625, "historical": {"phone": {"positive_clips": 16, "hits": 4, "recall": 0.25, "premature_events": 2, "late_events": 0, "latency_ms_median": -4.999999999999671, "latency_ms_p95": 8.499999999999883, "common_negative": {"clips": 48, "false_events": 0, "false_activation_clips": 0}, "cross-target_negative": {"clips": 16, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 6}, "vctk": {"positive_clips": 60, "hits": 28, "recall": 0.4666666666666667, "premature_events": 13, "late_events": 0, "latency_ms_median": -18.031249999999943, "latency_ms_p95": 101.08125000000007, "common_negative": {"clips": 180, "false_events": 7, "false_activation_clips": 7}, "cross-target_negative": {"clips": 60, "false_events": 0, "false_activation_clips": 0}, "positive_any_event_clips": 41}, "vox": {"source_hours": 8.173672725694447, "false_events": 1, "false_per_hour_upper_95": 0.5803834674561834}}, "authenticated_traces": 467, "new_development_used": false, "final_used": false, "v30_retained": true, "unseen_check_only": "Rejected by no eligible old-stage point guard before unseen-data access", "limits": "Reused development rejection; no isolated causal conclusion, no gate relaxation or promotion."}

Original waiter finished historical regression, streaming probe, old phone/VCTK/Vox and fixed 138-point calibration. All 467 saved scores authenticate against original report hashes. Original reports preserved byte-for-byte, including original validation calibration already delivered in 5562095. unseen_development --candidate v47 --check-only rejected at old-stage guard before unseen data access. Expanded factorized coverage did not yield an eligible candidate. Different selected checkpoints (v46 10k, v47 5k) and trajectory/condition-mass changes prevent an isolated augmentation causal claim. No automatic v48, no threshold/exposure rescue; v30 and unseen/final boundaries retained.


## v47 actual float old-development diagnostic

Pinned d9d751b fixed all380 consumed PCM at original .9765625/smoothing5. Every saved INT8 trace reproduced exactly; actual exported float states reset per clip. All380 float arrays finite, saved archive hash authenticated. Negative paired decisions {"both": 7, "float_only": 0, "int8_only": 0, "neither": 297}. Float and INT8 both: phone any6/16, negative0/64; VCTK any41/60, negative7/240. The seven negative clip failures already exist in float, so quantization is not necessary for these historical-threshold failures. No background float/FAR, embedding mechanism or universal quantization claim; no new data, threshold search or training. Original report preserved byte-for-byte and paired negative witnesses retained.


## Frozen v46/v47 factorized embedding diagnostic

Preregistered 112b942, implementation149e3e0. All40320 raw arrays authenticated, all8064 sources×fivearms×two common complete endpoints,80640 windows/model. Selected original v46/v47 Flatten vectors300dim extracted with frozen original heads, all161280 head outputs reconstructed (v47 max error2.98e-7), allfinite/zero norms0. Saved arrays SHA e82b16f41d191a0b53555cba59bae7cddf33232906416c22ed8a8139e45cb821. Arithmetic audit independently recomputed 420 quantile sets and full original logits/norms/cosines/paired margins. Originalreport byte-preserved.

At earliest common endpoint, target median logit v46 quiet9.548/noise15 5.439/RIR9.437; v47 quiet7.832/noise15 3.673/RIR7.571. At latest common endpoint v47 quiet4.889/noise15 .189, target-vs-Mix median logit margin14.922→9.432 and cosine .427→.715; v46 target logit5.980→2.049, margin20.258→14.393, cosine .210→.430. Quantiles/full arrays include all phrases and arms; these medians do not establish tail failure rates or streaming events. All sources are reused training, fixed endpoints not exhaustive timelines, flattened temporal content and different selected5k/10k checkpoints confound mechanism/causal comparisons. This supports investigating noise/window-placement sensitivity, not a claim that a specific contrastive objective will solve it. No newdevelopment/final reads, threshold search, retraining or automaticv48. V30retained.


## Original v46 model handoff for Mac independent inference

Mac Issue6815817541 requested original v46 artifacts before any loss/training decision. Original H5 SHA16b01e2f17473121c635801e324f5cf550c2ea0b72f8a054c04d845a0f596027 and rejected TFLite preserved, together with original training/initialization/export audit and existing SavedModel. Existing upstream checkout clean at4665173cd35f1cff9a61e06fc427f124766c488e. No retraining/reexport. Archive https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-v46-model-handoff-v1.tar.gz size537044 SHA4f289611e3ae763558c00e9cdaed5a18b1ad78982b19a3f67f4efcb8dff31812. Anonymous curl redownload verified archive, internal manifest and all10 payload sizes/SHA. Separate40320raw package under construction; no NPZ required. This is diagnostic handoff, not promotion.


## Original factorized raw handoff completed

Independent raw archive contains all40320 original raw-NPY and three original reports (factorized report, original balanced report, controls report); original relative ../ references preserved. Allraw file SHA/array SHA/dtype/shape authenticated before packaging, no NPZ or regeneration. Archive https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-factorized-raw-features-v1.tar.gz size266551321 SHA25e2832127b2388b7a5560ef5ba722553c99b8831f9c1d7c43ddddd65349561e. Anonymous curl redownload verified archive, internal manifest and all40323 payload sizes/SHA, exact member set with no duplicates. Manifest preserves original artifact audit-evidence source commits explicitly, rather than pretending untracked raw files were Git blobs. Mac can now independently perform its requested absolute phase1 inference without regeneration. V46 remains rejected, no new training or dev/final use.


## Exact symmetric logit decomposition

Pinned ad0ed56, saved-training-arrays only, no model inference or loss choice. For e=r*u and a=u dot w, delta logit=.5*(r-rquiet)*(a+aquiet)+.5*(r+rquiet)*(a-aquiet). Symmetric norm/direction terms sum exactly (maxerror7.11e-15); avoids order-dependent norm-first attribution. All240 quantile/mean groups recomputed from saved arrays. Latest common endpoint target noise15 mean delta v46 -3.84923 = norm -1.04510 + direction -2.80413; v47 -4.54093 = norm -.44694 + direction -4.09399. V47 1785/2016 target windows decrease. This is an algebraic description, not causal mediation or a counterfactual intervention. Both norm and direction terms matter; normalized cosine alone omits the original classifier scale. No evidence yet that a specific consistency/contrastive loss improves held-out FAR. Original report byte-preserved; Mac independent phase1 work remains separate. No training/newdev/final/promotion.


## Single-keyword denominator compatibility probe

Pure synthetic 2D numerical example, no model arrays or training. Target-only anchors with positives other targets and denominator all nonself (including Unknown) yields loss .6108831 / gradient norm .7786556, while target-only denominator gives identically0 for two targets. Closed-form normalized-vector gradient matches finite differences, permutation and positive-rescaling invariance pass; zero-norm/one-target guards exercised. Unknown pairs are never mutual positives. Explicit counterexample retained: fully collapsed normalized embeddings have loss log3 and zero gradient even with all-nonself denominator. Thus avoiding the single-keyword denominator degeneracy is necessary but not an anti-collapse or training-success guarantee; do not select a loss/temperature/weight or discard main binary supervision based on this toy. Mac independent frozen phase1 measurement remains pending, no newtraining/dev/final/promotion.


## Candidate-independent paired schedule feasibility

Verified all672 conditioning IDs×3speeds×4phrases×5arms exactly present. For each speed even the intersection across all IDs/phrases/arms has common complete phase1 endpoints: .9=291..336/step3 (16),1.0=297..339 (15),1.1=309..339 (11), using Mac19aed75 inclusive-end geometry. Therefore any two distinct IDs can share a valid endpoint at matched speed without filtering/mining. Separate local seed227726 generated50blocks of twoIDs/all40views; all2000 sampled originalraw file/array hashes and150×40window hashes authenticated and full ordered trace retained. No inherited sampler mutated, no model inference, no adopted batchsize/loss/coefficient or training. This prepares reproducible cross-ID/cross-render pairing while Mac independent representation measurement proceeds; IDs do not establish physical speaker diversity.


## v47 paired phase/reset preflight with CPU control

Pinned c25b94b first/last50block schedule =80views. First GPU execution stopped on window-vs-stream probability mismatch5.42402267e-5 above fixed rtol1e-5/atol1e-5; original failure log preserved. Historical compare_streaming deliberately usesCPU, whereas first new preflight had usedGPU. Repeated same80views at same tolerance with CPU2threads, no weights changes or threshold adjustment: all80 pass, max probabilityerror1.1324882507324219e-6, repeat stream reseterror0. This supports the prepared phase1 schedule on these80CPU cases, not universal GPU equivalence; GPU discrepancy remains recorded and no causal numerical-kernel diagnosis claimed. No newmodel training, no all-source/window equivalence, no newdev/final or acceptance.


## Rendering consistency implementation and mathematical precheck

Implemented only training-time normalized300dim same-source cosine loss,32pairs across2IDs/4phrases/fivearms, no deployment layers. Independent NumPy analytic derivative matches TensorFlow autodiff (max6.94e-18) and central differences (1.29e-11). Equal/zero/tiny vectors finite; zero vectors loss1/gradient0 explicitly retained. Fully constant representation plus zerohead and weighted-prior bias is stationary even with pair BCE, norms<3e-16; nonzerohead/opposite labels yield opposing representation gradients. Tiny-vector test caught implicit float32 conversion of epsilon before float64 cast; constant now constructed directly in inputdtype, no tolerance relaxation. Corrected proposal paired40 census before training:positive10/negative30, not8/32;32 counts consistency pairs. Pair weighted odds1/60 used in collapse example. Real initialization main+pair gradient scale, head disconnection and fullsampler preflight stillpending; no trainingstarted or lambda chosen.


## Actual immutable initialization main+pair gradient preflight

Source b5e1d39, execution3dfda11, GPU actual50 inherited main128 pluspaired40batches. Original reconstructed encoder and classifierhead SHA a41cc84a... match, BN frozen. Optimizerupdates0 and allmodel/BN variables exactunchanged. All50 orderedpair source/end/window hashes match prior e3e701f precheck; mainbatchfunction AST exactly matches originalv47 source. Saved allactualinput/label/weight tensors and main/pair/supervised/consistency gradients. Independent arrays audit recomputed all50aggregate/pervariable norms, headDgradient zero, BCEsum, inputtensorhashes and fixedscale rule. Main historical tensors were not saved, so source identity is not historicaltensor replay.

Median encoder supervisednorm256.3368631, Dnorm1.54086058, R166.3595439. Preregistered cap/decade rule yields lambda1.0; scaledperbatch ratios .0018603..0189365, cosine -.27253...65270,17/50conflicting batches. Allpairedembeddingnorms>=6.003955, no epsclamp/zero embedding case atinitialization. This is initialization scale only, not guarantee latergradientbalance/anti-collapse/efficacy. Originalreports preserved; actualtensors and gradientNPZ locallyretained for Mac independentreview. Formal matchedtraining integration/smoke/executionpins stillpending, not started.


## Probe original-array handoff and explicit snapshot gap

Mac requested same-instance H5 only after original probe process had exited; no H5 or fullvariableinventory was saved, so cannot supply such an original snapshot. Issue6816372029 disclosed gap; no seeded rebuild or probe rerun substituted. Future probe now saves the actual unchanged instance and full shape/dtype/value-byte inventory, but originalreport remains unchanged. Original4files (report, gradientsNPZ, batchesNPZ, initialization) released at https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-paired-init-gradient-probe-v1.tar.gz size40133649 SHA29928a8c77f5834864d094559289c8c832b906ece42b67edb20bdc8a3db58d0a. Anonymous redownload verified archive/internalmanifest/all4payloads; snapshot absence explicitinmanifest.

Matched training helper implemented but not yet connected/executed: both arms explicit supervised and D backwards, return both gradientnorms so control D evaluation not silently optimized away, one shared optimizerupdate, only coefficient0/1 difference. Formaltrainer integration and smoke remainpending; no trainingrunstarted.


## Paired smoke export pass but strict numerical comparison failure

Control0/treatment1 each2steps,39.0/42.9s,selected1,54312B. Finite/frozenBN/INT8 exportcontracts pass;exact2optimizerupdates, initializationinventory andorderedinputsamplingtrace identical. Actualfirstmain tensor matchesoriginalprobe. However strictfirststep comparisonrtol1e-5/atol1e-5 fails:controlpairBCE8.28726387 vsprobe8.28610420, and twoarms firststeploss/gradientnorms also differ slightly despite equalinputs/initialvalues. Originalreports/fullcomparison preserved;formalnotready. Need fixed matched numerical execution strategy and newfreshsmokes, do not relaxcomparison orchangeoriginallambda torescue. Noformalstarted.

Certified derivedprobeinitialization fromactualpre-update controlsmokesnapshot:29nonheadvariables exactlyoriginalinitializer,headSHAa41cc84a exact;not originalhistoricalinstance. Package https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-derived-probe-initialization-v1.tar.gz size166656 SHA7e8084d19117592e713c8e6014e13e3c867b3f2c085ea0f01e4e6f7baf9bfe50. Originalinitializer/certifiedheadNPZ/derivedH5/inventory/certification andmanifest;anonymousdownloadall5payloadsexact. Snapshotprovenancegapremainsdisclosed, numericalidentitynotusedtohideGPUnumericaldrift.


## Deterministic full-FP32 first-batch replay

Certified initialization and saved first actual batch replayed without updates on CPU and CUDA, twice per device. Each device repeats all losses and full supervised/D gradients exactly; all variables unchanged. CPU/CUDA arrays pass original rtol1e-5/atol1e-5 elementwise comparison (max absolute supervised gradient difference1.90735e-5, relative tolerance also applies). This is a new explicit execution policy (op determinism enabled, TF32 disabled), only first batch, not retroactive passage of old probe or failed v1 smokes. Original lambda1 unchanged. Matched trainer now records/applies this policy for both arms; fresh v2 two-step smoke scripts prepared before execution. Formal training still pending.


## Fresh v2 matched smokes pass

Execution8858a9a, control/treatment each2updates,41.4/42.7s. Initial inventories and full ordered main/pair sampling trace exact between arms; all pre-update first-step observed losses and encodergradient norms now exactly equal. Main/pair/D losses match new CUDA full-FP32 first-batch reference at original1e-5 tolerances. Both54312B INT8 exports pass finite/frozenBN/shape/dtype audits. This clears fresh smoke engineering checks only, does not erase v1 failure or establish acceptance. Formal120k scripts prepared, execution pins/postprocessing still required before launch; no new/final data accessed.


## Fixed formal execution pins

Fresh v2 smokes pass. Prepared one120k run per arm, original5k checkpoint/PTQ/selection and all inherited main recipe checks. Complete120k ordered main/pair sampling and initialinventory must match before scoring. Sequence fails closed, no automatic retries. Postprocessor uses only original historical corpora and unchanged138-point calibration implementation, never newdev/final or automatic promotion. Execution receipt pins trainer/helper/commands/postprocessors and explicit deterministic full-FP32 policy; originallambda1 and v1 failures unchanged.


## Formal postprocessor path correction during unchanged control run

Found inherited v47 root format `model-gpu-{tag}-ami-extra` did not match fixed formal commands' `model-{tag}`. Corrected only that path before scoring. Original execution pins preserved; separate correction receipt records old/new postprocessor SHA. Both arm root expressions checked against actual command output directories; all other pinned files including trainer/helpers remain byte-identical. No training restart, recipe/selection/gate changes or new-data access.


## Deterministic first-batch arrays handed to Mac

Original CPU/CUDA arrays and reports, original comparison and execution script released as gpu-paired-numerical-replay-v1.tar.gz,325906B,SHA f4ca8bedceefd4234802446d1dd0075391a0700ad1cb1c1fec1ef02c4d4daf09. Anonymous redownload archive/manifest/all6payloads authenticated against original bytes. Enables independent verification of new numerical policy; only first batch, not original50batch re-estimation or retroactive v1 passage. Formal control continues unchanged.


## Mac independent network autodiff and Adam receipt

Mac63c6082 original archive16108076B SHA043a156d8662d29e9d060c80e0004f15926225b0228371214271cc832924bd1b anonymously downloaded; manifest and all4payload sizes/SHA verified. Report pins certified derived initialization and all50actualbatches,0updates/unchangedvariables; allthree currenthelper bytes exact. Independent NumPy output derivatives + CPUencoderautodiff and firstAdam formula checks were executed byMac, not rerun here. FirstMaclosses versus newCPUfullFP32reference pass original1e-5 tolerance,maxabs1.3678674e-6. OriginalGPUgradientrelative-L2 differences remain nonzero and retained, not rewritten as pass. Macsingle-step update reports pass their fixed bounds, not fullsampler/120k/acceptance evidence. Formal sequence314584/start5559859 and control314586/start5559860 confirmedlive; norestart.


## Optional strict Mac full-gradient mismatch retained

Mac3564de9/993929d first-batch cross-platform comparison reports all_passed:false:2/21841 supervisedcomponents exceed originalatol/rtol1e-5 versus bothLinuxCPU/CUDA; losses and allDcomponents pass. Independentderivative and bounded eager/graphfullTF controls retain same2mismatches, exactrepeat,0updates/modelBNunchanged. Platform/kernelcause not isolated; no tolerance adjustment, repeatednumericalrescue or formalrestart. Archive324674B SHA889ca05f52ca503cd62cd90e93132446d56e0a21e4804905b69d496bb71fcd69 anonymouslydownloaded and manifest/all3payloadsverified. Originalcomparison/TFcontrolreports copied byte-identically intoresults; arrays retainedlocal. LinuxCPU/CUDApass is not universalcross-platformequivalence; originalfailuresremain. This optionalcheck ends here while fixedformaltraining continues.


## Formal control completed; original sequence advanced to treatment

Control314586/start5559860 completed120000updates,4566.5s,originalselectionstep5000. OriginaltrainingSHAaaf1df422ed740d4ab9d483de506ceaf78ce412f5825cb89be187fe2902d27d1;54440BmodelSHA51e1e513b2c80d59091e42acf3dc79d4a2e4c89ca20a91de3519a0dd34989a19. Completionexportaudit finite/frozenBN/INT8shape/dtype passes. Bothsampling/loss120000rows,reporthashes/snapshot/inventoryverified;initialinventoryexactv2smoke. This is controlarmcompletion/exportonly, not acceptance or matchedall120ksamplingequalityyet. Sameoriginalsequence314584/start5559859 automaticallystarted treatment331749/start6018884,coefficient1,once120k. No restart,newdev/finalaccess orpromotion. Historicalscoring remains scheduledafterbothrunsandfullinputaudit.


### Paired formal treatment completion and full input audit (2026-11-12 UTC)

Original treatment completed 120000 updates in 4614.6 s, selected step5000 by unchanged rule. Original training JSON and completion export audit retained byte-for-byte. Model 54440 bytes, SHA256 `64f8fe18c5362edb0a9855da866aa0e63891bbd1bc7bee98ac16d488a6d8db2f`. Finite weights, frozen BN, INT8 [1,3,40] → UINT8 [1,1] export contract passed. All initialization/sampling/loss report hashes and 120000 trace rows verified. Original sequence advanced to control historical postprocessing without restart. Full cross-arm input audit passed: exact initialization inventories, all 120000 ordered main/paired draws and first-step observations. This proves matched inputs, not model quality. Fixed historical 138-point gate pending; no new dev/final access or acceptance claim. Mac control fallback interpretation retained.


### Paired control historical gate rejected (2026-11-12 UTC)

Original control postprocessor finished normally and sequence advanced to treatment. Fixed 138-point reused historical development grid: 0 eligible; max macro 0.6458333333333333; minimum negative false clips 6. All 467 saved score traces independently hash-match their original reports and model. Original reports retained byte-for-byte. Control rejected; no threshold/grid rescue or untouched development/final access. Treatment historical evaluation remains running. v30 retained.


### Paired treatment historical gate rejected; matched experiment ended (2026-11-12 UTC)

Original sequence session7001 exited code0; treatment historical postprocessor ended normally. Fixed138 points: treatment0 eligible, maximum macro .5458333333333334, minimum negative false clips1; control0 eligible, max macro .6458333333333333, minimum false clips6. Extrema are not a joint operating point. Both original selected checkpoints are5000. All467 treatment score traces hash-match original reports/model; originals retained. VoxConverse at original per-arm historical thresholds:0 events/8.173672725694447h, one-sided upper95 .3665099367309777/h in both, insufficient to pass joint synthetic gate. Both rejected, v30 retained; no new dev/final, threshold changes or coefficient sweep. Single matched seed does not establish universal benefit or failure of consistency learning. Next intervention must address phrase discrimination without collapsing target coverage and be specified before training, using these old failures only; no automatic new training launched.


### Phrase ranking formula/metadata preflight (2026-11-12 UTC)

Training-only loss implemented with explicit metadata-derived10×4 indices; checks cover all40 unique indices. Independent NumPy float64 analytic value/softmax-minus-onehot gradient passed4 cases including equal and large logits; maximum gradient absolute error1.3877787807814457e-17. All40 finite-difference components and common-logit shift invariance passed. Authenticated raw report used for one metadata block; permutation accepted, duplicate/wrong-speed/unknown-rendering rejected. CPU formula check only; no claim of actual network gradient, full sampler/endpoint authentication, CUDA or optimizer smoke. Next: authenticate all50 original saved paired batches and run no-update actual-network ranking gradients with existing fixed coefficient1. No formal training started; original failures and data quarantine unchanged.


### Phrase ranking actual-network initialization probe (2026-11-12 UTC)

RTX4090 deterministic fullFP32 probe completed50 saved original batches, authenticated2000 raw paired windows and500 metadata groups/common legal endpoints; all main and paired tensors from pinned original NPZ. Certified derived initialization explicitly retained as derived, not original missing probe instance. Zero optimizer updates; all network/BN variables exactly unchanged. Supervised/ranking full gradients connected and finite; saved arrays SHA5180cb7da05895b7dd9ea80cb9380c5c6de6984ccfe0650a445b8c127eab3159. Coefficient remains predeclared1, no scale retuning. First script attempt stopped before gradients due to confusing raw_file_sha256 with raw_sha256; original failure log retained and separate correction receipt. Corrected run verifies both plus dtype/shape. Actual training integration and matched smoke remain pending; no formal candidate or acceptance claim.


### Phrase ranking fresh matched smoke passed (2026-11-12 UTC)

Explicit phrase-ranking mode integrated, metadata indices checked per batch, full supervised/ranking backwards in both arms and one combined Adam update. Fresh control/treatment2 updates completed44.1/38.8s on4090. Actual initialization inventory/full two-step sampling exact; first observations exact excepttotal; losses match saved actual-network probe at original rtol/atol1e-5. First ranking1.5313886404037476, ranking encoder norm12.62767219543457, supervised encoder norm301.43511962890625. Both finite/frozenBN/INT8[1,3,40]→UINT8[1,1] exports54312B passed. No formal training started; next prepare pinned single120k-per-arm commands and unchanged historical-only gates.


### Phrase ranking formal execution frozen (2026-11-12 UTC)

Fresh2-step engineering preflight passed. Commands/source pins prepared for one120k control and one120k treatment, coefficients0/1, no rendering D. Both explicitly calculate supervised/ranking backwards; one Adam update per step. Original initialization/main sampler/pair draw schedule/frozenBN/checkpoint/PTQ unchanged. Fresh control required rather than treating prior completed control as automatically interchangeable. Sequence fails closed, authenticates full120k cross-arm input/trace hashes before original historical-only postprocessors. No new dev/final access, restart or acceptance claim. Delivery of these pins through Issue13 precedes launch.


### Phrase ranking control completed; treatment started (2026-11-12 UTC)

Fresh control completed120000 updates in4738.8s; original selection5000. Original training.json copied unchanged. Both paired trace files contain120000 rows. Export audit passes finite weights/frozenBN and INT8[1,3,40] to UINT8[1,1],54440B; model SHA51e1e513b2c80d59091e42acf3dc79d4a2e4c89ca20a91de3519a0dd34989a19 (byte-identical to prior consistency control; this does not replace the scheduled cross-arm audit). Original sequence launched treatment PID377186; actual starttime recorded in receipt. No restart, quality acceptance or new dev/final access. Full treatment/input audit/historical gates remain pending.


### Phrase ranking treatment completed and full input audit passed (2026-11-12 UTC)

Original treatment completed120000 updates in4708.6s; original selection5000. Original training.json preserved unchanged. Export54440B SHA598f9245fc02bd3262a286c89e5c50bcfe2fef6195fc265bc5eb11f05a7af591 passes finite/frozenBN/INT8[1,3,40] to UINT8[1,1] contract. Original sequence full120k input audit passed: actual initialization inventories and complete ordered sampling traces exact; each arm120000 sampling/loss rows with authenticated hashes and saved snapshots; first observations exact excluding total. Historical-only postprocessing now runs in original sequence; no quality acceptance or new dev/final access.


### Phrase ranking control historical gate rejected (2026-11-12 UTC)

All original historical reports copied byte-for-byte;467 saved score traces authenticated. Original fixed138-point joint calibration has no viable operating point; control rejected. Original sequence continues treatment historical evaluation. No new dev/final access or v30 replacement.


### Phrase ranking matched experiment rejected (2026-11-12 UTC)

Original sequence finished both historical evaluations. Treatment original reports copied byte-for-byte and467 saved score hashes authenticated. Both fixed138-point joint grids have0eligible operating points. Full comparison receipt preserves separate macro/false extrema and limitations; no new-data rescue, promotion or Hi ESP equivalence. Retain frozenv30 and quarantine untouched dev/finals. Next research hypothesis must address retained failure evidence rather than sweep this coefficient.


### Phrase ranking full same-threshold diagnosis (2026-11-12 UTC)

Authenticated both original historical calibration JSONs and retained all138 matching thresholds, identical group exposures and background hours. Treatment false clips lower at138/138; macro higher106/lower32; background events lower64/equal39/higher35. At94 points macro improves without increasing either false clips or background events. This differs from the earlier consistency pair (macro lower everywhere), but all original gates still reject both arms. Correlated thresholds and one seed are descriptive evidence, not independent trials or acceptance. No threshold chosen, inference rerun, new dev/final access, coefficient sweep or training started. Next diagnosis should identify the retained confusion/background failures using existing authenticated traces before specifying any new intervention; frozenv30 remains unchanged.


### Phrase ranking authenticated failure inventory (2026-11-12 UTC)

Replayed detector events on all existing saved score traces, not model inference:467 per arm, each NPY hash checked against byte-identical committed/original reports. Reconstructed every original138-point group positive/pregeneration/negative count, macro and background event count exactly. Compact inventory retains per-record source/PCM identity and full-grid arrays; thresholds are not reselected. At original maximum .99609375 control residual negatives6 (4Nick/2Next), treatment2: p334-1.1-hey_nick and p299-1.1-hey_next. Treatment positive presentation triggers6/16 Kokoro and29/60 VCTK, versus control8/16 and17/60; both background0. At grid .5 treatment10/16+45/60 vs control10/16+40/60; treatment negative24 vs control29, background6 vs8. Thus aggregate discrimination improves while some target voice coverage remains absent; highest-threshold Kokoro recall regresses. Neither endpoint is a selected or accepted working point. These old validation clips remain excluded from training/mining, including the two identified failures. Before a new objective is specified, investigate training-only paired support and temporal discrimination, keeping validation provenance separate; no new candidate launched or new dev/final consumed.


### Selected-checkpoint float training-pair fit (2026-11-12 UTC)

RTX4090 deterministic FP32, original saved first50 training batches (2000 windows/500 four-phrase groups), authenticated NPZ/probe/pack and selected5000 H5 via original export audits. No updates; all variables unchanged. Control target strict-top1 496/500, ranking500/500; ranking minimum target-minus-max-negative logit margin .1454281807. At fixed .99609375 window probability: control positive368/500, negative1/1500; ranking positive322/500, negative0/1500. At .5:480→468 positives,5→5 negatives. Relative ordering improves while absolute target coverage falls on these reused training windows; ranking alone does not establish calibrated absolute detection. Float isolated-window diagnostic is not streaming INT8, independent validation or evidence of a unique mechanism. First draft rendering labels used positional rather than sorted metadata order; corrected to actual authenticated group metadata; correction receipt retains original/corrected log hashes. Global metrics unchanged. No coefficient retune/new training/new dev/final or promotion. Next distinguish absolute calibration and temporal aggregation on training-only evidence before fixing a new intervention.


### Training-only bounded temporal/INT8 diagnostic (2026-11-12 UTC)

RTX4090 selected5000 float windows and reference-kernel full-context INT8 streams on first2 original paired batches:80 authenticated training records, five adjacent endpoints each. Actual stream phase endpoint2 mod3 is explicitly aligned, one feature frame before original paired endpoint0 mod3; not claiming original saved-window exact phase equality. All raw file/tensor identities verified, model/H5 authenticated, float variables unchanged,0updates. No scores selected sample membership. At fixed .99609375, all20 target records: control float any18/last18/mean18 and INT8 any19/last19/mean17; ranking float19/18/15 and INT8 18/17/12. Restricting descriptively to18 records where all5 windows meet inherited conservative full-container support: control float16/16/16, INT8 17/17/15; ranking float17/16/14, INT816/15/10. All60 negative records remain below.5 on these five outputs in both modes/arms. This small reused training diagnostic suggests temporal persistence is a distinct remaining issue, but float-window/full-context INT8 differences cannot uniquely identify quantization; container support is not phoneme truth. The prior v39 direct-five-score failure remains binding evidence against blindly repeating it. Next isolate float full-stream/context effects before specifying a different intervention; no new training, threshold selection, dev/final access or acceptance. Full per-record matrices retained in4090-phrase-ranking-temporal-probe-v1.json.


### Full float streams and explicit phase correction (2026-11-12 UTC)

Original80 raw training records replayed as float full streams on4090, with exact reset repeat and nonstate variables unchanged. SavedModel nonstate shape/dtype/value-hash multisets exactly match original selected H5 variables; current SavedModel files and full saved score archive authenticated. Among18 five-window complete-support targets at .99609375: control float-stream any17/last17/mean15, INT8 17/17/15; ranking float-stream16/15/11, INT816/15/10. Float-window unaligned maximum differences .38249/.18011 were not evidence of context truncation: existing validation_window_phase=1 contract requires shifting the window by1 frame relative to stream endpoint. Additional explicit aligned-window run on identical records passes original rtol/atol1e-5 for every score in both arms (ranking max2.980232238769531e-7). Original unaligned matrices and their limitations retained rather than overwritten. Thus observed float/window discrepancy is explained by phase in this bounded probe; ranking persistence failure remains in full float, so quantization is not the main demonstrated cause here. Full float arrays retained locally with report SHA receipt. No training/dev/final/threshold changes or acceptance. Prior v39 failed sequence BCE remains distinct evidence against simply repeating that objective.


### Absolute temporal tail-risk proposal and bounded formula/feasibility preflights

Preregistered a distinct surrogate, weighted log-mean-exp of five individual frame BCEs, fixed coefficient1 against fresh0 control; retain last-view pair BCE and phrase ranking1 in both. This anchors absolute logits and emphasizes difficult frames, unlike shift-invariant ranking and v39 BCE of arithmetic probability mean; no claim of guaranteed detector improvement. New paired seed237726/common endpoint schedule explicitly changes both fresh arms, so prior completed arms are not controls. Formula independent float64/analytic/finite-difference checks passed4 cases/10components, max gradient error2.220446049250313e-16, mean/max bounds, permutation and absolute shift sensitivity. All40320 metadata rows/fourphrase-fivearm groups checked; all676368 distinctIDpair/speed intervals feasible, min7 legal endpoints, no source excluded. Raw authentication/RNG replay/actual-network gradients/smoke/execution pins still required before formal training. No updates/newdev/finals, coefficients not estimated, historical rejection/v30 unchanged.


### Temporal tail-risk actual-window/network preflight (2026-11-12 UTC)

Implemented independent training-only sampler seed237726,40records×5windows per draw, every endpoint complete-container supported, fail-closed on infeasible groups. RTX4090 zero-update probe at certified derived initialization authenticates50draws/10000raw windows/500phrase groups, exact independent RNG/metadata endpoint enumeration, file/tensor/window hashes and float tensors/labels/weights. All paired-BCE+ranking and temporal gradients connected/finite, temporal encoder/head norms nonzero across50; all model and BN variables unchanged. Full logits and gradients saved locally in authenticated arrays NPZ; report/traces committed byte-identically. Coefficient remains1, not retuned from norms. Scope is paired supplementary gradients, NOT main128/full optimizer integration, smoke/export, formal execution or acceptance. Next integrate complete trainer with explicit backwards and one update, then fresh matched2-step checks. No newdev/final, source filtering or model promotion; v30 retained.


### Temporal-risk complete trainer fresh matched smoke passed

Integrated explicit temporal-tail-risk mode retaining original main sampler and last-view pair BCE/ranking1. Both arms calculate base/ranking/temporal backwards, one Adam update. Fresh4090 two-update arms pass exact actual initialization inventory, full2draws, inherited main draw hashes and first observations except total. First losses main12.048437118530273/pair13.043477058410645/ranking1.5453941822052002/risk12.64472770690918; temporal encodernorm205.9175567626953. Pair losses match original no-update probe at original1e-5; full source/window traces replay original first2 temporal draws exactly. Sampler change since probe only retains first50trace metadata. Both finite/frozenBN/INT8[1,3,40]→UINT8[1,1] exports54312B pass. Exactly2 optimizerupdates per arm; no candidate acceptance. Raw training JSONs/audits/comparison retained. Next fixed120k commands, full execution pins, cross-arm audit and unchanged historical-only postprocessing before formal launch. No newdev/final, threshold/coefficient changes or v30 replacement.


### Temporal-risk formal original sequence launched

Fixed commands fc50373 and214 execution source/data/preflight hashes c384f78 pushed and delivered via Issue13 before launch. Fresh verifier passed; original sequence session33314 PID409484/start7685669 and control trainer409487/start7685681 observed live. One120k per arm, control then treatment, full120k input audit before unchanged historical-only postprocess. No auto-restart or newdev/final route. Startup receipt preserves exact argv/PID/start identity. This is execution startup only, not completed gradient updates, export or quality acceptance; v30 retained. Do not rerun freshness verifier or launch this sequence again after outputs exist.


### Live temporal-risk control startup authenticated

Original sequence409484/start7685669 and trainer409487/start7685681 still live at2026-11-12T12:36:13UTC;1706 complete loss rows observed. All214 fixed execution hashes unchanged. Actual initialization inventory exactly matches fresh smoke, allfirst50 temporal draws exactly match original independently authenticated zero-update probe, first actual loss/norm observation equals control smoke exactly. No restart or modification of pinned training files. Bounded startup evidence only; full120k pair audit and historical quality gates pending. Newdev/finals untouched, v30 retained.


### Mac independent support feedback received while temporal control runs

Issue13 comment6823741277 /Macf11ad5e received and original report imported byte-identically. All6 referenced local report hashes match. Positive support proposal is separate future preflight, not an amendment to live temporal experiment. Original enumerated candidate grid has no full-original-PCM candidate for34180/60000Piper,200/3000Kokoro,30/300Qwen rows. This does not prove no endpoint outside that grid exists, nor target/phoneme truncation. Before any replacement experiment, enumerate feasibility and explicit handling without silently removing/reweighting sources, and distinguish full long-command PCM from target support. Preserve original temporal pins, no parallel formaltraining/newdev/final use.


### Temporal-risk both formal arms completed; full input audit passed

Both original arms completed120000 updates and exported54440B INT8 models. Control selected5000/modelSHA56b89e620147247c46399b610e183bb89357794e9ec17020183372d85b9e4a06; treatment selected20000/modelSHAde65867f7134b82c0b8b881617d1ade96415f2792409c0d591fbc240d96bb896. Original training reports preserved byte-identically in5dcbf8c/d10630b. Both full loss traces contain continuous steps1..120000 and finite numeric values. Original sequence full cross-arm audit passed: exact actual initialization inventory, all120000 ordered input draws and first observations aside total. Shared traceSHAf619dfd7f3e417aa620f0b7c4c969045882f8d80d968ee49336c77f26e8adf4f. Audit original copied byte-identically; hashes independently rechecked. Original session33314 now runs control historical postprocessing (observed15:54UTC); treatment scoring pending. Training/input equality do not establish quality acceptance. No newdev/final access or v30 replacement.


### Temporal-risk matched formal sequence failed historical joint gate

Original session33314 exited0 after both120k arms, full input audit and both unchanged historical postprocessors. Each fixed138-point grid has0 eligible points and no selected threshold. Control maxmacro0.66875/minimum synthetic falseclips1; treatment maxmacro0.8166666666666667/minimum falseclips10 (extrema need not share threshold). Temporal treatment does not solve joint persistence/discrimination requirement and is rejected for promotion. Original evaluation/calibration/export/streaming JSON reports copied byte-identically with source paths, sizes and SHA256 in4090-temporal-tail-risk-formal-results-v1.json. No newdev/final data consumed; v30 retained. Full objective remains unachieved.


### Temporal-risk failure localized from authenticated saved traces

Authenticated467 original score arrays per arm (380synthetic+87background); reconstructed all138 original calibration aggregates exactly. At every matched threshold treatment macro is higher and synthetic falseclips are also higher (138/138); background eventcount is higher121/138, equal17, lower0. No same-threshold recall gain without increased falseclips/background. At original maximum0.99609375 control residual is p245-1.1-hey_next; treatment10 negatives span Nick/Next/Mix and Kokoro/VCTK voices. This is a broad discrimination cost, not proof of a unique mechanism or phoneme defect. Scripts preserve every threshold and original score/report hashes, no inference, training or threshold selection. Next hypothesis must explain this tradeoff rather than sweep temporal coefficient or relax gate.


### Generation-reference proposal rejected before gradients: inherited positives overwritten

Integration review found the63300-row inherited main-positive sampler is fully overwritten by full_source.first80 then complete/US/factorized replacements. Executing actual batch AST on symbolic tensors and extracting replacement slices from source proves arbitrary changes to all32 inherited positives leave entire final batch/labels/weights exactly unchanged. Surviving positive slots:16fixed full-source,8US,8factorized,0inherited. Metadata/reference feasibility receipts remain valid only for inherited draws; they do not characterize current network positive coverage. Prior proposal marked rejected, no gradients/smoke/formal launched. Next inspect actual surviving packs before another intervention. This corrects prior main-sampler interpretation, no model/data/threshold changes or v30 replacement.


### Actual surviving positive support census; avoid repeating v38/v39

Authenticated current surviving pack NPZ hashes and all included positive records: fixed full-source258variants/16slots, US81variants/8slots, factorized10080variants/8slots. Every base crop meets original single complete-container ledger; none of the base crops supports preceding5 stride3 contexts. Fixed full-source has only its258base endpoints in current pack. US temporal sampler has5304/6276five-complete candidate endpoints; factorized683549/804509, all their variants have at least one. These are unweighted variant/endpoint counts, not actual source-balanced probabilities or phonetic/physical support. Do not infer fixed sources lack feasible later endpoints elsewhere. Historical v38 already tested same-source temporal0..14 jitter and failed; v39 directly supervised five-score probability mean and failed. Census alone does not justify repeating either or previous temporal-risk loss. No training, scores or newdev/final accessed. Next distinct hypothesis must target actual surviving inputs and explain prior failures.

### Common-noise construction preflight and matched control definition

Original first50 paired schedules rendered100groups/2000windows without trimming any source PCM. All100 scheduled groups satisfy conservative full-source+RIR support. Quiet raw prefixes reproduce authenticated original features exactly; component arrays and source hashes retained. No network inference, optimizer or newdev/final access. Report `4090-common-noise-rendering-preflight-v1.json` is construction evidence only.

Matched independent/common preflight now defines phrase-specific independent seed domain before scoring, identical source buffers/endpoints/RIR and a headroom scalar jointly computed over both conditions and all four phrases per nonquiet arm. Quiet original PCM stays exact. This removes a separate-condition headroom change from the nuisance comparison. Original historical training controls cannot substitute for this new matched control. No formal run launched; next requires completed paired artifacts, independent component audit and original-init zero-update network probe.

Matched construction completed exit0:50 batches,100 groups,2000 windows per condition, all scheduled groups retained. Both arrays shape[50,40,150,40], finite. Quiet PCM matches authenticated original, quiet/RIR features identical across conditions; original source arrays reauthenticated per group. Hashes and original report saved in `4090-matched-noise-rendering-preflight-v1.json`. No network inference or optimizer updates; component audit and original-init probe remain outstanding.

### Matched noise component audit and RTX4090 zero-update probes

Independent audit reconstructed all4000 saved PCM arrays exactly from authenticated source signals/noise/RIR/headroom. All100 independent seed groups reproduced, shared vectors byte-exact; no nonquiet group headroom changed relative to original common rendering in this sample. Window archives hash-verified. Component audit does not itself regenerate frontend features.

Both independent/common first50 probes completed exit0 on RTX4090, same explicitly derived initialization SHA21f53fee... (not the missing historical original snapshot), fixed BCE+ranking1, no optimizer. Original reports copied byte-identically. All variables includingBN unchanged, all gradients connected finite, main BCE exactly identical across conditions, paired logits changed (max absolute difference0.5597888231). Negative supervised/ranking cosine batches27/50 independent and29/50 common; median cosine-.03310/-.02684, median norm ratio.03019/.03016. These small descriptive differences establish neither improvement nor a unique mechanism; no coefficient fitting or arm selection. Scripts and hashes in matched-noise results. No newdev/final or promotion; v30 retained.

Next: implement actual paired sampler insertion with auditable source schedule, regenerate features through that runtime path and verify emitted network tensors; fresh matched two-update smoke plus pinned initialization/input audit before any formal training. Current offline probe is not a substitute for that integration check.

### Actual matched-noise sampler integration and fresh two-update smoke

Added opt-in `--paired-noise-condition independent|common` at actual paired training input; requires phrase-ranking/coefficient1. Original sampler runs first, preserving its local RNG, source/voice/speed/end draws and original windows. Runtime renderer authenticates original PCM and uses the matched joint-headroom construction, caching raw frontend features per voice/speed. Original head/loss/main sampling remain unchanged. Runtime asserts full source+RIR support; no resampling or silent unsupported fallback.

Both fresh RTX4090 smoke executions completed exit0, two optimizer updates each. Saved actual pre-step network tensors match independently rendered offline windows exactly for both steps/conditions. Main tensors, labels and weights equal original saved first2 batches; cross-condition initialization inventory and ordered source draws exact, first main BCE exact. Paired inputs differ and subsequent gradients/updates differ as intended. FrozenBN preserved; all saved weights finite. Each export54312B, int8 input[1,3,40], uint8 output[1,1]; export size need not match frozenv30. These smoke exports are not deployment candidates. Original training reports copied byte-identically; hashes in `4090-matched-noise-smoke-comparison-v1.json` and export audits.

No formal120k or newdev/final access yet. Next pin execution/source populations and validate full-schedule support before formal independent/common sequence; do not reuse smoke weights or previously trained controls. No quality improvement claim, v30 unchanged.

### Full matched-noise source support and formal execution pins

Authenticated8064 original training PCM source hashes/16k mono/nonempty and all2016voice×speed groups. Every original legal low endpoint covers complete source+960RIR generation container, all legal ranges nonempty. Replayed exact120000-step paired RNG schedule, all2016groups visited, first50match original. Geometry/schedule SHA6d1e5aa342fd387877355be27bed4e30a08a69036f00594e6d41ae96a5632532. This is generation geometry, not phonetic truth or exhaustive frontend rendering; runtime separately asserts source support.

Prelaunch fixed one fresh120000-step run per independent/common condition, BCE+pairBCE+ranking1 both, original24float checkpoint selection/PTQ, inherited main/source populations unchanged. Every current execution/training source and original input-pin set hashed in4090-matched-noise-execution-pins-v1.json. Sequence refuses existing outputs, no retries; runs both arms, full cross-arm source/main/initialization audit and unchanged historical-only scoring. Different paired network tensors expected, source and original windows must match; compare schedule to preflight full120k digest. Reused-development diagnostics cannot justify promotion. No newdev/final access, v30 preserved. Formal results pending.

### Live independent formal first50 prefix authenticated

Original formal sequence session93786 and independent trainer PID478492 confirmed live; no restart. Independently audited first50 complete trace/loss rows against authenticated original saved main batches and offline matched frontend windows. Every ordered voice/speed/endpoint/pack index/original raw-window hash matches original schedule; all50 actual-main tensor hashes exact, all50 paired-network tensor hashes match independent offline rendering. Saved actual step1/2 tensors match elementwise, labels/weights retained; formal initialization inventory equals fresh smoke. Loss steps1..50 consecutive finite. Report explicitly limited to immutable first50 prefix, not full run or quality acceptance. Observed at least598 completed loss rows during audit; newdev/final untouched, v30 retained. Pinned execution sources unchanged.

### Independent formal first5000 checkpoint, training continues

Original session93786/PID478492 verified live, passed5000 and continued beyond5209 loss rows. First progress JSON record preserved byte-identically with external size/SHA receipt. Original float training-validation selected threshold.99, macro recall.4125 (Piper.3416667/Kokoro.4833333), maximum-source confusion rate.00875, background0events/1.204811h. These are reused float selection diagnostics, not final calibrated INT8 quality. Even0events on this duration has Poisson one-sided95% upper about2.49/h, not<=.5/h; provisional float criterion is weaker than acceptance. No manual selection/config changes, no early stop, newdev/final untouched; v30 preserved. Full120k independent and matched common remain pending.

### Independent formal10000 checkpoint has no float-eligible point

Original sequence still live and training beyond10167 updates. Second original progress record10000 copied byte-identically. Under original float criterion ambient<=.5/h and maximum-source confusion<=.01, grid has0eligible points; recorded fallback threshold.994 gives macro.38515625, maximum-source confusion.0104166667 and0events/1.204811h. This is weaker reused-validation evidence and already fails even that float criterion; not formal joint acceptance or final candidate failure before preregistered run completes. No manual reselect/restart/config changes, no newdev/final access. Full run/common pending, v30 retained.

### Independent formal15000 checkpoint remains float-ineligible

Original trainer continues beyond15336 updates, same session93786/PID478492. Preserved third original progress record byte-identically using reusable observation-only checkpoint collector. No original float-eligible point; fallback.994 gives macro.5427083, maximum-source confusion.0328125, background1event/1.204811h=.8300055/h. Higher recall coincides with more confusion/background; this is reused float validation, not a matched common-noise comparison or independent acceptance. No execution source changes, manual reselect, restarts or newdev/final access; planned full training continues, v30 retained.

### Independent formal20000 checkpoint, no float-eligible point

Original session93786/PID478492 remains live; fourth original progress record20000 preserved byte-identically. Grid again0float-eligible points. Fallback threshold.994: macro.48645833, maximum-source confusion.0225, background0/1.204811h (one-sided95% upper2.48647/h). Since5000 the next three checkpoints are float-ineligible; no inference that later or common checkpoints pass. Fixed training continues without early changes. No newdev/final access, v30 retained; full matched formal results pending.

### Independent formal25000 checkpoint retained

Original session93786 remains live; fifth original progress record25000 preserved byte-identically. Grid0float-eligible points; fallback.992 macro.53255208, maximum-source confusion.0375, background2/1.204811h=1.660011/h, one-sided95% upper5.225543/h. Same continuing float failure pattern; no model-quality acceptance or config changes. Newdev/final untouched, v30 retained; await fixed120k independent/common sequence. Record pushed on branch; no repetitive Issue status comment for unchanged conclusion.

### Independent formal30000 checkpoint retained

Same original formal sequence remains live. Sixth original progress record30000 preserved byte-identically;0float-eligible points. Fallback.994 macro.4765625, maximum-source confusion.0197916667, background0/1.204811h (one-sided95% upper2.48647/h). First quarter of fixed120k run completed, no quality acceptance. Execution sources/config unchanged, no restart or newdev/final access; v30 retained. Original checkpoint plus size/SHA audit pushed; unchanged failure pattern does not warrant repetitive Issue comment.

### Independent formal35000 checkpoint retained

Original session93786 still live. Seventh progress record35000 preserved byte-identically;0float-eligible points. Fallback.994 macro.471875, maximum-source confusion.01666667, background1/1.204811h=.8300055/h, one-sided95% upper3.93743/h. No quality acceptance or change in failure pattern. Fixed training continues, no restart/config change/newdev/final access; v30 retained. Original checkpoint/audit pushed without repetitive Issue status.

### Independent formal40000 checkpoint retained

Same original formal session93786 remains live. Eighth progress record40000 preserved byte-identically;0float-eligible points. Fallback.994 macro.54973958, maximum-source confusion.028125, background0/1.204811h, one-sided95% upper2.48647/h. First third of fixed120k independent run completed; no quality acceptance, common arm still pending. No restart/config changes/newdev/final access, v30 retained. Checkpoint/audit pushed; unchanged failure pattern, no repetitive Issue status.

### Independent formal45000 checkpoint retained

Same original formal session93786 remains live. Ninth progress record45000 preserved byte-identically;0float-eligible points. Fallback.994 macro.45260417, maximum-source confusion.02083333, background0/1.204811h (one-sided95% upper2.48647/h). Continuing fixed120k plan without restart/config changes; no quality acceptance/newdev/final access, v30 retained. Original record/audit pushed; no repetitive Issue status for unchanged conclusion.

### Independent formal50000 checkpoint retained

Original formal session93786 remains live. Tenth progress record50000 preserved byte-identically;0float-eligible points. Fallback.994 macro.50052083, maximum-source confusion.02604167, background0/1.204811h (one-sided95% upper2.48647/h). Still no quality acceptance; no restart/config change/newdev/final access. Fixed120k independent/common sequence continues, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal55000 checkpoint retained

Original session93786 remains live. Eleventh progress record55000 preserved byte-identically;0float-eligible points. Fallback.994 macro.4625, maximum-source confusion.015625, background1/1.204811h=.8300055/h (one-sided95% upper3.93743/h). Fixed120k plan continues unchanged, no quality acceptance/restart/newdev/final access; v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal halfway60000 checkpoint retained

Original session93786/PID478492 remains live and beyond60146steps. All60146 written loss rows independently checked consecutive and finite. Twelfth original progress record60000 preserved byte-identically;0float-eligible points. Fallback.99 macro.46302083, maximum-source confusion.0109375, background0/1.204811h (one-sided95% upper2.48647/h). Half of fixed120k independent run complete, common arm not started. Training recipe's preregistered half-run learning-rate transition is unchanged. Only5000checkpoint has any original float-eligible point so far; this does not establish quality acceptance. No restarts/manual config changes/newdev/final access; v30 retained. Midpoint original/audit pushed and Issue13 updated.

### Independent formal65000 checkpoint retained

Original session93786 remains live. Thirteenth progress record65000 preserved byte-identically;0float-eligible points. Fallback.992 macro.49895833, maximum-source confusion.02083333, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues without restarts/manual configuration changes; no acceptance/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal70000 checkpoint retained

Original session93786 remains live. Fourteenth original progress record70000 preserved byte-identically;0float-eligible points. Fallback.994 macro.48411458, maximum-source confusion.02291667, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues without restarts/manual configuration changes; no acceptance/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal75000 checkpoint retained

Original session93786 remains live. Fifteenth progress record75000 preserved byte-identically;0float-eligible points. Fallback.994 macro.42421875, maximum-source confusion.01197917, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k independent/common plan continues unchanged; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal80000 checkpoint retained

Original session93786 remains live. Sixteenth progress record80000 preserved byte-identically;0float-eligible points. Fallback.994 macro.48541667, maximum-source confusion.02708333, background0/1.204811h (one-sided95% upper2.48647/h). Two-thirds of fixed120k independent run complete; no acceptance/restart/config changes/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal85000 checkpoint retained

Original session93786 remains live. Seventeenth progress record85000 preserved byte-identically;0float-eligible points. Fallback.992 macro.4671875, maximum-source confusion.02083333, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues without restart/config changes; no acceptance/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal90000 checkpoint retained

Original session93786 remains live. Eighteenth progress record90000 preserved byte-identically;0float-eligible points. Fallback.994 macro.4734375, maximum-source confusion.03020833, background0/1.204811h (one-sided95% upper2.48647/h). Three-quarters of fixed120k independent run complete, common still pending; no acceptance/restart/config changes/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal95000 checkpoint retained

Original session93786 remains live. Nineteenth progress record95000 preserved byte-identically;0float-eligible points. Fallback.992 macro.49244792, maximum-source confusion.025, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues unchanged; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal100000 checkpoint retained

Original session93786 remains live. Twentieth progress record100000 preserved byte-identically;0float-eligible points. Fallback.994 macro.48489583, maximum-source confusion.02395833, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues unchanged; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal105000 checkpoint retained

Original session93786 remains live. Twenty-first progress record105000 preserved byte-identically;0float-eligible points. Fallback.992 macro.51953125, maximum-source confusion.02916667, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues unchanged; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal110000 checkpoint retained

Original session93786 remains live. Twenty-second progress record110000 preserved byte-identically;0float-eligible points. Fallback.994 macro.525, maximum-source confusion.03020833, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues unchanged; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal115000 checkpoint retained

Original session93786 remains live. Twenty-third progress record115000 preserved byte-identically;0float-eligible points. Fallback.994 macro.56041667, maximum-source confusion.04166667, background0/1.204811h (one-sided95% upper2.48647/h). Fixed120k plan continues unchanged, final5000updates pending before export/common arm; no acceptance/restart/newdev/final access, v30 retained. Original record/audit pushed without repetitive Issue status.

### Independent formal120k complete; original sequence starts common arm

Independent original run completed120000 updates, selected5000,5166.7s. All120000loss rows continuous finite and120000sampling traces present, original report hashes authenticated. Original training.json609013B SHA0ade411379c272b13fa7a92aa8c6949a3396983e57eaf96fe4572fe744532a59 copied byte-identically. Streaming INT8 model54440B SHA83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca; saved weights finite, frozenBN unchanged, interface[1,3,40]int8→[1,1]uint8 verified. Final120000checkpoint itself0float-eligible points, fallback.994/macro.53828125/max-source confusion.034375; original selection retains5000. No promotion or acceptance conclusion.

Same original sequence session93786 has proceeded automatically to common120k arm, live trainer PID504191 observed. No restart. Full matched input audit/historical scoring wait for both completed arms as pinned. No newdev/final access, v30 preserved. Independent completion originals and receipts pushed, Issue13 notified.

### Common formal first50 actual-input audit

Original sequence93786 remains live; common arm passed immutable first50 audit. All source draws/original windows and main tensors match authenticated original batches; paired tensors match independent offline waveform frontend reconstruction. Saved actual network inputs steps1/2 match elementwise, initialization inventory matches fresh smoke, first50 loss rows continuous finite. Prefix-only evidence, not full-run or quality acceptance. No execution files changed, restart, or newdev/final access. Issue13 remains the Mac coordination channel; frozen v30 handoff unchanged.

### Common formal5000 checkpoint retained

Original sequence93786/trainer504191 remains live. First common5000 progress record preserved byte-identically (3956B, SHA30a6def5fca99526aa7ccda8d4defdadc9b0a979eac2e2cbcd71feb26231a918). Zero original float-selection eligible points; fallback threshold.994, macro recall.33125, max-source false rate.015. Background0/1.204811h gives one-sided95% FAR upper2.48647/h, not the required .5/h. This is reused float training-validation only, not INT8/independent acceptance. No same-threshold cross-arm conclusion drawn from separately selected fallback points. Continue pinned120k plan, no restart/config edits/newdev/final use; v30 retained.

### Common formal10000 checkpoint retained

Original trainer504191 remains live without restart. Original10000 record preserved byte-identically (4043B, SHA95733866b38c563acb8af57a15587391e04bc0c90ad39d928f3b814b66f9debe). Three original float-selection eligible points; selected.99 macro.36223958/max-source false.00833333, background0/1.204811h. These are float heuristic selection points only; one-sided95% FAR upper remains2.48647/h, not independent acceptance, no claim of shared-noise advantage. Fixed120k plan continues, v30/newdev/finals unchanged.

### Common formal15000 checkpoint retained

Original15000 progress record preserved byte-identically,4077B SHA b29dbd0805ff4292dcdbe8b6aaf56ec74169f98d09fc793257ef0da1cf98970c. Zero float-selection eligible points; fallback.994 macro.47994792/max-source false.02291667, background0/1.204811h, one-sided95% upper2.48647/h. Higher fallback recall does not establish improved tradeoff or acceptance. Original trainer504191 remains live; observed first13247 complete loss rows continuous and finite. Continue fixed120k without restart/newdev/final use, v30 retained.

### Common formal20000 checkpoint retained

Original sequence93786/trainer504191 remains live. Original20000 record4101B SHA b4daad425154b7992008d7b7001853ff7626024c2d3d9f4ea310ffd24c24fa91 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.47317708/max-source false.01302083, background0/1.204811h, one-sided95% upper2.48647/h. No acceptance, same-threshold advantage, or full-run claim. Fixed120k continues unchanged; no restart/newdev/final use, v30 retained.

### Common formal25000 checkpoint retained

Original25000 record4010B SHA3c5fbabf54d2bc4edabfb6842760c038d44296442dc106903577d0523376acb6 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.45/max-source false.01145833, background0/1.204811h (one-sided95% upper2.48647/h). Original sequence/trainer remains live; fixed120k continues unchanged, no restart or newdev/final access. No quality acceptance or shared-noise advantage inferred, v30 retained.

### Common formal30000 checkpoint retained

Original trainer504191 remains live, quarter of fixed120k updates completed. Original30000 record3997B SHA04ad9b76211cd6f9206d4781c283d8a76119ceebdd025c8f4f98e7ee9b9e9368 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.51145833/max-source false.02291667, background0/1.204811h (one-sided95% upper2.48647/h). No acceptance or cross-arm advantage claim; no restart/config changes/newdev/final use. v30 retained.

### Common formal35000 checkpoint retained

Original35000 record4113B SHA68f08e96d8bd7824b662d80a04b94d52c7359a7a06f0faae1cfd980bb591f198 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.56640625/max-source false.02916667, background0/1.204811h, one-sided95% upper2.48647/h. Higher fallback recall is accompanied by excessive confusion and is not acceptance. All239 execution pins reauthenticated unchanged this observation turn; original trainer504191 remains live. Fixed120k continues without restart/newdev/final use; v30 retained.

### Common formal40000 checkpoint retained

Original trainer504191 remains live; fixed120k plan one-third complete. Original40000 record3989B SHAb9ae18e822e9650c7b47b0e405654515765b4091e9ebfdc4f6179109154201d6 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.42369792/max-source false.0125, background0/1.204811h (one-sided95% upper2.48647/h). No acceptance or cross-arm advantage conclusion. No restart/newdev/final access; v30 retained. Issue13 checked, no new Mac reply.

### Common formal45000 checkpoint retained

Original45000 record3954B SHA8820bcafa703bdbfb69164801dcf5f50c581c1d9f53cee5169496c2a339c1396 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.5234375/max-source false.0234375, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k plan unchanged. No restart/newdev/final use or acceptance claim; v30 retained.

### Common formal50000 checkpoint retained

Original50000 record4014B SHAbdfaed23539b263e129112cbfc1a0436da6aa07f16328d3ec71347d72b996322 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.47890625/max-source false.03072917, background0/1.204811h, one-sided95% upper2.48647/h. Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance claim; v30 retained.

### Common formal55000 checkpoint retained

Original55000 record4018B SHA3c0b42f24850382ee1f28b62247ba9468b874c867bdcfeb604fb46efca8588f9 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.5390625/max-source false.02864583, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use, quality acceptance or cross-arm advantage claim; v30 retained.

### Common formal60000 half-run checkpoint retained

Original trainer504191 remains live beyond half of fixed120k; first60744 complete loss rows continuous finite. Original60000 record4173B SHA725dc394b8c36e1dd6e50b9e4f6fd0b5addf46d47b77279bcbfed0779c4f7c13 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.5796875/max-source false.04895833, background1/1.204811h (empirical.83000552/h, one-sided95% upper3.93743/h). Not qualified or acceptance; 12 checkpoints so far only10000 has float-eligible points. Fixed120k continues unchanged, no restart/newdev/final use, v30 retained. Half-run status synchronized through Issue13.

### Common formal65000 checkpoint retained

Original65000 record4039B SHA4502c8005a4821f2bd4070540786f5be63546460a0b6189d02292c49dfc85bfa preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.51276042/max-source false.02083333, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal70000 checkpoint retained

Original70000 record3992B SHA8a36f103e3e8384e38201ff5cdca4f8c84b77e363ba7607c9f6097afde0e9a74 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.53307292/max-source false.0296875, background0/1.204811h, one-sided95% upper2.48647/h. Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal75000 checkpoint retained

Original75000 record4038B SHAe318d8a4c4a72df38247980a2802503889d85b76cd44d3ef80086e44ebe1743b preserved byte-identically. Zero float-selection eligible points; fallback.992 macro.50625/max-source false.02447917, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal80000 checkpoint retained

Original80000 record3946B SHAd01e3323dbd8065c0d7aa31b5942828d88bdbe04647fb804cc54aef6d2b77f47 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.4703125/max-source false.01979167, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, two-thirds of fixed120k completed without restart/config change. No newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal85000 checkpoint retained

Original85000 record3960B SHA53adb99231a97c02ab755cfbbebc5ceaf0d645290acfe9b32b862be5de0b4d85 preserved byte-identically. Zero float-selection eligible points; fallback.99 macro.59791667/max-source false.04791667, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; higher fallback recall does not establish improved tradeoff. v30 retained.

### Common formal90000 checkpoint retained

Original90000 record3974B SHA118f908d804d48adc2225b4c6b4f9dc89b69b64cc4f2396f7637773c06c23a89 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.60833333/max-source false.04322917, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live; three-quarters of fixed120k completed. No restart/config change/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal95000 checkpoint retained

Original95000 record3943B SHA3fca3e2604f91449737a8fcd593547d4c4dda7122df34e4d62a00080dd7ff11c preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.6171875/max-source false.04322917, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal100000 checkpoint retained

Original100000 record3971B SHAe7ab5c4eca30ac4cc86fecfa7c1b0161eb99f435a572e7e10cbce74ba2eae01b preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.57213542/max-source false.0375, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal105000 checkpoint retained

Original105000 record3926B SHA570400b9305a685a5faa3d157303a486590c69bda177cee7300adfddaeb6a166 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.54296875/max-source false.03333333, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live, fixed120k unchanged. No restart/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal110000 checkpoint retained

Original110000 record4035B SHAc79028f0f33c52ceef810d0f4d1208c5a9dcee731a0e4dcb6b6c1d8207faeca7 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.57473958/max-source false.046875, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live; fixed120k final10000 updates pending. No restart/config changes/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal115000 checkpoint retained

Original115000 record4036B SHA6a6a9726d29421fe3c27a9a57e50c068e4418209ab27bc67ce29e5af0eb893c6 preserved byte-identically. Zero float-selection eligible points; fallback.994 macro.57473958/max-source false.03802083, background0/1.204811h (one-sided95% upper2.48647/h). Original trainer504191 remains live; fixed120k final5000 updates pending before export/full input audit/fixed historical scoring. No restart/config changes/newdev/final use or acceptance/cross-arm advantage claim; v30 retained.

### Common formal120k complete; full matched input audit passed

Common original run completed120000 updates, selected10000,5323.5s. All120000 loss rows consecutive finite and120000 sampling traces present; hashes match original training report. Original training608113B SHA275fd2b99cc2536477e1a52482e6f7eb6608c3bc824aa46fc7af40b4f564b106 copied byte-identically. Streaming INT8 export54440B SHA1f81950ee88f0dc2f70b0ea795b34d7a4b1e5e515e7b2de98b7e0c395bcfff93; weights finite, frozenBN unchanged, input[1,3,40]int8/output[1,1]uint8. Last120000 checkpoint itself0float-eligible, fallback.994 macro.60807292/max-source false.04635417. No quality acceptance.

Original sequence93786 completed full matched input audit:120000 ordered source draws/original windows/main inputs exact across arms, initialization exact, schedule SHA6d1e5aa342fd387877355be27bed4e30a08a69036f00594e6d41ae96a5632532; all120000 paired network batches differ intentionally. Original audit copied unchanged. Original sequence automatically entered independent historical postprocess (PID529964 observed), no restart. Fixed scores pending; newdev/final untouched, v30 retained.

### Independent matched-noise historical scoring complete

Original sequence93786 finished independent historical postprocess and automatically entered common postprocess (PID531568 live observed). All11 original independent reports copied byte-identically with source/hash/size receipt. Unchanged138-point reused joint gate has5eligible points; selected.98828125 macro.57916667 (phone10/16,VCTK32/60), negative false clips0/304 and pre-generation positive events0, background0/8.17367273h gives upper95.36650994/h. This is exploratory historical/reused development, not independent acceptance, no v30 replacement. Common scores and same-threshold comparison pending. No newdev/final access or restart.

### Matched-noise formal sequence completed; common-noise hypothesis rejected

Original sequence93786 exited0; both original postprocess arms completed without restart. All11 common original reports copied byte-identically with source/hash/size receipt. Common fixed138-point reused joint gate has0eligible points, minimum negative false clips2, maximum macro.74375 (extrema need not share threshold). Independent has5eligible points, previously documented.

Same-threshold audit covers all138 original points: common macro lower87/higher51/equal0 vs independent; negative false clips higher138/lower0/equal0; background events lower114/equal10/higher14. No point improves recall without more false clips/background. Reject shared-noise hypothesis/promotion on this matched seed and reused protocol; no independent generalization/unique mechanism claim. Independent historical viability remains exploratory and requires untouched development/independent acceptance before any replacement. Newdev/finals unconsumed by this experiment, v30 retained.

### Reserved development stage preregistered for independent-noise candidate

After completed matched comparison, select only independent SHA83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca for the previously reserved SLR83/dev2-dev3 stage. Adapted unexecuted v36 runner without gate changes; all outputs required fresh. Reauthenticated90 synthetic presentations and86 full background PCM files against original file/PCM hashes; expected reserved score outputs absent. Saved execution/input pins before scores. Original SLR83 source roles, licenses and mechanical quality audits retained; no claim of certified physical-person/pretraining separation. Fixed138 grid/zero confusion and pre-generation events/FAR95<=.5/equal-condition macro>=max(.48125,v30); null on failure. Baselinev30 and officialHiESP same full PCM, separate background exposure; no old/final merging or automatic promotion. This stage consumes reserved development only, not final acceptance.

### Reserved development v1 stopped before inference; explicit execution repair

Original v1 session42179 exited1: all90 child logs rejected missing old endpoint metadata; zero report.json/input.pcm outputs, so no model replay occurred. Preserve logs and SHA receipt. Parent already accepts exact SLR83 synthetic-generation manifest; lower runner did not. Add only exact manifest SHA to lower development validator, retain role/split/PCM/hash/path checks and synthetic boundary disclaimer. Direct validator check accepts pinned manifest and rejects altered hash. v2 uses short fresh /tmp socket paths; v1 outputs retained. New execution pins saved; data/models/gates unchanged, no automatic retry/training restart.

### Independent-noise candidate rejected on reserved development gate

Original v2 session45133/PID537901 remains live in official continuous-background replay (10/86 observed). Completed all90 paired official runs and both Nyx synthetic/background scores. Fixed138-point reserved SLR83/dev2-dev3 calibration yields0eligible thresholds, selected=null; every grid point has>=1 synthetic negative false clip. No fallback, new exposure, rerendering or final access. This reservation is now consumed development.

At original historical thresholds: independent.985 triggers9/18, common confusions1/54, cross-target0/18; frozenv30.91796875 triggers8/18, confusions3/54,cross-target0/18. OfficialHiESP.63 full-presentation triggers17/18, confusions0/54,cross-target0/18. Full-presentation generator/RIR-end median/p95 delays: independent-12.75/172.625ms,v30-25.375/82.625ms,HiESP-248/-109.7125ms. Correlated two voices, generation reference only, not human last-phoneme or hardware latency. Each Nyx background historical threshold has1event/8.48721898h; fixed grid remains rejected. Nine completed original reports copied byte-identically with receipt. Retainv30; official background still pending, no claim of final matched-FAR equivalence.

### Consumed reserved-development fixed-grid failure localized

Authenticated every candidate saved score trace against original report. Southern English female Hey Nick, speed.85 clean variant0, reaches smoothed.99609375 and fires at all138 fixed thresholds including grid maximum; this single clip suffices to invalidate every original point. Other grid-dependent false clips: same source noise variant1(max.86015625), northern Hey Next speed1 variant1(max.56171875), southern Hey Mix speed1 RIR variant2(max.66328125). PCM/score hashes and all-point counts retained. No threshold extension, rerendering, training on these development voices, acoustic phonetic truth or unique causal mechanism claimed. Candidate rejection stands; official continuous background remains original live job.

### Reserved development completed; exact continuous official comparison retained

Original v2 session45133 exited0, all86 official full recordings completed. Summary verifier authenticates per-record original PCM hashes/sample counts/firmware stage104/CRC and identical full8.487218975694445h exposure across three engines. Official request.63/getter.62496001 unchanged across paired/background;0events, FAR95 upper.3529698341/h. v30.91796875 and independent historical.985 each1event, upper.5589421614/h, failing .5/h at those historical operating points. Candidate fixed138 grid still0eligible due synthetic confusion; no selected new-development threshold. Official original full report and authenticated comparison pushed. This is consumed two-voice development, not final acceptance, physical-person generalization, phonetic truth, hardware latency or matched population FAR. No existing finals accessed or training restart; v30 retained despite its disclosed development failures. Next research must preserve rejection and cannot relabel SLR83/dev2-dev3 as untouched or train on its reserved voices.

### Actual matched-noise training fit audit; next bounded hypothesis

RTX4090 zero-update fit check authenticates original first50 actual matched-input arrays and selected H5 via original export audits; all model variables unchanged. Independent target strictly top1 in499/500 training groups, common500/500. At fixed.99609375 window probability, independent362/500 positives and1/1500 negatives above, common364/500 and0/1500. These are reused float training windows, not held-out/INT8 or generalization evidence. Better common training fit coexists with rejected development behavior; simple failure to fit these groups is unsupported.

Propose distinct training-only per-draw independent waveform noise, not repeated shared-noise/temporal/auxiliary experiments. Current renderer caches noise by voice/speed/phrase, so repeated draws reuse acoustic templates. First census actual schedule reuse and bounded first50 matched render/component audit; no new formal run authorized by proposal alone. No coefficient/grid/source mining or development-voice transfer. All historical/reserved rejection evidence retained, v30 frozen.

### Per-draw noise schedule census and bounded rendering completed

Authenticated original120k trace:240000 group draws,2016 unique voice-speed groups, each89..159 draws,237984 repeated beyond first. Reproducible census confirms deterministic cached noise reuse, not a causal memorization conclusion. First50 matched construction completed exit0:100 supported source groups,2000 windows/condition; original complete signals/buffers/endpoints/RIR and joint clipping headroom retained. Quiet/RIR inputs identical800windows;1200noisy windows differ. Fixed window NPZ SHA cb5315f9f6e926e01eac2b76e3890a64e1b941120a17afbc044f05029b63d39c exactly matches prior independent first50 archive, observed despite joint-gain recomputation. Per-draw NPZ SHAc52f2d7e6d7a61baf4fc7fb0aedd752e9eeb68ae5b0c930a32bbc25611b03a55. All full PCM finite/nonempty/unclipped, no model inference/optimizer/development/final access. Component reconstruction/runtime integration/gradient and smoke checks still required; no formal run or quality claim. v30 retained.

### Per-draw component and runtime rendering audits passed

Independent saved-component audit reconstructed all4000 PCM arrays across100groups exactly: fixed/block-specific seeds, full source signals, RIR, noise components, joint headroom; original nonquiet gains unchanged. New PerDrawNoiseSampler renderer uses block-specific seed domain and bounds cache lifetime to one draw. Direct runtime render audit from authenticated source PCM matches every actual rendered network tensor in independent offline archive:50batches/2000windows percondition exactly,4000total. Fixed control retains original first50 SHA. No optimizer/inference/development/final access. Runtime render evidence does not yet prove actual batch RNG/main inputs/trainer insertion; those and zero-update gradient/fresh2-step smoke remain required before formal training. v30 retained.

### Per-draw noise initialization gradient probes and trainer integration

Both RTX4090 conditions completed50 zero-update batches from the same derived initialization certificate (SHA21f53feec03bdd73ae51cbc92091bf723d9de5fba07e3aa337c818ed722fe350); this is not recovery of a missing historical snapshot. All variables/BN unchanged, gradients connected and finite, coefficient1 retained. Main BCE exact across conditions; negative gradient cosine27/50 each, median fixed-.0331038488 versus per_draw-.0129283982. No demonstrated quality advantage. Trainer now has explicit opt-in fixed/per_draw rendering with incompatible interventions rejected. Fresh two-update integration smoke authenticates actual main and paired tensors against independent archives and checks finite weights, frozenBN and INT8 export. No formal120k launch, new development/final consumption or promotion. v30 and Mac handoff unchanged.

### Fresh-noise full schedule gate and formal protocol

Authenticated8064 original PCM files and2016 voice-speed groups. Replayed all120000 draws against the authenticated completed historical trace, not merely first50: source order/endpoints exact; schedule SHA6d1e5aa342fd387877355be27bed4e30a08a69036f00594e6d41ae96a5632532. All endpoints retain full source plus960-sample RIR and legal buffer geometry. Derived960000 fresh phrase-noise seeds; seed schedule SHAab5e17c7ca61e9ca1b70507cd12ff3ace702d5d70b7e2d4e86be675dc1664798. This is source/hash/geometry/schedule evidence, not exhaustive waveform rendering. Formal commands and postprocess now preserve the historical fixed138 exploratory gates, two fresh120k arms, full matched input audit and no automatic retries. New-source development and independent acceptance must be separately preregistered before their scores; historical survival alone cannot promote. v30 remains frozen.

### Fresh-noise formal fixed arm live; actual first50 authenticated

Original sequence session65524 remains live (not restarted). Fixed arm observed423 consecutive loss records at this audit; per_draw starts only after fixed completes. New standalone read-only prefix audit authenticates actual first50 source order, endpoints, original windows, main tensor hashes and paired tensor hashes against independent offline archives. Saved actual network inputs1/2 match exactly; initialization inventory equals fresh two-step smoke; all50 loss rows finite and consecutive. All271 prelaunch execution files rehashed unchanged during the live run. Prefix receipt SHA fields bind observed records; this does not establish later batches, full completion, model quality or acceptance. No new development/final access. Existing v30/Mac handoff unchanged. Formal fixed/per_draw120k sequence and historical-only postprocess remain pending.

### Future synthetic source lineage review, metadata only

During the unchanged live formal run, inspected three previously unmentioned Piper source cards at immutable rhasspy/piper-voices revision2f358b6a560e3a913ebf509b2f3344f20b367f94. Original card bytes/hashes preserved. Jenny Dioco and Hi-Fi Captain female explicitly derive from hifi_tts_92/Cori Samuel, already represented in training; a new target-corpus name cannot establish pretraining independence. Kristin claims scratch training on~11.5h LibriVox, but speaker overlap with existing LibriTTS/LibriSpeech sources remains unresolved. No source is reserved or certified independent on this evidence. No model/audio downloaded, generation or classifier scoring performed; no changes to training, consumed-development roles or excluded final sets. This records a concrete lineage constraint for future independent development selection, not a new quality result.

### Fresh-noise fixed formal first5000 float checkpoint

Original sequence65524 continues beyond5000 updates without restart. Preserved original progress JSONL record byte-identically (3915B SHA26957a4f6b5400ba783e9c3f341094168125f382564535f21823148a179984df). Training-internal float selection at.99: macro.4125, max-source false.00875, ambient0/1.2048112673611107h;3 training-float eligible grid points. Zero-event one-sided95 FAR upper2.486474317/h, not<=.5/h. This is reused float selection only, not INT8 or independent acceptance, no cross-arm advantage claim. Per_draw arm pending; fixed120k continues. v30 remains frozen.

### Fixed formal10000 float checkpoint, zero eligible points

Original10000 JSONL preserved byte-identically (4048B SHA1ce14230c6e75d0af18e265ffa9ef7f9b14980e02dfaee4375fc2a1aa1b81e9d). All11 grid points fail original training selection gate; returned fallback.994 has macro.38515625/max-source false.010416666666666666, background0/1.2048112673611107h (upper95 2.486474317/h). This is not a gate pass or replacement of the better5000 selected checkpoint. Original session65524 continues beyond10000, no restart/pinned changes/newdev/final access; per_draw pending. No quality improvement or acceptance claim; v30 stays frozen.

### Fixed formal15000 float checkpoint rejected by training gate

Original JSONL preserved byte-identically (4176B SHAbf8199c9b04e92960297af3e46b279583bf1ad7b91f0a52a984c5d1909a2a0f5). Zero eligible11-point training float thresholds. Fallback.994 macro.5427083333 but max-source false.0328125 and1event/1.204811h (.8300055/h), failing both training selection limits. Greater recall does not establish quality gain; fallback not pass. Original session65524 remains live beyond15000; no restart/config changes/newdev/final access. v30 frozen, per_draw pending.

### Fixed formal20000 float checkpoint

Original record preserved byte-identically (3939B SHAe0b7cebf621d3d0cc96dfd44dc9a28a5e3b927d439c73a9cee1fd17dc3dd45b9). Eleven-point training grid0eligible; fallback.994 macro.4864583333/max-source false.0225, ambient0/1.204811h (upper95 2.486474317/h). Fallback not pass. Original session65524 remains live beyond20000, no restart/config changes/newdev/final access; per_draw pending, v30 frozen.

### Fixed formal25000 float checkpoint rejected

Original newline-complete record preserved byte-identically (4132B SHAaebeb9b332fdf87f766912ac690fe689070b516e45444dbee0b5e0315a3797e0). Eleven-point training grid0eligible; returned fallback.992 macro.5325520833/max-source false.0375, ambient2/1.2048112673611107h (1.6600110359/h). Both training selection limits fail; fallback not pass. Original fixed trainer PID555098 observed live at25000, no restart/config changes/newdev/final access; per_draw pending. Reused float selection only, not INT8 or independent acceptance; v30 frozen.

### Fixed formal30000 float checkpoint

Original newline-complete record preserved byte-identically (4084B SHA17598c09b7386931d51b4a5337e21de5bc0ff917375034a1c337a70dc2d8a0df). Eleven-point training grid0eligible; returned fallback.994 macro.4765625/max-source false.0197916667, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass, not INT8 or independent acceptance. Original PID555098 confirmed live beyond30000; no restart/config change/newdev/final access, per_draw pending. v30 remains frozen.

### Additional source metadata: distinct Jenny checkpoint claim

Preserved author TTS index and pinned HFC male card with hashes in future-synthetic-source-lineage-review-v2. Author direct Jenny is described as scratch287epochs, unlike previously reviewed Cori-finetuned Piper Jenny; same voice name does not identify checkpoint bytes or lineage. Model/config identity, license and physical-speaker overlap remain unverified; no reservation or model/audio download/scoring. HFC male explicitly derives from existing Lessac. John/Bryce have upstream-base limitations; Norman/ManyVoice LibriVox physical-speaker overlap unresolved. Existing live experiment unchanged.

### Direct Jenny artifact identity checked without inference

Downloaded author direct model/config only. Model63531379B SHAd33f72ffbabcbbcb673ff6fac06949e8b1a0e56fca0ffd97ae557ac614bab8f1; config7084B SHA39bae6c39dac08347a1232ecfafae9e0d18028c180b23a0a4da18105948a3330. Fixed Piper repository original LFS pointer identifies63221984B SHA62ba183a504cfcd8816ec1b50ee856ed318ccc9dd22fd0bedd7d839c95dfa20e. Export bytes demonstrably differ; this does not authenticate scratch training or independent lineage. Original config, pointer and retrieval receipt preserved. No inference/audio generation/source reservation. Live formal run unchanged.

### Jenny primary dataset metadata and license preserved

Pinned dioco-group/jenny-tts-dataset README atad714a8d15fa010b4ccc024df678822695238070 and original Kaggle mirror metadata with byte hashes. Primary22419 raw prompts differs from mirror20977 RNNoise/VAD/SOX processed prompts; inspected author model metadata lacks exact training manifest. Primary license supplies Jenny/Dioco interface attribution and commercial permission. This is documentary evidence, not proof of model training lineage or physical-speaker independence. No audio generation/inference or source reservation; formal training unchanged.

### Fixed formal35000 float checkpoint rejected

Original record preserved byte-identically (4201B SHA45aa7736037c1a7752f2ac8db17040bd777d9826e5d345b876c8eb827e9b9043). Eleven-point training grid0eligible; fallback.994 macro.471875/max-source false.0166666667, ambient1/1.2048112674h (.830005518/h). Both training selection limits fail. Fallback not pass, no INT8/independent acceptance/cross-arm claim. Original trainer remained live through35000; unchanged run continues to120k, per_draw pending. v30 frozen.

### Fixed formal40000 float checkpoint rejected

Original record preserved byte-identically (4117B SHAa1ea4956c72dd508fbc9600dd72246cd15cdf0a2cd09c6d856a833701e8a3cae). Eleven-point training grid0eligible; fallback.994 macro.5497395833/max-source false.028125, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused floating checkpoint only, no INT8/independent acceptance/cross-arm claim. Original trainer continues unchanged to120k before per_draw starts; v30 frozen.

### Fixed formal45000 float checkpoint rejected

Original record preserved byte-identically (3986B SHA67ec4b0f3b1432d820a001883a610eb3b7f4c1174cf31dc5ad6aa95f71aa5941). Eleven-point training grid0eligible; fallback.994 macro.4526041667/max-source false.0208333333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal50000 float checkpoint rejected

Original record preserved byte-identically (3936B SHA606b5c4949f40ea7bb53bcdd8adb82bf2bb6f5f417a2ed35214e6ca2c3b0074b). Eleven-point training grid0eligible; fallback.994 macro.5005208333/max-source false.0260416667, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal55000 float checkpoint rejected

Original record preserved byte-identically (4190B SHA7a4e1e630a96561e89536c2bdf9a945053aeaa90ec6d066cf89abed735cd3a1a). Eleven-point training grid0eligible; fallback.994 macro.4625/max-source false.015625, ambient1/1.2048112674h (.830005518/h). Both training selection limits fail. Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts; v30 frozen.

### Fixed formal60000 float checkpoint rejected

Original record preserved byte-identically (3973B SHA58663e27075013e952e8b781f56dbbe9a1138568f881c3f86bd54b3285952ebd). Eleven-point training grid0eligible; returned fallback.99 macro.4630208333/max-source false.0109375, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Fixed arm halfway through frozen120k run, per_draw not started; original process continues, v30 frozen.

### Fixed formal65000 float checkpoint rejected

Original record preserved byte-identically (4065B SHA7fba13c0896c1128ad288e30859568bf07931b134ad04188fa25aa92c4d8d48e). Eleven-point training grid0eligible; fallback.992 macro.4989583333/max-source false.0208333333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal70000 float checkpoint rejected

Original record preserved byte-identically (4037B SHA9df2c3a2eb0c5cdc0074d1ef63e47eeb1ac8379ef5db945081d254d8a976015d). Eleven-point training grid0eligible; fallback.994 macro.4841145833/max-source false.0229166667, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal75000 float checkpoint rejected

Original record preserved byte-identically (4005B SHAca34c85be72a55a187985f45ef125e29f51d81c33279b8abdc0f67b0c95ba171). Eleven-point training grid0eligible; fallback.994 macro.42421875/max-source false.0119791667, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal80000 float checkpoint rejected

Original record preserved byte-identically (3971B SHA5cdea6f71784496826806a5b4c58fdfd6265c45cfa85251368dfd4f47c4181f7). Eleven-point training grid0eligible; fallback.994 macro.4854166667/max-source false.0270833333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal85000 float checkpoint rejected

Original record preserved byte-identically (3990B SHA9d02df96323b7a57692a0fb97b5efcfdd52ee5088941302f815def2222d14841). Eleven-point training grid0eligible; fallback.992 macro.4671875/max-source false.0208333333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal90000 float checkpoint rejected

Original record preserved byte-identically (4051B SHA7abb915a718d1ba3daff91c478c5e81516413ce25d751f2cf2e952f5cdfdeece). Eleven-point training grid0eligible; fallback.994 macro.4734375/max-source false.0302083333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Fixed arm completed75% of frozen120k schedule, original trainer continues unchanged before per_draw starts. v30 frozen.

### Fixed formal95000 float checkpoint rejected

Original record preserved byte-identically (4117B SHAe155f94988bb4e3b6491bf8d8abaacd99308fa76d9b2a507463b412a80f5e8d2). Eleven-point training grid0eligible; fallback.992 macro.4924479167/max-source false.025, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Fixed formal100000 float checkpoint rejected

Original record preserved byte-identically (3978B SHA269789b6b98556e2d610dd92fea9f0d2e1e008dbdcdbfe679b969efc1aa42b13). Eleven-point training grid0eligible; fallback.994 macro.4848958333/max-source false.0239583333, ambient0/1.2048112674h (one-sided95 upper2.486474317/h). Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw starts. v30 frozen.

### Mac original main-input dependency handoff

Published gpu-mac-main-input-handoff-v1.tar.gz to the existing research Release in response to Issue13. Original FullSourceBatch training.npz/report.json and all reachable original main-pack JSON metadata included; 1509-file relative-path/size/SHA256 inventory covers JSON/BIN/NPY/NPZ through inherited symlinks. Arrays other than FullSourceBatch are inventoried, not bundled. Anonymous redownload authenticated archive and every member. Archive SHA256 a482b6181fa1765a94506cc5d2da83474bb2956a8ccaa084c97d2bd2d45bb63e. No input regeneration, formal pinned-file mutation, trainer restart or probe-success claim. Mac can compare array hashes and request only missing originals. Existing v30 handoff unchanged.

### Actual training-source exposure inventory for future reservations

Read-only audit of authenticated formal input metadata: main training145980 records (Piper120000, Kokoro23000, Qwen2980); British included1002 records including Cori36/Alba36/Kokoro930; factorized40320 included records; US pinned Lessac/Ryan/LJSpeech. Cori/ Lessac ancestry excludes reviewed Piper Jenny/HFC checkpoints from claims of independent pretraining. Direct-author Jenny remains a potential source with documented scratch claim, not certified independence: Kokoro/Qwen physical-speaker and pretraining inventories are incomplete. No new audio/scoring/reservation; this audit separates known overlap from unknown ancestry rather than treating absent names as proof. Original formal trainer continues unchanged.

### Fixed formal105000 float checkpoint

Original newline-complete record preserved byte-identically (4029B SHAf058ee64ccf9238301296a7654955771aceff3565977617fe749daa11635a3d0). Eleven-point training grid 0 eligible; returned threshold 0.992, macro recall 0.51953125, max-source false 0.029166666666666667. Background 0/1.2048112673611107h; zero-event one-sided95 upper 2.4864743173555484/h. Fallback is not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before per_draw. v30 frozen.

### Future source physical-speaker identity leads

Original LibriSpeech SPEAKERS.TXT and Piper source config authenticated and cross-referenced. Kristin LeMoine14 and Kristin Hughes28 appear in the Piper source mapping; author Kristin source card lacks surname/utterance manifest, so a same-first-name match is only an overlap lead, not identity evidence. Other Jenny/Norman/Cori name matches retained with IDs/subsets. No synthesis, inference, reservation or existing final-set access. Continue explicit lineage review without treating absent names or corpus labels as independence proof.

### Read-only checkpoint collection sidecar

Added standalone unpinned collect_per_draw_noise_checkpoints_v1.py to preserve newline-complete original fixed/per_draw validation records with SHA256/byte receipts as they arrive. Existing originals/receipts are authenticated and left untouched. Eleven-point training eligibility only; zero-event upper field null for nonzero counts; never claims complete training or promotion. Checked original5000 record eligibility3, exact bytes, partial-line exclusion, idempotence and invalid-step rejection in a temporary sandbox. One-shot verified existing records through105000. Started one read-only30s watcher (log .local/wakeword/per-draw-noise-checkpoint-collector-v1.log); no trainer restart, configuration/input/pinned-file change or automatic model selection. Formal original sequence remains authoritative.

### Fixed formal110000 float checkpoint rejected

Read-only collector preserved original record (3964B SHA4316fffe42498e443a68049e228184ccecd767eeafd4a8c0b9d73a458b7d632b); independently matched newline-complete source bytes and recomputed11-point eligibility. Zero eligible; fallback 0.994, macro recall 0.525, max-source false 0.030208333333333334. Background 0/1.2048112673611107h; zero-event one-sided95 upper 2.4864743173555484/h. Fallback not pass; reused float validation only, no INT8/independent acceptance/cross-arm claim. Original fixed trainer continues unchanged to120k before automatic per_draw. v30 frozen.

### Fixed formal115000 float checkpoint rejected

Collector preserved original newline-complete record (4091B SHA5307f17e2d1d6564ef97f55195816a0da847f17ff8122a3dc1652bad68ca4ea3); independently matched source bytes and recomputed11-point eligibility. Zero eligible; fallback 0.994, macro recall 0.5604166666666667, max-source false 0.041666666666666664. Background 0/1.2048112673611107h; zero-event one-sided95 upper 2.4864743173555484/h. Fallback not pass; reused float checkpoint only, no INT8/independent acceptance/cross-arm claim. Original trainer continues to120k before automatic per_draw. v30 frozen.

### Fixed120k complete; per_draw automatically started and first50 authenticated

Final original float record3979B SHA58669e891a33e09ab599f41d6ed8c8e72c48275486cb30d7edb9a34ccae361c6:11-point grid0eligible; fallback.994 macro.53828125/max-source false.034375, background0/1.2048112674h, upper95 2.486474317/h. Original training.json preserved byte-identically. Fixed completed120000 ordered finite loss records and24 validations; trace/loss/initialization hashes match original report. Original selection retains5000 weights; export54440B SHA83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca. Export contract finite saved weights, frozenBN and INT8[1,3,40]→UINT8[1,1] passed; model hash equals prior independent export, no quality/promotion inference.

Original sequence launched per_draw automatically (PID631432); no restart/configuration change. Its first50 source draws, original windows, main hashes, paired frontend hashes and saved actual network inputs1/2 match authenticated preflight; first50 losses consecutive finite. Fresh-smoke initialization and cross-arm actual initialization inventory identical. Rehashed all271 pinned execution files unchanged. Per_draw full120k and cross-arm full schedule audit/historical postprocess remain pending. No freshdev/final scoring; v30 frozen.

### Direct author Kristin/Norman config evidence

Archived original author configs with URL/bytes/SHA256. Both say dataset=training, one speaker and empty speaker_id_map; neither supplies reader surname/ID, utterance manifest or initial checkpoint lineage. They do not resolve name-collision leads. Re-reading the same metadata cannot establish physical independence; retain limits for any future reservation instead of treating missing IDs as proof. No model/audio/inference or newdev/final scoring. Per_draw original trainer continues.

### Per_draw formal5000 original float validation

Read-only collector preserved complete original record (4026B SHA67f0ff2b5ba3040554361e6a108e8cd3b0d785d8bc95f3d155828f65575ca5f6); independently authenticated exact source bytes and all11 grid points. Three training-eligible thresholds; returned threshold 0.99, macro recall 0.2916666666666667, max-source false 0.009895833333333333. Background 0/1.2048112673611107h, zero-event one-sided95 upper 2.4864743173555484/h. Training eligibility only, not independent FAR acceptance or INT8 quality; no cross-arm superiority/promotion claim. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal10000 float checkpoint rejected

Original complete record3974B SHA5a1fbd950947b5d36f2c188b8cfc0282267b65791a36b0ac636a7348902e1610 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.409375/max-source false.0213541667; background0/1.2048112674h, one-sided95 upper2.486474317/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal15000 float checkpoint rejected

Original complete record4048B SHA5c57401c141cb8372fef2d12919bb5233302ad8e0e2479045ce42b79a60c8eb8 preserved; independently authenticated exact source bytes and11-point eligibility. Zero eligible; fallback.994 macro.49296875/max-source false.0364583333; background0/1.2048112674h, one-sided95 upper2.486474317/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal20000 float checkpoint rejected

Original newline-complete record4018B SHA7ff23385a5de4f21d4ebc74575629fc038df21250cbe02ed20a010c383f8cf88 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback0.994 macro0.48749999999999993/max-source false0.025; background0/1.2048112673611107h, zero-event one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal25000 float checkpoint rejected

Original complete record4040B SHAa99abd39d08ee549ef0cd85f9e5fb82a04daaa51b28279edc946a5eabfaf12d8 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback0.994 macro0.3567708333333333/max-source false0.0140625; background0/1.2048112673611107h, zero-event one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal30000 float checkpoint rejected

Original complete record4045B SHA1f06aae6f5a491615bdcb2e8c916a5a7e62ea1bf3be9b3e6941dd1f3791ff548 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback0.994 macro0.4888020833333333/max-source false0.013541666666666667; background0/1.2048112673611107h, zero-event one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal35000 float checkpoint rejected

Original complete record3965B SHAc89a2ba0a586f535356810e32ff119bcc098e5fafc76f9dfd5dcafda1045c676 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5354166667/max-source false.0239583333; background0/1.2048112674h, one-sided95 upper2.486474317/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal40000 float checkpoint rejected

Original complete record4058B SHA1cc19b6a42a16787ffe3e563f12c3c3a8819ab81a17239a92d55cbbbd825f819 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback0.994 macro0.48906249999999996/max-source false0.0234375; background0/1.2048112673611107h, zero-event one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only; no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal40000 loss-prefix integrity

Read all first40000 original newline-complete loss records: exact consecutive steps1..40000, every numeric field finite and all losses/norms nonnegative. Prefix9368396B SHA80ff0fae1c69030e9562e15c7c2b5785d8b96db8607677c20bc32c18b7302dba; receipt archived. Original PID631432 independently observed alive beyond41000; collector622620 and sequence555095 alive. No restart or recipe change. This is prefix numerical integrity, not full-run/quality acceptance; v30 frozen.

### Per_draw formal45000 float checkpoint rejected

Original complete record3956B SHA4bf385a557f9b3641fc6620d3435dcfa2a508f8cd2f91e8612a250bb094ff3f7 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.54140625/max-source false.0229166667; background0/1.2048112674h, one-sided95 upper2.486474317/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Prospective author Kristin/Norman original model preparation

Downloaded original author ONNX exports (each63531379B), pinned model/config hashes and statically checked ONNX schemas; both pass checker. No inference/audio, source reservation or wakeword scoring. Inventory preserves actual model inputs/outputs and stochastic graph operators. Physical identity/unknown pretraining limits from original config review remain unresolved; downloaded bytes do not establish independence. This prepares reproducible prospective generation without changing running training or consuming new development/final data.

### Prospective author synthesis input preflight

Added standalone non-inference preflight for pinned Kristin/Norman exports/configs: Hey Nicks, Hi E S P, Hey Nick, Hey Next, Hey Mix, Hey Nexus. All12 inputs map completely through original eSpeak/config phoneme tables; saved actual phonemes and token IDs, package version and script/inventory hashes. Independent token concatenation recheck passed. No waveform generation, role reservation or scoring; this phrase list is preparation, not a frozen new development/acceptance protocol or acoustic truth. Existing VCTK/pinned runtime files unchanged.

### Per_draw formal50000 float checkpoint rejected

Original complete record3979B SHA9a0538b51a6d9a7ab9a119a156665ec550f18875b1a97805387adbdb64b7d221 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5729166667/max-source false.0442708333; background0/1.2048112674h, one-sided95 upper2.486474317/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal55000 float checkpoint rejected

Original complete record4210B SHA2b9217e88eb57876b7525de9869b0fed48dbcd219ac0ca6a4706a604443c0617 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.50390625/max-source false.0192708333. Background1/1.2048112674h, point FAR.830005518/h; exact Poisson one-sided95 upper 3.937433726679058/h. Receipt zero-event-only upper is correctly null. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal60000 float checkpoint rejected

Original complete record4103B SHA54e4a6147b4607e63a22b36a75373d56dd50eb8ac3630421c4177d819f185c9e preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5890625/max-source false.059375. Background1/1.2048112674h, point FAR.830005518/h, exact Poisson one-sided95 upper3.937433726679058/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw halfway through frozen120k schedule, continues unchanged; v30 frozen.

### Per_draw formal65000 float checkpoint rejected

Original complete record4214B SHA131416719dc12f112da8f28a1e9b4d51c3020a802f526c36d13e159edfcafe13 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5677083333/max-source false.0364583333. Background1/1.2048112674h, point FAR.830005518/h, exact Poisson one-sided95 upper3.937433726679058/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal70000 float checkpoint rejected

Original complete record4146B SHA26015efa7b6ca15476a6819f8d0902e34f397b83ce05798c0c3a4c0a3874d605 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5328125/max-source false.0338541667. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal75000 float checkpoint rejected

Original complete record4134B SHA869d1e911bf120898bd2fa366e716625affdf4d5584824185f7ffe2fda2a9403 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5528645833/max-source false.030625. Background1/1.2048112674h, point FAR.830005518/h, exact Poisson one-sided95 upper3.937433726679058/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal80000 float checkpoint rejected

Original complete record4044B SHAf130624dc3fa3b38d5ad7769d9d117efaab051ec7098400e1583aa8cf459739b preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.99 macro.4606770833/max-source false.0145833333. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal85000 float checkpoint rejected

Original complete record4165B SHA6b83eb8c61c8f049038fb451fdc14cec97bfd2149e4dc041a858cd62919e342d preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5354166667/max-source false.0317708333. Background1/1.2048112674h, point FAR.830005518/h, exact Poisson one-sided95 upper3.937433726679058/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal90000 float checkpoint rejected

Original complete record4147B SHA7f6aacfb9ab3aa07fe19f8212b5eae4489452fffaec8702eb55e2369314a5e18 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5458333333/max-source false.0354166667. Background1/1.2048112674h, point FAR.830005518/h, exact Poisson one-sided95 upper3.937433726679058/h. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal95000 float checkpoint rejected

Original complete record4163B SHA3861ef3ad9d072ea86be3e4a8b502e41fc30c0f531e3c6fc0a1751ec1ca32dc7 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5518229167/max-source false.0338541667. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal100000 float checkpoint rejected

Original complete record4002B SHA096be5797abc45b9eef32004c4f37c6dc151ebad02232570ade38390b715ba53 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5486979167/max-source false.0375. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal105000 float checkpoint rejected

Original complete record4143B SHA932f13b90d827bac44d450a82dc3117f36a14fc21db87128bdc9207edd3c3f46 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5486979167/max-source false.0385416667. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal110000 float checkpoint rejected

Original complete record4093B SHA2db4febbac85a8b165656a162ebcfe6b1412b30a6497a407192a8950059b480b preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.5440104167/max-source false.0322916667. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal115000 float checkpoint rejected

Original complete record4018B SHAa8a8da62a916cdd509f74af0a7c50f278dc1eb45e12f66f71ccf3727ac2f869a preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.992 macro.5328125/max-source false.03125. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Reused float validation only, no INT8/independent acceptance/cross-arm conclusion. Original per_draw continues unchanged to120k; v30 frozen.

### Per_draw formal120000 float checkpoint rejected

Original complete record3995B SHA367245dedd6e733e5bea846b9712a527d283282f0302b2067bf280302cb28a44 preserved and independently matched source bytes/11-point eligibility. Zero eligible; fallback.994 macro.55234375/max-source false.0359375. Background0/1.2048112674h, one-sided95 upper2.4864743173555484/h independently recomputed. Fallback not pass and does not replace earlier eligible5000 checkpoint. Original trainer exited after export; sequence555095 observed running fixed historical postprocess PID712028 after automatic cross-arm audit. Full export and cross-arm reports still require independent review. Reused float validation only, no INT8 quality/independent acceptance/cross-arm superiority conclusion; v30 frozen.

### Per_draw formal completion and full actual-input control audit

Independently authenticated both complete120000-record sampling/loss traces and actual initialization hashes. Every loss step consecutive, all numeric losses/norms finite and nonnegative. Full ordered source draws/endpoints/original window hashes/main tensor hashes equal across arms; all120000 paired network batch hashes differ as intended. Original initialization inventories byte-identical; schedule hash matches preregistered support. Preserved original cross-arm audit.

Per_draw training original613489B SHAdb923fd6e60f71c46b203647d8639862e38652a910062b980aea513b952b48a7 archived unchanged. All24 validations present, original selection5000 retained. INT8 model54440B SHA0d18c3f0ed1f96134440d54beb365f55199de1707d2040be824b0d603c201c2d; saved weights finite, frozenBN unchanged, INT8[1,3,40] to UINT8[1,1] export contract verified. Automatic sequence running fixed historical postprocess; per_draw follows without restart. This verifies run/control/export integrity only; historical scoring and fresh independent quality gates pending, no superiority/promotion claim; v30 frozen.

### Fixed formal historical INT8 regression archived

Original historical postprocess finished. All467 synthetic/background saved score hashes authenticated; reran original138-point calibration from saved scores to fresh temporary output, exact JSON equality. Five historical eligible points; selected.98828125 macro.5791666667, Kokoro10/16 and VCTK32/60 targets, zero false clips across64+240 confusions, zero pre-generation events; background0/8.1736727257h upper95.3665099367/h. Original reports archived unchanged, including original role fields that refer to their earlier development role; all are now consumed historical diagnostics. Fixed model byte-identical to prior independent matched-noise export, reproducibility only. Original sequence now running per_draw historical postprocess PID713321; no restart/promotion/new development/final scoring. V30 frozen.

### Per_draw formal historical INT8 rejected; matched experiment complete

Original sequence and both historical postprocesses finished. Per_draw all467 saved synthetic/background score hashes verified; original138-point calibration independently recomputed to fresh output and exact JSON matched. Zero eligible points, no selected threshold. At common fixed-selected.98828125: per_draw Kokoro5/16, VCTK25/60, macro.3645833333,3/304 confusing clips versus fixed10/16,32/60,macro.5791666667,0/304. Both0/8.1736727257h background, upper95.3665099367/h. At.99609375 per_draw4/16,21/60 with1 false clip.

Across all138 same-threshold grid points per_draw macro lower;14 points have more false clips. Both original selected_step5000. Thus this one matched seed/selection experiment does not support per-draw noise improvement; reject treatment, retain v30. Does not prove universal causal dominance or qualify fixed for independent acceptance (fixed reproduces an earlier candidate already rejected on consumed SLR83). No fresh dev/final consumed; next work requires a new justified hypothesis and prospectively frozen unused-source roles/quality/threshold/acceptance protocol. Original reports and grid comparison archived unchanged.

### Consumed historical source/phrase diagnosis

Added reproducible saved-score-only paired diagnostic at unchanged fixed-selected threshold.98828125; all380 score hashes and exact paired PCM records authenticated, totals independently matched archived grid. Kokoro target regressions: am_liam.85 and all4 bm_george speeds; no target gains. VCTK11 target losses/4 gains across multiple speakers, net32 to25. Three new VCTK false clips are Hey Mix p334.9/p2521.0 and Hey Nick p2991.1. No Hey Next/Hi ESP false clips here. This supports investigating cross-source target/near-word separation rather than another noise seed or single-voice patch; it does not establish unique cause, acoustic truth or justify mining these consumed clips into training. No inference, new source reservation, threshold tuning or training. Next intervention still requires prospective training-only coverage and independent dev/acceptance protocol.

### Training-only phrase group metadata coverage

Authenticated source reports and inventoried included pack_index records only. Factorized672voices/40320variants has10080 complete voice-speed-condition groups, four phrases10080each. British22voices/1002variants has274 groups,53 missing at least one phrase; counts target258/Nick273/Next224/Mix247. All incomplete groups are Kokoro voices. Original British report includes context-excluded variants (e.g. af_bella.8 target29200samples cannot fit24160sample support). This is named pack coverage, not proof of effective full-training absence: inherited main/full-source/other packs may compensate. Next audit must check those overlapping inputs and actual sampling before deciding whether to repair coverage; do not repeat generic balance/temporal changes already failed. No dev/final/audio/model scoring/new training.


### 2026-11-13 — Actual CompleteCoverage recovery integration

Executed Mac Issue #13 request 6846855576 / 6839952933 using original full GPU inputs, isolated base 4b60a882 and Mac patch/tools 5c92e9b. Original, patched/default-off, and patched/enabled each performed exactly two actual CPU optimizer updates with original 120000/5000 schedule; no validation/export. Mac comparator passed: default-off exactly matches original including final variable bytes; initialization equal; enabled changes exactly slots16..23/56..67 in both steps, saved draws reach real optimizer; other108/main and all pair40 inputs, labels, weights, permutation and captured RNG exact. BN unchanged, finite updates, enabled final weights differ. Only integration evidence, no full-training/quality/acceptance claim or new formal run.

Full three captures/configs/base/patched/tools/manifest delivered as gpu-complete-recovery-integration-v1.tar.gz in existing research Release:8397595 bytes, SHA256 5116044fd41a89caf63c83cce5cf24457c315482e5dafdbbcd53ce2dba0357e1. Anonymous download and all438 manifest payload members verified. Results:4090-complete-recovery-integration-v1.json and gpu-complete-recovery-integration-v1-{manifest,download}.json. v30 remains frozen; no new dev/final accessed. Coordination remains Issue #13.


### 2026-11-13 — Preregistered erased-draw recovery experiment

Previous turn was progress: CPU three-way actual two-update integration and full Mac handoff20735bc. Followed with RTX4090 original/default-off/recovered two-update captures using identical original120000/5000 schedule; unmodified Mac comparator passed all checks again. Default-off final weights/loss/input/state equal original exactly; recovery changes precisely20 rows in both updates and preserves108/main,40/pair and RNG; finite updates/frozenBN. CUDA reports archived under4090-complete-recovery-cuda-*. Source patch is the exact9-line Mac patch, trainer SHA2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904.

New hypothesis and preregistration:experiments/complete_recovery_formal_protocol_v1.md. Restore erased8positive/12negative CompleteCoverage draws, replacing20 British rows, not changing batch size or sampler RNG. Both fresh formal arms retain fixed paired noise and original recipe; one120k each, sequential, no seed/checkpoint search. This is not a claim about causality of British53group exclusions. Existing failed per_draw treatment is not repeated.284execution pins verified including inherited inputs/runtime and explicit new trainer/commands/protocol. GPU two-update evidence only, no current quality or acceptance claim. New development/independent synthetic quality and acceptance protocol remains required before any new-source scoring or promotion. All consumed/final exclusions and v30 freeze remain.


Formal sequence launched after cce5cd4 push, session18719; control trainer PID719837, recovered waits sequentially. Verified live via process state and growing consecutive loss trace (>108updates observed). First50 full trace exactly matches authenticated prior fixed trace, first2 actual main/pair arrays exactly match fresh CUDA disabled probe, actual initialization inventory equals prior fixed; all284 pins reverified during execution. Receipt4090-complete-recovery-control-formal-first50-v1.json. No timeout/restart.

CUDA complete three-probe archive gpu-complete-recovery-cuda-integration-v1.tar.gz:7827074 bytes SHA25675653fbd4266f5cbd0f543cb61ad2471a80a0dd9474a42759663ad0fb79ac0b4; anonymous download and51payload member checks passed. Release URL recorded in gpu-complete-recovery-cuda-integration-v1-download.json. These are execution proofs only; formal quality pending, goal remains active.


### 2026-11-13 — Native output-to-PCM quality gate

Implemented experiments/synthetic_output_quality_v1.py with policy document. The gate verifies raw/WAV hashes, native float32 mono finite/nonclipping output, full integer-ratio resampling length and every transported sample within PCM16 quantization tolerance, PCM format/nonzero/duration/RMS/saturation, immutable sample ledger and untrimmed synthetic output reference. No gain/cropping repair or score-based regeneration. Whole saved output integrity does not prove semantic/phonetic completeness or hidden TTS decoder behavior.

Actual audit on30 consumed historicalSLR83 preserved raw outputs:30pass; model/config/source/generator hashes, original runtime versions, planned text/phones/token mapping and unavailable duration records additionally authenticated. No new synthesis/scoring/data filtering/role changes. Six adversarial unit tests passed, including crop with adjusted ledger, gain change, silence/clipping, NaN and hash replacement. Runtime versions/commands/code+result hashes preserved in4090-saved-synthetic-output-quality-execution-v1.json. This closes an implementation gap; new-source role reservation, augmentation tails/background protocol and independent acceptance remain incomplete.

Original formal control PID719837 verified live at05:30, last observed step1866; original session18719 remains the sequence handle. Recovered waits. No execution pins or training files changed this turn, no restart, no checkpoint-quality claim.


### 2026-11-13 — Prospective ManyVoice exposes known heldout overlap

Verified original control PID719837 live; step2274 at06:39. No restart or execution-source modification. While independent-source protocol remains incomplete, retrieved original author mv2.onnx.json (7528B, SHA2561dae6569843412ace557acdc2f2136405219ab75219efb3ddf882af2055a4c1c), not the model or audio. Unlike generic Kristin/Norman configs this provides all16speaker names/IDs. Cori_Samuel is ID0, and Cori is already represented in the pinned British training pack. Thus a new mv2 filename or selecting another speaker ID cannot certify a pretraining-independent source; the shared model has known training-speaker exposure. Other15overlaps are not assumed. Kristin_Hughes ID2 is a lead, not proof of separate solo checkpoint identity. Original config and metadata review preserved; no new role reservation, generation/scoring or final access. This new evidence prevents accidental misreservation; it is not a reason to stop training or demand human listening.


### 2026-11-13 — British exact-source compensation and surviving slot audit

Completed earlier pending compensation check with training-only source reads, no detector inference. Authenticated all424 original British WAVs and all3manifest hashes, computed PCM16LE hashes independently and compared with all canonical CompleteCoverage source/alias PCM hashes.342unique British originals have included variants;106have at least one excluded variant,82have no included variant. Across all424, exact WAV overlap0 and exact PCM overlap0 with CompleteCoverage. This distinguishes same-name semantic coverage from actual audio identity, not a claim that all other corpora lack these voices/phrases.

Using the original authenticated fixed training report replacement slots, control ownership isUS20/factorized20/British40/ambient48 with zero surviving CompleteCoverage rows; recovered isUS20/factorized20/complete20/British20/ambient48. FullSourceBatch reads the same1002included British variants and original main80 phrase rows are overwritten, so neither is an additional copy of the excluded outputs. Current recovery adds20different canonical draws, not those exact excluded British WAVs. Audit script and full106source detail result:4090-british-source-compensation-v1.json. This clarifies the running intervention and closes the exact-byte compensation question, without proving failure causality or quality improvement. Runtime pins unchanged, no new training/filtering.


### 2026-11-13 — Full rendered-presentation integrity gate

Added synthetic_render_quality_v1.py: independently reconstructs all original dry source samples, direct-convolution RIR (contrasts prior FFT renderer), complete960sample tail, full seeded noise, declared SNR, headroom gain, lead/tail lengths, generation/RIR references and quantized PCM. All90 consumed old SLR83quiet/noise25/RIR+noise15 presentations passed exact full-length/PCM quantization checks, source and saved WAV/PCM hashes authenticated. No new synthesis/inference/data role changes.

Six adversarial integration tests passed: original, cropped output with adjusted ledger, erased960tail at unchanged length, shifted boundary, wrong noise seed, altered gain declaration. Code/report/test hashes and commands preserved in4090-saved-synthetic-render-quality-execution-v1.json. This supplements native output gate with fixed-condition rendering checks; it does not prove acoustic phones/semantic completeness, certify new voices, provide new development evidence, or complete acceptance/background reservation.

Original control trainer PID719837 remained live, last observed4696steps; same sequence18719, recovered waits, no restart or current pinned-file changes. First5000checkpoint pending at observation.


Control original5000step float event-validation arrived while preparing this turn:3915byte original line SHA2568d4a6107eee5a83e622511ee7d261582a05bb721fc5902dc741931bfb88228f1 preserved unchanged, receipt added by collect_complete_recovery_checkpoints_v1.py. Selected.99, macro.4125, max-source false rate.00875,3/11training-eligible points; selected/grid exactly equal prior fixed5000record. Background0/1.2048112673611107h has one-sided95upper2.4864743173555484/h, not<=.5. This is reused training-float evidence only, not INT8/independent acceptance or recovery benefit. Original control continues120k; recovered still waits, no restart/promotion.


### 2026-11-13 — Cross-machine source consumption and prospective background roles

Fetched Mac branch at318f5fb322308cbe861035f0ea5c496066cdfd3c after Issue6847221454. Original generation plan, v30 baseline and same-prefix diagnostic reports preserved byte-for-byte with commit/path/SHA receipts. Both Kristin/Norman model AND config hashes exactly match local prospective author inventory. Mac has already exposed these presets through126PCM v30 diagnostics and504continuation traces. They are consumed diagnostic/development sources, never training or relabeled fresh development/independent acceptance. Earlier local not-generated metadata remains historically accurate but cannot override external consumption. No new local detector inference/reproduction claimed.

Pinned mirror tree and partial background role plan before any new audio/score access:VoxConverse test4/5 development, test6/7 independent acceptance; old dev0..3 and original final test0..3 excluded. Exact file SHA/size/source revision frozen. Plan does not authorize scoring: recording-ID/PCM cross-role audits, full-recording continuity and>=6h per role still required. No post-score exposure extension, no duplicate-based silent substitutions. Independent acceptance raw files remain unopened until full positive/quality/threshold/acceptance protocol frozen and development passes. Original mirror/person/transcript caveats remain explicit. New downloader only fetches the two development files and preserves partial failures without auto-retry. No training pins/data changed.


### 2026-11-13 — New development background downloaded; strict annotation audit fails

Original downloader session91345 completed normally:both frozen test4/5shard sizes and SHA match plan09d296b391ac157b3141dda0f36a9fc803852ff97fb8cd2f8f672fee5fe4d41d. No acceptance download or detector score. Original audit v1 reused20ms annotation boundary condition and failed in both shards (sessions7235/11503); original source/failure receipt preserved. First violations:qwepo439.424s audio vs439.5s annotation andgtjow1200.064s vs1200.12s.

Separate diagnostic v2 retains same condition but enumerates every row and explicitly reports background_qualified=false:42recordings,7.902862222222223h full embedded mono PCM16/16k,8recordings with invalid annotation intervals. No trimming, metadata repair or deletion. Independent byte/ID comparison with173consumed dev0..3records:0overlap for recording names, encoded SHA and PCM SHA; within new42also0duplicates. This proves scoped format/duration/known-development disjointness only, not across all training/unopenedfinal or global physical people. Original mirror preprocessing/transcript limitations remain. Full future protocol must specify whole-waveform exposure without annotation-based segment selection before scores; current diagnostic is not a qualified full protocol or FAR evidence. Original training untouched and continues.


### 2026-11-13 — Lossless whole-waveform development package

Pre-score policy/preparer frozen and pushed atb339df1 before execution:full embedded WAVs, exposure samples/16000 only, no diarization-based segmentation/deletion, no target screening. Original20ms annotation audit remains failed8/42; annotations do not participate in the new preparation rule, so no tolerance change or original metadata repair. This is not a full scoring/qualification protocol.

Preparer session52688 completed normally.42original WAVs preserved byte-for-byte,455204864PCM samples=7.902862222222223h,8invalid-annotation records retained. Independent standard-library wave reader (contrasts soundfile/libsnfile) verified every encoded SHA/PCM SHA, frames, monoPCM16/16k and total samples. Original output manifest SHA b6ab85ea987955dd3ab7fb781ddd962114132578c3f10e89c1305b36a35ad6a3 copied to results with verification receipt. Manifest explicitlybackground_qualified=false/scoring_authorized=false; no detector inference/new FAR/acceptance access. Remaining cross-training/final/person and positive-source protocol gaps remain explicit.

Original control PID719837 live at25:25, observed8888steps, recovered waits in sequence18719; no restart or trainingpin changes.


### 2026-11-13 — New background vs actual training-source hashes

Authenticated current main human/ambient index hashes against original completed fixed training metadata. Decoded all1107actual training ambient originals (including45base environment and1062added noise/music), authenticated encoded source hashes and monoPCM16/16k format. No encoded/PCM whole-container matches to new42background recordings. Also compared273available human-stream PCM hashes and28539nested original human source encoded/PCM hashes:0matches. Full1107source details preserved in4090-new-background-training-overlap-v1.json.24legacy human rows lack stream PCM hashes; synthetic phrase packs and transformed/partial/physical-person overlap are outside this audit. Thus all_training_pcm_isolation_proven=false/background_qualified=false remain explicit. No training data/pins changed, no new detector scores/final audio accessed.

Original control10000float checkpoint now preserved unchanged:SHAed22fcdba40e3013736616441a1a3818cb6e602b4205b064c742cd01e061e782; original11point grid0eligible, fallback not a pass and does not replace selected5000. These are reused training-float metrics only. Original PID719837/sequence18719 continues, observed10373steps, recovered waits; no restart or promotion.


### 2026-11-13 — Replayed24legacy training human sources

Located original LibriSpeech dev-clean files for the24training speakers lacking PCM hashes; read only these training IDs, not validation/testing speakers or test-clean/test-other. Using unchanged original frontend source4adbe49b and original sorted-FLAC/3200zero-boundary/continuous-frontend recipe, replayed1626FLACs with2CPU workers. All24file counts, original/stream sample ledgers, frame counts and every saved feature value exactly match actual current training arrays. Report and original24completion log preserved in4090-legacy-training-human-pcm-audit-v1.json / original-v1.jsonl.

Newly reconstructed full-stream and originalFLAC encoded/PCM hashes all have0whole-container matches to new42background recordings. This closes the missing-source replay investigation; hashes are newly derived, not falsely described as historical same-instance PCM hashes. Frontend is noninvertible, so exact feature identity plus source ledgers is strong provenance evidence, not a unique inverse proof of past raw bytes. Re-encoded/cropped/person overlap and unopenedfinal/synthetic-pack scopes remain separate. No data/pin/training changes or detector scoring. Original audit session9770 completed exit0.

Original control PID719837 verified live at32:15, same sequence18719, latest completed checkpoint10000; recovered still waits, no restart/promotion.


### 2026-11-13 — New rdh-lineage prospective models and consumed-source guard

Pinned Piper revision2f358b6a560e3a913ebf509b2f3344f20b367f94 primary cards for9previously unreviewed candidates. Arctic/l2arctic/kusal/sam/reza explicitly derive Lessac; Kathleen/Danny derive Ryan, so do not relabel known training-base exposure as new pretraining independence. Joe/Mike cards instead declare Flemishrdh base (CC0, MRD/SDP, Piper>=1.5); these are prospective candidates, not proven physical/pretraining independent. No role reserved or audio generated.

Downloaded Joe/Mike original model/config at fixed revision, saved original LFS pointers/configs/cards and inventory. Each63221984B:Joe SHAabf4b998e34045cbd00eaf050de2de620f81ded6eb8b85a24b022a1c2b3f08d2;Mike SHA6876866e3dfadc7f044bc5e4fbac16c47b5a1e79eaec04e3bb0c2c6e9da8a857. Both staticONNX checker pass, same integer-phone/length/scales3 input, stochastic RandomNormalLike retained; no model inference/synthesis.12planned phrase phone/token mappings passed against pinned configs using recorded phonemizer version, not acoustic truth. Training runtime not installed/changed. Original model-preparation session72396 exited0.

Implemented check_known_source_consumption_v1.py authenticating copied Mac evidence:Joe/Mike no known consumed-byte conflicts; Kristin/Norman negative control correctly rejected2/2 with exit1. Both reports keep source_qualified=false/synthesis_or_scores_authorized=false. Guard covers current known registry only, not all training/pretraining/person or undisclosed external consumption. Before generation still need complete role/synthetic quality/threshold/acceptance protocol and further source-role checks.

Original control PID719837/sequence18719 verified live at36:31, step12770, latest checkpoint10000. Recovered waits; no restart or trainingpins changed.

### 2026-11-13 — Mac inherited-positive CPU integration handoff

Ran Mac driver unchanged from280c37728cc1823caecb19c4569e745077a0f372 against GPU base43254c0, outside repository to preserve git-apply root semantics. Complete recovery off/on each compared original/default-off/enabled with2 actual CPU optimizer updates per arm (12 total); both comparisons pass. Default-off input/loss/final-variable bytes exactly match original with common initialization. Enabled restores original already-drawn/scaled main slots24..31 into actual optimizer; other120 main rows, all40 pair rows, labels/weights/permutation/RNG remain exact. FrozenBN and original120000/5000 schedule retained; finite enabled updates produce different weights. Captured original source IDs/texts/crops: both steps draw one command-bearing phrase, not a forced command-only intervention. No validation/export/new trained candidate or quality conclusion; current formal trainer unchanged.

Published gpu-inherited-positive-cpu-probe-v1.tar.gz to existing research release:17689908 bytes, SHA256 7d6eb2adc84fad0ebfc2889fa25874a859d835cad0101acc2e07691d2cd9bbd3. Anonymous urllib download independently verifies archive and all458 manifest payload members. Original tools, configs, source snapshots, captures and logs included; manifest/download receipts committed under results. Issue13 is the cross-machine handoff channel.

Pinned rdh upstream primary card declares trained-from-scratch Flemish CC0 parent, improving declared Joe/Mike lineage beyond known English bases. Saved exact card/hash and limitations: declaration is not audited weights/logs/physical identity proof; Joe/Mike share parent. No synthesis, roles or scoring performed.

Original formal control PID719837 remains live at44:54; preserved step15000 original line SHA685d85cf56c261084de23f8c1654005c6991caa5c372e67f5683e13d75fa75d2,0 eligible training-grid points. Recovered still waits; no restart, candidate promotion or training-pin change.

### 2026-11-13 — Prospective Joe/Mike generation reservation

Read Mac Issue13 comment6847804337: authenticated v30 original recipe retained all32 original positives and had command support; later overwrite cannot be asserted as sole v30 failure cause. Actual12-update CPU integration has already been handed off in comment6847803714.

Before new inference, reserve Joe development / Mike acceptance, neither training. Distinct voice/model bytes but sharedrdh parent: explicitly voice/model holdout, not family/physical/pretraining independence. Mac reports no known prior Joe/Mike use, not universal proof. Frozen240-request plan (120pervoice):20texts (4HeyNicks,4HiESP,12commonconfusions),3lengthscales,2draws. Plan SHA d69dbc7486cb974641ae9849e33c965e98998a79ce55c95b8a334d4d4f71c03a pins original inputs/phones/tokens, models/configs, runtime and generator/validator/protocol hashes. Input token mapping and Python compilation pass; no audio yet.

Only Joe native development synthesis/mechanical auditing authorized by this reservation. Save full actual engine tensor and nativefloat32, full rationalresample PCM16; no gain/crop/regeneration, all failures retained. ONNXCPU2threads avoids GPU contention and environment changes. Mike inference, detector scores and acceptance remain prohibited until complete rendering/model/threshold/gate protocol is frozen. This advances actual native generation rather than requiring unknowable phonetic/physical-person certification or human hearing. Original formal trainer and284pins unchanged.

Joe original120-request run finished normally as a process but all120 rows failed the generator's rank3 output assertion: engine actually returned[1,1,1,N]. All exact fullfloat32 tensors were saved before assertion; no PCM or detector score generated. Original records SHA91b310721d53642bea56fe838d57040785681ced5dce5b0b2f17b6c45a1a7e03 and failed summary copied unchanged to results. Preserve this implementation failure. Added separate pre-execution saved-tensor adapter/policy:authenticate all120 ordered originals, permit exactly[1,1,1,N], remove only singleton axes with exact whole sample-byte identity, then apply unchanged transport/quality gate. No re-synthesis, waveform repair, draw replacement or audio selection. Original generator and generation plan remain unmodified; corrected transport does not erase original failure.

### 2026-11-13 — Actual new-source native waveform evidence

Executed saved-tensor adapter only after8ac36bf pushed. All120 Joe requested rows pass unchanged native/transport mechanical gate. All original tensors authenticated and singleton removal sample bytes exact; no further TTS calls. Independent stdlibwave reader plus full original tensor resample independently checked all120 complete sample sequences, monoPCM16/16k and quantization bound. Maximum error3.0517339371288754e-05; duration0.696625..2.7980625s; RMS0.0384703564..0.0685218788. Original120 rank assertion failures remain separately preserved; no discarded/replaced draws. Native mechanical qualification only, not phonetic truth, rendered-pack qualification, detector performance or promotion. Mike remains ungenerated/unscored; no wake-model inference/new threshold selected.

Full actual engine tensors, nativeNPYs, WAVs, original failed/adapter records, source plan and exact tools archived in gpu-joe-development-native-evidence-v1.tar.gz:30241016B SHA bbb934893d761e24df5a21da9ed59454739cc371204b8f237522c0bf164b8b24. Uploaded existing research release, then anonymously downloaded and verified archive plus371payload SHA/size entries. Manifest/download receipt and full audit ledgers committed. Next work: preregister full room/noise rendering and frozen same-PCM streaming references/threshold/acceptance gates before scoring. No requirement for human listening.

All284 original formal execution pins reauthenticated unchanged after synthesis/transport. PID719837 verified live50:46, step17500, selected checkpoint still original policy; recovered waits sequentially. No restart or training/runtime modification.

### 2026-11-13 — Full-source presentation preregistration

Mac independently completed original12-update integration archive verification (Issue13 comment6847863935, sourcef95ad2d), including its own original comparator producing byte-identical comparisons. No rerun needed; no quality/promotion inference.

Pinned all360 Joe presentations before rendering:120originals×quiet/25dBnoise/15dBnoise+finite960sampleRIR. Plan SHA b09550c6804f4c5f0e23933b95501dd8ab4c218bc35a430395894ce7e4ee64db. Exact original sourceSHA, source/runtimes/protocol/validator/renderer, deterministic seeds, full28800lead/16000tail frozen. Prescribed noisy presentation peak gain (not native repair) recorded. FFT renderer independently checked by existing direct-convolution gate; all rows retained, any failure prevents qualification. Whole generated/RIR end references explicitly not phonetic/wake-prefix boundaries, particularly for command containers. No detector scoring or Mike inference authorized. Original training PID719837 remains live51:40/18000steps; no restart.

### 2026-11-13 —360 complete Joe presentations verified and published

4524c4b frozen plan executed once:360/360pass existing direct-convolution reconstruction gate against FFT renderer, full source+960RIRtail retained, no detector inference. Independent stdlibwave validates all WAV/container/PCM SHA/counts and explicit generated/RIR boundaries. Maximum quantization error3.051734301030471e-05; all prescribed gains happen to1.0 (no actual gain reduction). Original120native sources/rank failures remain untouched. Replay input manifest explicitly scoring_authorized=false and whole-generated-output—not phonetic—references; existing official runner needs explicit authenticated new synthetic-reference support, not invented endpoint annotations.

Archive gpu-joe-development-render-evidence-v1.tar.gz:28122937B SHA52a5ba59c6e2c5d38697cc14deba648b478c09d0e3d5f3d673d6b6842c3f8c58; existing release upload followed by anonymous download verifies all367payload members and archive. Full row/summary/replay/manifest/download receipts saved in results. No new source scoring or Mike inference.

Preregistered new-source selection/acceptance decision rules before scoring. Sole candidate=recovered original selected export, no Joe-based arm/checkpoint choice or restore8 fallback. Original138threshold grid/ties retained; candidate requires zero common/cross-target/pre-source positive false activations, FARone-sided95<=0.5/h, macro>=max(.8,new-role frozenv30macro), standalone/command each>=.75, all mechanical/source/streaming gates. Acceptance once only after development passes and point locked, same gates; no additional duration/threshold/audio rescue. OfficialHiESP stays original fixed operating point; samePCM and measuredFAR do not establish matched populationFAR. Includes signed full-generated/RIR latency, misses/early/repeat events and command limitations. Executable/binary pins still required before scores; document itself grants no inference authorization. Original284executionpins reauthenticated unchanged; latest completed trainingcheckpoint15000, original process remains live.

### 2026-11-13 — Official runner authenticates new generation references

Added exact Joe360manifest SHAedf6f7844f654d2a7e22f588873b917381aa8d0957b1e3309a60a0f1973895bc allowlist plus original plan/record hashes, full-source/RIR/sample ledgers, voice/variant/role checks. Joe admission cannot fall back to fabricated endpoint annotations on a changed manifest. HistoricalSLR83/endpoint policy unchanged. No detector/runtime binaries or inference algorithm changed.5policy tests pass:all360 originals, changed bytes with forged annotations,12boundary/source/role mutations,5incomplete/final/plan mutations, historicalSLR83accept/reject. This implements source admission only, not complete new-score authorization; all pending source/execution gates still apply.

End-to-end regression replayed only one already-consumed historicalSLR83quiet fixture through exact existing officialQEMU/ELF/flash. All17audio/event/CRC/chunk/stage/getter/engine fields exactly reproduce cached original:54646samples,107calls,stage104,0events. Initial wholeUARTbyte comparison failed; inspected/preserved both originals and established the ONLY difference is parenthesized ESP-IDF boot clock timestamps. Removing those clock fields produces byte-identical text; audio event timestamps/counters unchanged. Both rawUARTs/individual hashes and mismatch receipt preserved, not described as byte-identical replay. No Joe/Mike inference or new data score.

Preserved original control20000checkpoint3939B SHA1f58b1c65c250b73aaea2bf65053fe7f2bf63eb61173a4b7f2e82873f3adf02b:0training-float eligible, fallback.994notpass;0events/1.204811267h implies upper95=2.486474317/h, not<=.5. Original284executionpins reauthenticated unchanged. Continue original120k×2sequence, no restarts or selection changes. Next actual new-score step still needs known-source/PCM audit coverage plus frozen orchestration and binary receipt; don't spend new data to test a transport wrapper.

### 2026-11-13 — Actual training PCM and historical evidence isolation audit

Compared all120newJoe native originals and42newbackground originals (162unique encoded andPCM objects) to immutable43254c0 historical metadata:1408JSON/JSONL files,475858559bytes, inventories pinned by gitblob+SHA.0matching recordedhashes. This scans saved final metadata only; no final waveform access. Authenticated actual main feature index to completed fixed formal training metadata, independently decoded/authenticated all14598distinct training-originalWAVs referenced by actual training rows:0encoded orPCM matches. TwoCPU workers, no training data/runtime changes. Full ordered training-source ledger preserved withSHA511439f60dabc8280cec93827f28a540a4de111a55baef42ecf804e740c27ac1.

Authenticated additionalBritish/balanced/US/factorized report hashes to original284executionpins:0recordedsource/PCM matches. Supplement authenticates CompleteCoverage sources/records/report to original actualtraining hashes, with5188/33722/7recordedhashes respectively and0matches. Also actualtraining human57351hashvalues, ambient2214originalencoded/PCM hashes and legacy24replay3300hashvalues:0matches. Known cross-machine consumed-model registry has0Joe/Mike model-byte conflicts. Supplementary executable reproduced all6original check structures exactly. Existing human/ambient provenance limitations retained; no claim of excluding all hidden/partial/transformed/physical/pretraining overlap. Joe/Mike sharedrdh parent disclosed.

Pinned concrete baseline-only orchestration and executionreceipt before any new scores:fe307466b5e08c53ef88d5eaaea1279f4df99ba13f1e1ea9ef7bc3f9b9b5bfb8. Explicitly authorizes fixed officialHiESP/v30 on all360Joe development clips only, not candidate or acceptance. All prior source/mechanical/runtime receipts authenticated, original360manifest exact. Official4workers/completeCRC/chunks/stage/getter evidence; no retries or incomplete combined summary. Then originalv30INT8 full-waveform streaming, fresh reset twice/bit-repeatable scores, expectedtracecounts/ranges, allsignedgenerated/RIR latencies and negative/early/repeatedevents. Fixed operatingpoints only, no thresholdselection/qualitypass/matchedFAR claim. Candidate remains original recovered arm and must wait for authenticated completed export; background evaluation separately pending.

### 2026-11-13 — Actual360 new-source fixed-baseline replays complete

84a6004receipt pushed before execution; session20399 ran original360official and v30full-waveform streaming baselines without retries. All360officialCRC/call/stage/getter/binary checks and all360Nyx tracecount/finite/range/reset-repeatability checks passed. Frozenv30 positive0/72 (both standalone/command0), common0/216/cross-target0/72, no pre-source positiveevents. OfficialHiESP intendedpositive64/72=.8888889, standalone29/36=.8055556/command35/36=.9722222; common18/216, cross-target0/72, no pre-sourcepositiveevents. Full row/rawtrace/event signedgenerated/RIRlatencies saved. These are single-Joe synthetic development fixed-point results, not human quality, matchedFAR, candidate failure or promotion. No candidate/Mike/acceptance score or thresholdselection.

Prepared original42recording background replay manifest before any background scores:455204864samples/7.9028622222h, exact originalWAVs, no gaps/annotationtrimming, everyevent counted, frontend/QEMU finalpadding excluded from exposure. Original failedannotationaudit preserved. Frozen backgroundfixedbaseline executionreceipt f11ad2860d597a335f1c7f04b38f727b6c88eae1a1bd376660763bc915133d0c pins executor, all originalbinaries/source/qualityaudit references and exactnewmanifest; samefixedofficial/v30points, fullcontinuous state per recording, repeatability, no retries/durationextension. This separately authorizes only these newDEVELOPMENT backgroundfixedbaselines after commit/push, never candidate/acceptance. Runtimeversions remain identical; no installs/changes. Original training continues live1:09:38; latest preservedcheckpoint20000, no restart/promotion.

Completed Joe baseline full evidence published:gpu-joe-fixed-baseline-evidence-v1.tar.gz29839545B SHAe2361d3835a192ccbd25567fad33ede42a63126771306b3b5c7566f732fc4160. Anonymous download verified archive and2526payload members (allofficialinputs/UART/reports/logs, actualv30traces, frozenreceipt/tools). Manifest/download receipt preserved. NewJoev30maximumsmoothedpositive score0.80546875, below unchanged.91796875 point; do not retune baseline. All18officialcommon falseactivations are intendedtext"Hi E S B."; this is synthesis-intent metadata, not human phonetic truth. Signedofficialpositive whole-outputend latencies range-1299.1875..175.25ms, median-994.65625ms, strongly command-container-dependent, not wakeword-end/humanlatency superiority. v30has no positiveevents, so positiveeventlatency isnull, not0.

After38e34depush launched frozen full-background baseline session82068/PID759335. Process directly verifiedlive; last11/42officialcompleted,0failures, no retries. All pinned Pythonversions rechecked unchanged. OriginalformalPID719837live1:11:39/last24500steps;284executionpins unchanged after newbaselines, recovered waits. Goal remainsincomplete: finish background, original120k×2/export/postprocessing, recoverednewdevelopment thresholding and gatedindependentacceptance, preserving failures. No candidate/Mike/acceptance inference or productionchange.

### 2026-11-13 — Complete new background baselines and executable candidate gate

Background session82068/PID759335 completed normally exit0, no retries. All42original continuous recordings and455204864originalsamples=7.902862222222223h consumed; all42officialCRC/calls/stage104/getter/engine and v30fulltracecounts/finite/range/resetrepeatability pass. Frozenv30=0events, officialHiESP=0events, bothnominalPoissonone-sided95upper0.3790692776004911/h. Sourcehours exclude frontend/QEMU finalpadding, no gaps/annotationtrimming/rowdropping. Original failedannotationaudit preserved. This meets the specified FAR bound on DEVELOPMENT background only; it cannot rescue frozenv30's0/72Joe positiveactivation or establish candidate/independentacceptance/physicalgeneralization.

All42raw officialinput files independently verified as exact16bytefrozenheader+entirestdlibwave-decoded originalPCM, withinputSHA/size/header/originalPCM provenance recorded. Published gpu-new-background-fixed-baseline-evidence-v1.tar.gz229537B SHAcf298670f15c77f4b005e9dd3391d5050e08b2c4937eafd2d87574be6ec794f8; anonymousdownload checks archive and256payloadmembers. Omitted42duplicate~910MBinputPCM from archive, explicitlyledgered with exacthashes/reconstruction, retainedlocally. Archive includes allactualtraces/UART/reports/receipts/tools; no claim of downloading/verifying omittedbytes as payload. Fulloriginalsources remain pinnedpublicshard/manifest.

Implemented future sole-recovered development scorer+calibrator without executingcandidate. Newfreeze refused untilbothoriginal120k/postprocessors complete, recoveredtraining/exportaudit exact, savedbaseline360/42complete and allsource/mechanical/runtime receipts reauthenticated. Pins originalselectedmodel/weights/calibrator/front-end/detector/versions before anycandidateinference; no arm/checkpoint choice. Scores entire360+42PCM, percontainerstate/resettwiceexact, no officialrerun/thresholdtuning whiletracescreated. Calibrator authenticates402wholetraces and exact138grid, frozen.8/v30macro/standalone/command/zero-confusion/pre-source/FAR gates, originalmacro/FAR/higherthresholdtie, nullfail and noacceptancerescue. Stores allgrid/selected signedendrefs; no phonetictruthclaim. Dependencyversions includingSciPy pinned/checked in bothinference andcalibration.

Six analytic score-trace counterexample tests pass (notaudio/model evidence):populationcounts; pre-sourceevent plus laterhit stillfails; finaltailhitcounts/no invented end+.8cutoff and negativesignedlatency retained; singleconfusion/shortexposure rejection; commandratefailure despite overallmacro>.8/nullnofallback; exact3levelselectionpriority/grid. Actualpremature--write-freeze rejectedexit1 beforecreatingfiles; prerequisitecheck stillmissing4originaltraining/postprocessing artifacts, no candidateoutput/freeze. CPUcodecompilationpass. OriginalformalPID719837live1:18:30,last27000steps;25000originalcheckpoint6436f40b… preserved0trainingeligible.284executionpins unchanged. Goalactive/incomplete, candidate/Mike/acceptance unscored, no modelpromotion/restart.

### 2026-11-13 — Original selected v30 float checkpoint handed to Mac

Located unfulfilled Mac Issue13 request6848030084 (reminded6848064836/6848243687) for originalv30selected70000float artifact, not a rebuild/retrain or substitutev46. Retainedbest.weights.h5 and originalSavedModel are present. Read-onlyCPUauthentication:originaltrainingSHA41d3a232920143d6bfa39b74fc1068f2675615f2df23ed599155b58a085312f9, selected70000/120000, originaltrainer29be79c7e0af59dfc5cddcdc3733a284c4c777ff SHAa9c76aee07f2a61557ed77095609671f546bfbbf418d979732da356b98898c20. AllSavedModel files exactly match historicalee0becbf3d8716d14a977afbe52b2e5e5aa0ecee2026-11-11float diagnostic hashes. RawretainedHDF5361168B SHAb40bf705297a71f208e073897ff517e90f9813337e73a959e27627b2d2668fa9, optimizeriteration70000, all31modelarrays dtype/shape/value multiset exactlyequalhistoricalSavedModel.32optimizerdatasets separately retained; statebuffers separately inventoried. FirstfullHDF5filehash recordednow, not inventedhistoricalhash; repeatedarrays don't prove uniquenamepairing. No forward inference/weightupdate/export/quantization performed. TensorFlow2.21CPUonly,2threads, no trainingruntime change.

Published gpu-v30-original-float-handoff-v1.tar.gz570686B SHA6f19e79f63aedde9e8afb0e3229e9f260179099cfaa27590745856233889f83b; anonymousdownload verifiesarchive+all15payloadmembers. Includes unmodifiedbestweights/SavedModel/frozenINT8/initialization/training/progress, originaltrainer+historicalfloatdiagnosticsource/report, authentication/README/manifest. Onlymodel/researchmetadata/code, no speechPCM. OriginalINT8remains54440B/04aa4564… and.91796875/smooth5. Sourcecommits identify originalcode/historicalevidence, not falselyclaimedcommittedrawartifactbytes.

SavedModel signature input_audio float32[batch,3,40],output_0[1,1]. Mac must explicitlyzero allstatevariables beforeEVERYstream and use matchedoriginalfrontend/priming/padding/quantizationcontract. PriorconsumedMac350float12/50vsINT811/50diagnostic alreadyexists; no rerun or extrapolation to new126Normancommand result. Newhandoff enables Mac's requested126same-checkpoint distinction, not quality/acceptance/promotion and not a formaltrainingprerequisite.

OriginalcontrolPID719837live1:25:52,last29500steps, latestsavedcheckpoint25000.284executionpins reverifiedunchanged. No restart, alternateformal or newcandidate/acceptance score; v30frozen. Continue original120k×2 and newdevelopment/acceptance plan.

Afterfloat Issue13handoffcomment6848557000, originalcontrol30000checkpoint completed and preservedunchanged SHAeb501136e4911b68f795ee19d5a95e021be2e12f8d4d4f47ef6e8926f71d10e6,0training-floateligible points. No selectedcheckpoint replacement, INT8pass or newqualityclaim; originalruncontinues, recoveredwaits. Completecheckpointline/receipt pushed separately from originalfloatarchive.

### 2026-11-13 — Preregister consumed-Joe original-float contrast

With originalformal stilllive1:27:40/30000steps, prioritize a concrete causal diagnostic during the wait:does frozenv30's0/72Joeactivation persist in its authenticated originalfloat graph, or mainly arise afterquantization? This complements, does not duplicate, Mac'sconsumed126originalfloat task. All360Joe clips alreadyconsumed; no newaudio/acceptance access.

Pinned CPUonly2thread TF2.21 three-arm diagnostic beforeexecution:originalSavedModel+unroundedclip(raw/25.6,0,26); samefloatgraph+exactoriginalINT8-inputquantize/dequantizefeatures; savedoriginalINT8smoothedtraces authenticated to completedbaseline. SamewholePCM/continuousfrontend/padding/steps, zeroallfloatstates eacharm/stream, firststreamrepeat-exactbothfloatarms, allfinite/range/count checks. Save allfloatraw/smoothedscores, exactoldINT8trace/event consistency and fullperclip events/peaks/errors. Frozen.91796875/smooth5/latch only, no grid/tuning/alternatecandidate. Inputrounding is isolated withinfloatgraph; residualdifferences include weight/activation/state/runtimequantization, not an additivecausal decomposition or solelyweights. Anyincomplete/error preservespartialartifacts andpreventsfullsummary, no retries. Originalformal/checkpoint/gates remainunchanged; Mike/acceptanceunopened.

### 2026-11-13 — Joe original-float contrast completed

The diagnostic frozen and pushed at b3dd457 completed all 360 consumed Joe presentations normally (session 11357, PID 766325), using CPU only. Both float arms reset state before every stream; the first stream reproduced exactly after reset in each arm. All saved counts, finite/range constraints, original INT8 trace hashes and original events matched. The two float raw traces and three smoothed traces per presentation are retained in scores.npz (SHA256 5a98f616c4d4882ea5f17bc5702bed4df6415a50c0f20292a5e9bb1ec1e8c11c), with complete per-row inputs, events, peaks and errors.

At the unchanged 0.91796875 threshold and mean5, original float, input-rounded float and original INT8 each activate 0/72 intended Hey Nicks positives, 0/72 cross-targets and 0/216 common confusions; no pre-source events. Their highest positive five-score means are respectively 0.8511093139648439, 0.8495105981826783 and 0.8054687500000001. Thus zero activation at this operating point already occurs before quantization on this corpus; it cannot be explained solely as an INT8-introduced failure. This is not a float-optimal threshold, phonetic truth or generalization claim, and does not alter the frozen baseline or sole recovered candidate.

Across all 48,420 smoothed scores, original-float vs input-rounded maximum absolute difference is 0.0199132621 (MAE 0.0000120489231); input-rounded vs INT8 maximum 0.117543364 (MAE 0.000229482462); original-float vs INT8 maximum 0.137423724 (MAE 0.000232659452). Input rounding affects the float graph, but these errors are not an additive causal decomposition. The residual includes weight/activation/state quantization and runtime numerics. Mostly small background scores do not rescue failed positive activation. No threshold search, new audio, retraining, re-export, candidate or acceptance inference occurred. Mac's consumed126 diagnostic remains separate.

Published gpu-joe-v30-float-diagnostic-evidence-v1.tar.gz, 1,255,888 bytes, SHA256 9605f66fc5b3a535123e05b82bb12e6513aa4e705e9cf4496cac1ac1da75ee9d. Anonymous download verifies archive and all 7 payload members, including full raw/smoothed arrays, records, summary, pins and exact tools/protocol. Manifest/download receipts and complete records are committed. Original formal PID 719837 verified live at 1:36:04, last step 33500, latest preserved checkpoint 30000. All 284 execution pins remain unchanged; recovered waits sequentially. Goal remains incomplete, v30 frozen, Mike/acceptance unopened.

### Mike acceptance material admission preparation (2026-11-13)

Previous goal turn was no progress toward research (Issue channel confirmation
only). This turn verified original control PID719837 live at 1:43:58/36000
steps; preserved original35000 validation line SHA
`8b6bcd4c9932831957bd0333200e7c912015626f0b8165baf9f2c3db1574fdaa`,
0 training-float eligible points, fallback is not a pass. All284 formal execution
pins remain unchanged. Original sequence continues; recovered waits, no restart.

Added `prepare_mike_acceptance_v1.py` and material protocol. Metadata-only
reservation authenticates original120 requests/model/config/runtime/native and
render gates; SHA `20891f7806ed28db3cb45b771b9ec14a3174addaf33ab2a2b35930e0ade55289`.
No TTS inference/audio/background generation or access. The future generator
preserves complete float32[1,1,1,N] tensors, exact singleton transport, all120
native/360 full quiet/noise/RIR presentations and failed draws. Whole generated
boundaries are not phonetic truth; Joe/Mike share rdh parent.

Four admission tests pass, including actual absent-development write-lock and
generate rejection before any output/lock/session; analytic null/failure/stale
selection, forged eligibility, original138 grid and tie tests are not model
quality evidence. Script compiles. Mike and test6/7 remain unconsumed. A passed,
authenticated sole recovered development decision and pushed lock are required
before material generation; this component grants neither scoring nor promotion.
Goal remains incomplete; frozen v30 remains the reference.

### Analytic acceptance-generator execution test (2026-11-13)

Previous turn made progress (8143c44). Original trainer PID719837 verified live
at1:45:21, latest36500steps; no newer validation snapshot yet. Issue13 latest
comments contained no new Mac result at this observation.

Extended the admission suite to execute the actual generator path in a temporary
directory with a fake engine returning an analytic0.5s/220Hz float32[1,1,1,N]
sine tensor. Model/config/version/published-lock fixtures are mocked only inside
the test; no real TTS session, Mike source audio, acceptance background or model
inference. Full tensor bytes survive saving, native full-resample/PCM audit and
all3quiet/noise/fullRIR audits pass. One-source pack correctly exits1 with
mechanically_qualified=false (requires120native/360renders). Temporary outputs
are discarded and never represented as research quality evidence. All5tests
pass. This changes tests only; reserved generator and all formal inputs stay
unchanged. Research/acceptance remain incomplete.

### Require completed matched-input audit before new development (2026-11-13)

Previous turn progressed via generator integration test. Original PID719837
verified live at1:46:58/latest37000. Review found the future new-development
freeze did not explicitly require the existing complete two-arm input-audit
artifact, though the original sequence runs that audit before postprocessing.
Added that artifact as a required/pinned dependency and authenticate120000
steps/arm,120000different main batches, exact paired tensors/ordered source
schedule/initialization inventory, original schedule digest and no acceptance
claim. Training/postprocessor/canonical runtime sources are unchanged.

Two admission tests pass: seven corrupted/incomplete analytic audit cases are
rejected; actual unfinished run cannot create a freeze/output. Current readiness
lists5missing original training/audit/postprocessor artifacts, candidate not
scored. The original verify_complete_recovery_execution_pins_v1 prelaunch
command rejects existing output directories as designed (not a hash failure);
therefore independently rechecked all284 file_sha256 entries directly, all
unchanged. No restart, scoring, Mike inference or acceptance access. Objective
remains incomplete; v30 frozen.

### Original control40000 checkpoint (2026-11-13)

Previous turn was a verified wait. This turn repeatedly polled original
PID719837 live while progressing38000→40000, with no restart. At1:54:32 the
original40000 validation completed. Preserved full original line SHA
`1df54e3d8dfc78ed5142276856de0f78568dd76f04b201c96a5b214b89177146`.
Original11-point training-float grid has0eligible points; selected fallback
.994 has recall .5074074074, macro .5497395833, hard-negative false rate
.0247159091, max-source false rate .028125. Ambient0events/1.2048112674h
has zero-event nominal one-sided95upper2.4864743174/h, not a <=.5/h pass.
No INT8/new-development/acceptance conclusion follows. All284formal hashes
reverified unchanged; original sequence continues and recovered waits.
Issue13 had no new Mac response at turn start. No Mike/test6/7 access,
checkpoint substitution, alternate arm selection or promotion. Goal incomplete.

### Original control45000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 confirmed live through
44000→44500→45000, elapsed2:08:21. Preserved full original45000 line SHA
`1346124f77053ca4d3b1bc57f2914ddfd779a0dd50f29262f7e27bcf73a236ff`.
0training-float eligible points; fallback.994 recall .4416666667,
macro .4526041667, hard-negative false rate .0139204545,
max-source false rate .0208333333. Ambient0events/1.2048112674h still
nominal zero-event one-sided95upper2.4864743174/h; not a pass. No cross-arm,
INT8 or new-development/acceptance conclusion. All284formal hashes unchanged.
Original sequence continues/recovered waits; no restart/model/threshold change,
Mike/test6/7 remain unconsumed, v30 frozen and objective incomplete.

### Original control50000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly confirmed live
through48500→49000→49500→50000. Preserved original50000 validation line SHA
`fbf50f158a795a890a89be4340755c527e3bbe9311bf3ee5dbdbc2ed431ca1c6`.
Original training-float grid0eligible; fallback.994 recall .4717592593,
macro .5005208333, hard-negative false rate .0210227273,
max-source false rate .0260416667. Ambient0events/1.2048112674h still gives
nominal one-sided95upper2.4864743174/h, not a pass. No INT8/new-source or
cross-arm inference. All284formal hashes unchanged; original sequence
continues/recovered waits, no restart or checkpoint/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control55000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through53500→54000→54500→55000, elapsed2:36:37. Preserved original55000
validation line SHA
`942598e042ce49ebdbf5e2d83dbb4908ae11485351703d467b293b01e1cb3156`.
0training-float eligible points; fallback.994 recall .4365740741,
macro .4625, hard-negative false rate .0130681818,
max-source false rate .015625. Ambient1event/1.2048112674h=.83000551795/h;
receipt correctly leaves zero-event upper null, not a zero-event calculation.
Not a pass, INT8/new-development/acceptance result or cross-arm conclusion.
All284formal hashes unchanged; original sequence continues/recovered waits,
no restart/checkpoint/threshold substitution. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete.

### Original control60000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through59000→59500→60000, elapsed2:50:45. Preserved original60000 line SHA
`3799fc1391f10594edc12fa67814890dbfdf646abd5c8808a3a93ad327d56616`.
0training-float eligible points; fallback.99 recall .4375,
macro .4630208333, hard-negative false rate .0082386364,
max-source false rate .0109375. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes unchanged; original
sequence continues/recovered waits, no restart/model/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control65000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through64000→64500→65000, elapsed3:04:45. Preserved original65000 line SHA
`59666c76e7fe8a8e466c3fcf307489293a1e120204d1df182c54f4da05deb24c`.
0training-float eligible points; fallback.992 recall .4625,
macro .4989583333, hard-negative false rate .0167613636,
max-source false rate .0208333333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes unchanged; original
sequence continues/recovered waits, no restart/model/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control70000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through68000→68500→69000→69500→70000, elapsed3:18:26. Preserved original
70000 line SHA
`f4456d985f30f1cd53c7b311bdb1fe65dd23539a0491f88399f6d2dad846213f`.
0training-float eligible points; fallback.994 recall .4652777778,
macro .4841145833, hard-negative false rate .0167613636,
max-source false rate .0229166667. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes unchanged; original
sequence continues/recovered waits, no restart/model/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control75000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through73000→73500→74000→74500→75000, elapsed3:32:19. Preserved original
75000 line SHA
`1fff6b3cb0678fdf6414ff7627c4ddfab5817a57ddde3f0e1300f8fa979f04c7`.
0training-float eligible points; fallback.994 recall .4268518519,
macro .42421875, hard-negative false rate .0071022727,
max-source false rate .0119791667. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes unchanged; original
sequence continues/recovered waits, no restart/model/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control80000 checkpoint (2026-11-13)

Previous turn was verified wait. OriginalPID719837 repeatedly verified live
through78500→79000→79500→80000, elapsed3:46:30. Preserved original80000
line SHA `e8996a7d3060e044bb2cf50e052d47782cf8e21a91db1b93f41ed9e13b4b87b1`.
0training-float eligible points; fallback.994 recall .4740740741,
macro .4854166667, hard-negative false rate .021875,
max-source false rate .0270833333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes unchanged; original
sequence continues/recovered waits, no restart/model/threshold substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.

### Original control85000 checkpoint (2026-11-13)

Previous turn made coordination progress through Issue13. Original PID719837
verified live at84500 and85000, elapsed3:59:35. Preserved original85000 line
SHA `9217508f930a9034bae911574980ac090db357cd0da3357049b17353169256a9`.
0training-float eligible points; fallback.992 recall .4611111111,
macro .4671875, hard-negative false rate .015625,
max-source false rate .0208333333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits, no restart or candidate substitution.
Mike/test6/7 remain unconsumed, v30 frozen, objective incomplete.
GPU/Mac communication stays in Issue13 per user instruction.

### Original control90000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
89000→89500→90000, elapsed4:13:51. Original90000 line SHA
`ccf47ae22d0e3ceba59b9ce9f7bdbd400f0a020ecbbfa7a282fca831286d9aaa`.
0training-float eligible points; fallback.994 recall .4657407407,
macro .4734375, hard-negative false rate .0235795455,
max-source false rate .0302083333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control95000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
94000→94500→95000, elapsed4:27:42. Original95000 line SHA
`7d28b7e9be46b859767a2c077b9a9f84332015f337345f199b6db3be896f9aba`.
0training-float eligible points; fallback.992 recall .4833333333,
macro .4924479167, hard-negative false rate .0198863636,
max-source false rate .025. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control100000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
99000→99500→100000, elapsed4:41:45. Original100000 line SHA
`eb9c417312227d0bf76ac90fd8033731c710465ac5279f5a3096a56f51bfa19a`.
0training-float eligible points; fallback.994 recall .4763888889,
macro .4848958333, hard-negative false rate .0181818182,
max-source false rate .0239583333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control105000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
104000→104500→105000, elapsed4:55:12. Original105000 line SHA
`752a584357770db8dd160b54d099c8d9b11980ef26461844c757b5aa887a1540`.
0training-float eligible points; fallback.992 recall .5023148148,
macro .51953125, hard-negative false rate .0227272727,
max-source false rate .0291666667. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control110000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
109000→109500→110000, elapsed5:09:26. Original110000 line SHA
`350551b5a47d37c0747fe893c0efb5a04c762d41d6f586d22093111f7355a1ff`.
0training-float eligible points; fallback.994 recall .4990740741,
macro .525, hard-negative false rate .0232954545,
max-source false rate .0302083333. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control115000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID719837 verified live through
114000→114500→115000, elapsed5:23:33. Original115000 line SHA
`aa37d729115d3d49a1c8ba547d2041adcb77c839ab22fee35305e3c93eda473c`.
0training-float eligible points; fallback.994 recall .5361111111,
macro .5604166667, hard-negative false rate .0323863636,
max-source false rate .0416666667. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. No INT8/new-development,
acceptance or cross-arm conclusion. All284formal hashes independently unchanged;
original sequence continues/recovered waits. Mike/test6/7 remain unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original control120000 and sequential recovered launch (2026-11-13)

Previous turn was verified wait. Original PID719837 reached120000 and exited;
original sequence PID719834 automatically started recovered shell817143 and
trainer817144. No restart or alternate experiment. Original120000 line SHA
`8fc3f2ef396cfd81b67b5dfb77c1e97b03652b57a4719dfff3518b341a78b13b`.
0training-float eligible points; fallback.994 recall .5194444444,
macro .53828125, hard-negative false rate .0284090909,
max-source false rate .034375. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. Control training.json
exists; export/input-audit/postprocessors await recovered120000 in original
sequence. Collector receipt formal_run_complete=false remains deliberately
conservative: matched experiment is unfinished. All284formal hashes unchanged.
Recovered startup records seed726 and original initialization/frozenBN hashes;
actual full matched input equality is not yet proved. No INT8/new-development,
acceptance or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen;
objective incomplete. Coordination remains in Issue13.

### Original recovered5000 checkpoint (2026-11-13)

Previous turn was verified wait. Original recovered PID817144 verified live
through4000→4500→5000, elapsed14:51. Original5000 line SHA
`a3a1231765cd9bf80776227cc45f52caf72744f5e0eb563aa445c4b1a8efa150`.
5training-float eligible points under original narrow training grid criteria;
selected.98 recall .3236111111, macro .3661458333,
hard-negative false rate .0051136364, max-source false rate .005625.
This is not prospective development eligibility or acceptance. Ambient0events/
1.2048112674h gives nominal zero-event one-sided95upper2.4864743174/h,
insufficient for required≤.5/h. All284formal hashes independently unchanged.
Original120000-step recipe/checkpoint selection continues unchanged; no early
stopping, alternate candidate, INT8/new-development or cross-arm conclusion.
Full matched input audit/postprocessing await both arms. Mike/test6/7 remain
unconsumed; v30 frozen; objective incomplete. Coordination remains in Issue13.

### Original recovered10000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 verified live through
9000→9500→10000, elapsed28:54. Original10000 line SHA
`d603d6ac84d25b308f8f23d9561a1947fcc83a987598f344d36d1c67f30bc82f`.
0training-float eligible points; fallback.994 recall .3560185185,
macro .40625, hard-negative false rate .0133522727,
max-source false rate .016875. Ambient0events/1.2048112674h gives nominal
zero-event one-sided95upper2.4864743174/h, not a pass. All284formal hashes
independently unchanged. Original120000-step recipe/checkpoint selection
continues unchanged; no early stopping, alternate candidate, INT8/new-development
or cross-arm conclusion. Full matched input audit/postprocessing await both arms.
Mike/test6/7 remain unconsumed; v30 frozen; objective incomplete.
Coordination remains in Issue13.

### Original recovered15000 checkpoint (2026-11-13)

Previous turns were verified waits on original PID817144, progressing
10500 through14500; still live at15000, elapsed42:35. Original15000 line
SHA `53821717d8fa57a399c0a287c1814384f890aedd57db567d334199a749d4c5c0`
(4130 bytes) preserved with collector receipt. 0training-float eligible points;
fallback.994 recall .4819444444, macro .5682291667,
hard-negative false rate .0247159091, max-source false rate .03625.
Ambient1event/1.2048112674h, observed .830005518/h; zero-event upper
is correctly null. Not a pass or INT8/new-development/acceptance conclusion.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue; full matched input audit/postprocessors
await both arms. No cross-arm benefit claim. Mike/test6/7 unconsumed,
v30 frozen, objective incomplete. Coordination remains in Issue13.

### Original recovered20000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 continues through19500
to20000, elapsed56:29. Original20000 line SHA
`96a0ba5a959c17060bbc80d039aa6496c396e41cb833cc6055a4dbb746f5fcf5`
preserved with collector receipt. 0training-float eligible points; fallback.994
recall .4546296296, macro .5255208333, hard-negative false rate .0252840909,
max-source false rate .029375. Ambient2events/1.2048112674h, observed
1.6600110359/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue unchanged. Full matched input audit and
postprocessors await both arms; no INT8/new-development/acceptance or
cross-arm benefit conclusion. Mike/test6/7 unconsumed; v30 frozen;
objective incomplete. Coordination remains in Issue13.

### Original recovered25000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through24500 to25000,
elapsed1:10:48. Original25000 line SHA
`b1d831de129c345d1f8d2260bbd5d3b177b480f2e2b4457e54419b1082873741`
preserved with collector receipt. 0training-float eligible points; fallback.994
recall .4615740741, macro .5184895833, hard-negative false rate .0292613636,
max-source false rate .03375. Ambient1event/1.2048112674h, observed
.830005518/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms. No INT8/new-development/acceptance or cross-arm benefit
conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Coordination remains in Issue13.

### Original recovered30000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through29500 to30000.
Original30000 line SHA
`e737b6ed8e3bd8d2e5aeff290cce79572a1f5d78f1581637d1b933d92628b78c`
(3927 bytes) preserved with receipt. 0training-float eligible points;
fallback.994 recall .4819444444, macro .528125, hard-negative false rate
.0261363636, max-source false rate .035. Ambient0events/1.2048112674h;
nominal zero-event one-sided95upper2.4864743174/h, insufficient for≤.5/h.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered35000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through34500 to35000,
elapsed1:39:04. Original35000 line SHA
`6ad3e99be0cf7c49f0f420f09b3eb2c791bbce4e2accc1d5287b7dc6c4f9adb1`
preserved with collector receipt. 0training-float eligible points; fallback.994
recall .4833333333, macro .5690104167, hard-negative false rate .0198863636,
max-source false rate .023125. Ambient0events/1.2048112674h; nominal
zero-event one-sided95upper2.4864743174/h, insufficient for≤.5/h.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered40000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through39500 to40000,
elapsed1:52:50. Original40000 line SHA
`6936c7e3fdefae8df5fdd53fd05f6f6b9473bb0d054ba30184a1d7ca932621cd`
preserved with collector receipt. 0training-float eligible points; fallback.994
recall .4953703704, macro .56484375, hard-negative false rate .0235795455,
max-source false rate .0260416667. Ambient2events/1.2048112674h, observed
1.6600110359/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered45000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through44500 to45000,
elapsed2:06:39. Original45000 line SHA
`8c8cb941714ea3705d749cac21fb223393fd9d989cc10b5ecd308dc5a1a06fea`
preserved with collector receipt. 0training-float eligible points; fallback.992
recall .5101851852, macro .6059895833, hard-negative false rate .0275568182,
max-source false rate .02875. Ambient1event/1.2048112674h, observed
.830005518/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered50000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through49500 to50000,
elapsed2:21:22. Original50000 line SHA
`9c8fd81b58ff52beefd0312c38768119bcb0a23f0960850f3368c07ed5c4ee1c`
preserved with collector receipt. 0training-float eligible points; fallback.994
recall .5615740741, macro .5966145833, hard-negative false rate .0386363636,
max-source false rate .041875. Ambient1event/1.2048112674h, observed
.830005518/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered55000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through54500 to55000.
Original55000 line SHA
`c783a4561d2fba0c7b85805819dae074d3c848994c12c87bdfadbb9357be0d59`
(4087 bytes) preserved with collector receipt. 0training-float eligible points;
fallback.992 recall .5171296296, macro .5661458333, hard-negative false rate
.0303977273, max-source false rate .033125. Ambient1event/1.2048112674h,
observed .830005518/h; zero-event upper correctly null. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered60000 checkpoint (2026-11-13)

Previous turn was verified wait. Original PID817144 live through59500 to60000.
Original60000 line SHA
`31d5fa291d419d8a0920892c118cf8b9ee732fda80535b7d2830d7c8ec1f3445`
(4037 bytes) preserved with collector receipt. 0training-float eligible points;
fallback.994 recall .4768518519, macro .5489583333, hard-negative false rate
.0184659091, max-source false rate .021875. Ambient0events/1.2048112674h;
nominal zero-event one-sided95upper2.4864743174/h, insufficient for≤.5/h.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete. Issue13 coordination.

### Original recovered65000 checkpoint (2026-11-13)

Original PID817144 remains live. Original65000 validation bytes preserved:
SHA `5814ccbd8b657a844226c6a5ec6d66633d35df33c6ad996aeff9f26d907f5168`.
0 training-float eligible points; fallback .994 recall .5893518519,
macro .6158854167, hard-negative false rate .0340909091, max-source
false rate .0390625. Ambient0events/1.2048112674h; nominal one-sided95
upper2.4864743174/h, insufficient for≤.5/h. Not a pass.
All284formal hashes independently unchanged. Original120000-step sequence
and checkpoint selection continue. Full matched input audit/postprocessors
await both arms; no INT8/new-development/acceptance or cross-arm conclusion.
Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
User explicitly requests communication through Issue13.

### Original recovered70000 checkpoint (2026-11-13)

Original PID817144 remains live; all284formal execution hashes unchanged.
Original70000 validation SHA `d6e63ec596221fa71047d6446e3e5337768721f1ee5d6c7e75d3df213a369309`
preserved with collector receipt. 0training-float eligible points; fallback
.994 recall .5796296296, macro .6213541667, hard-negative false rate
.0340909091, max-source false rate .0380208333. Ambient0events/1.2048112674h;
nominal one-sided95upper2.4864743174/h, insufficient for≤.5/h. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Results coordinated through Issue13.

### Original recovered75000 checkpoint (2026-11-13)

Original PID817144 live. All284formal hashes unchanged. Original validation
SHA `af54f7615c31f11b5ab4d44f8f4c613021ff310610522f3c6fe06dcb0c5b8070`
preserved with collector receipt. 0training-float eligible points; fallback
.994 recall .6208333333, macro .6627604167, hard-negative false rate
.0465909091, max-source false rate .0541666667. Ambient1event/1.2048112674h,
observed .830005518/h; zero-event upper correctly null. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Issue13 coordination continues.

### Original recovered80000 checkpoint (2026-11-13)

Original PID817144 live; all284formal hashes unchanged. Original validation
SHA `9617b709254e8c37bddc23d579344d2accfd505f4e1458e33471d8b998b9ace6`
preserved with collector receipt. 0training-float eligible points; fallback
.994 recall .5824074074, macro .61015625, hard-negative false rate
.0338068182, max-source false rate .0385416667. Ambient0events/1.2048112674h;
nominal one-sided95upper2.4864743174/h, insufficient for≤.5/h. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Issue13 coordination continues.

### Original recovered85000 checkpoint (2026-11-13)

Original PID817144 live; all284formal hashes unchanged. Original validation
SHA `22117e2c859073aa6fca3d83c038b6cfdd9d92e7137fd7df2ea31d4403096a62`
preserved with collector receipt. 0training-float eligible points; fallback
.994 recall .5745370370, macro .63671875, hard-negative false rate
.0321022727, max-source false rate .0348958333. Ambient0events/1.2048112674h;
nominal one-sided95upper2.4864743174/h, insufficient for≤.5/h. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Issue13 coordination continues.

### Original recovered90000 checkpoint (2026-11-13)

Original PID817144 live; all284formal hashes unchanged. Original validation
SHA `b0efac565cb7cf94f750d45e123552085399f67dd416b27428a32954cc573a52`
preserved with collector receipt. 0training-float eligible points; fallback
.994 recall .5726851852, macro .5973958333, hard-negative false rate
.0397727273, max-source false rate .0442708333. Ambient2events/1.2048112674h;
observed1.6600110359/h; zero-event upper correctly null. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Issue13 coordination continues.

### Original recovered95000 checkpoint (2026-11-13)

Original PID817144 live; all284formal hashes unchanged. Original validation
SHA `9f51d55c2e02b0709b21c03a336779008bf9f899ddcf9128d53e7c56cda5a907`
preserved with collector receipt. 0training-float eligible points; fallback
.992 recall .5865740741, macro .6088541667, hard-negative false rate
.0423295455, max-source false rate .0479166667. Ambient1event/1.2048112674h;
observed .830005518/h; zero-event upper correctly null. Not a pass.
Original120000-step sequence/checkpoint selection continue. Full matched
input audit/postprocessors await both arms. No INT8/new-development/acceptance
or cross-arm conclusion. Mike/test6/7 unconsumed; v30 frozen; objective incomplete.
Issue13 coordination continues.

### Independent event arithmetic audit (2026-11-13)

Authenticated public Mac analytic archive SHA3371f8e7848be123f46c3df0a1678301ad5361ce51af5b81e4f9f1d860a8066e.
Executed archived script with authenticated original runtime/evaluator on Linux,
NumPy2.4.6 and Debian C++12.2.0. 28874windows/138points yielded356legacy
predicate and356compiled event disagreements (164single-trigger,192rearm).
Canonical UINT8 integer sum /1280 with binary64 threshold had0compiled
disagreements on these356streams. Platform-dependent count differs from Mac385.
No saved waveform regression, model inference, candidate scoring or acceptance.
Original formal inputs unchanged. Full platform/build/source receipt saved in
results/4090-numeric-event-policy-independent-v1.json. Future metadata/arithmetic
changes and new Hey Willow/Nyx paired protocol remain to implement before scoring.

### Future calibration metadata compatibility (2026-11-13)

Added validation_only=True, final_data_used=False, sliding_window=5 to future
calibrator only, outside284formal pins. AST after removal equals prior source;
6decision tests pass. Mac packager audit failed because prepare_runtime.py is
absent on GPU branch; integration remains unverified, not candidate success.
No scoring/freeze or historical report changes. Explicit arithmetic integration
and new Hey paired protocol remain outstanding.

### Isolated Mac metadata consumer integration (2026-11-13)

Fetched full immutable7a1f82c26bcce4d3d62fdcf9ae089c8ef687b31b, archived consumer
into isolated /tmp directory and ran original Mac audit unchanged. All7analytic
contract checks pass; current patched calibrator SHA88027cb2165baad13a197b1e2c733fd1a852f4c95b881ebe9599540198918bdc
matches audited proposed bytes. Resolves previous missing-import limitation.
No actual candidate scoring/packaging, waveform inference, firmware or acceptance.
Original failed attempt preserved; active formal sources unchanged.

### Original recovered100000 checkpoint (2026-11-13)

recovered 原100000步完整验证已保存。原始记录4034 bytes，SHA256 `3e56de298dc6e80414a11f1e118f182b9c853db5962571efb983adb842331e21`。

training-float grid 0 eligible；fallback 0.994：recall 0.600462962962963、macro 0.6239583333333334、hard-negative false rate 0.04403409090909091、max-source false rate 0.051041666666666666。背景0 events/1.2048112674h，名义单侧95%上界2.4864743174/h，不满足≤.5/h，不是通过。

原 PID817144 已继续至100500；284 formal哈希全部一致。完整逐批审计/原后处理等待两臂完成；无INT8/新开发/独立验收或跨臂收益结论。Mike/test6/7未消费，v30冻结。新Hey配对协议67293a9已提交，原实验和新协议分开推进，目标未完成。


### Future development arithmetic wiring (2026-11-13)

Future scorer now saves original UINT8 traces and derives integer-sum/1280
binary64 means; future calibrator authenticates raw hashes and exact means.
Both declare uint8-sum-binary64-v1; freeze includes detector/policy module bytes.
356compiled boundary streams match integrated means/event indices;6decision
and2matched admission tests pass. All284formal hashes unchanged; no candidate
freeze exists and no speech scored. Consumer opt-in profile and full future
freeze still required before scoring; prior metadata-only compatibility receipt
does not certify this arithmetic-enabled producer with latest Mac consumer.

### Explicit pilot target provenance admission (2026-11-13)

Added separate validate_hey_target_pack_v1.py, outside active trainer sources.
Requires target/pronunciation identity, ordered row hashes, feature file hashes,
source PCM byte hashes, reciprocal target labels and voice model role separation.
Analytic six-row fixture accepted; wrong target, cross-target positive label,
voice-role leakage and changed PCM hash mutations rejected. This is declared
provenance only; no real pack generated and no training launch adapter admission
claimed. Feature dimensions/quality/source-overlap/material lock and CPU batch
integration remain necessary before new training. Original live PID817144 observed
at102500; no formal input edits.

### Mac contract acknowledgement and recovered105000 (2026-11-13)

Read Mac6860757149/6860895742: native actual-speech diagnostic complete, not
independent acceptance/parity. Adopted13configuration official grid and explicit
binary32/getter/event/runtime certification contract in paired protocol. Clarified
120base/voice,240base/target,720noise-augmented/target, dev one seed/cell,
Hi E S P literal, original-rate replay, command-container latency limitations,
and paired official comparison beyond engineering gates. Material/schedule lock,
real target adapter/CPU batch and runtime certification remain incomplete.

Preserved original recovered105000 record SHA256
72ead1644764cbddb0a474872bde3451808d9547fa352f1b923e9430ce2099c9.
0eligible float points; fallback .994, recall .5921296296, macro .6283854167,
hardnegative .0465909091, maxsource .0541666667. Background0/1.2048112674h,
zero-event one-sided95 upper2.4864743174/h: not pass. All284formal pins verified
unchanged, PID817144 live. No INT8 candidate promotion or acceptance conclusion.

### CPU actual base batch execution (2026-11-13)

Added audit_hey_training_batch_v1.py. Extracts unchanged scale/sequence/sample/batch
AST directly from current train_stream.py, without TensorFlow/model updates.
Executed on existing features-gpu-ami-extra-v1 training rows/ambient only:
128x150x40 finite scaled inputs,32positive/96negative labels,1/20weights and
same-seed byte-identical repeat passed. Receipt4090-hey-base-training-batch-audit-v1.json
binds trainer source and batch bytes. Optional --target invokes provenance gate.
This is NOT new Willow/Nyx pack certification: target_admission=null, no new
TTS or pack exists, no full schedule/launcher/optimizer/model-gradient audit.
Actual target adapter and new-pack CPU execution remain required. No old
validation/testing audio or final material scored, no historical pack relabeled.

### Local source inventory, role reservation and separate target adapter (2026-11-13)

Current local inventory checked174ONNX bytes and4745JSON/JSONL files, including
candidate model hashes. No candidate bytes at prior model paths; only source-name
match is Semaine text in qwen vocabulary, not synthesis evidence. Saved local
inventory SHA54b08cc19c4dc73320f06a3a0bc00dca8739c1f754b33566f381b5169a368b2e
and tracked receipt4090-hey-local-source-history-audit-v1.json. Bounded current
accessible metadata/model evidence, not deleted/external history or PCM overlap.
No audio opened. Shared lessac parent remains an explicit limitation.

Exact voice-role reservation: Kusal/Reza training; Semaine speaker0/prudence
only development; HFCmale acceptance. Model/config/card bytes bound in
4090-hey-paired-voice-role-reservation-v1.json. No other Semaine speaker used in
another role. This is not full material lock or TTS authorization execution;
text/seed/renderer/RIR/negative inventories and new PCM overlap checks remain.

Separate train_hey_target_adapter_v1.py now constructs an AST derivative without
editing original train_stream.py. Removes validation calls and best-score
selection, saves fixed final20000 only; deterministic ops/noTF32, fixed base
teacher60/context150/frozenBN recipe. Launch requires pushed identical training
lock, material commit, bound code/core features/weights, target provenance and
training-only rows. No new launch has occurred and full lock is absent.

Found and fixed preparation conflict: old provenance validator required all
three splits, which would unnecessarily force reserved material into training
packs. Added optional required_splits; existing default remains unchanged;
adapter requires training only and rejects other feature/ambient roles. External
fresh development/acceptance are not needed to train or export.

AST source construction/compile passes; actual transformed loop executed with
stub model20000updates, exactly one final save, learning-rate assignments at
10000/16000,4diagnostic records. Two executable tests cover final controlflow,
training-only provenance admission, default split rejection, wrong target and
reciprocal mislabel rejection. This is not TensorFlow gradient/export validation
or new actual-pack batch admission. Previous actual historical CPU batch remains
separate evidence. All284formal hashes rechecked unchanged; PID817144 live107000.

### Exact staged training synthesis lock (2026-11-13)

Added generate_hey_paired_material_v1.py with native and complete rendered outputs.
Plan enumerates2640native requests:2400training plus120dev/120acceptance RESERVED.
Training expands to7200quiet/25dB/15dB+fullRIR clips,720positives per target.
Fixed texts/carriers/speeds/draws, phones/token IDs, model/config/card, phonemizer
binary+espeak data hashes, runtime versions, explicit seeded ONNX derivation,
noise/RIR/padding/gain and code/quality bytes. Only training generation authorized
after pushed exact plan. Background/overlap/features/full schedule admission still
required before training; future development/acceptance not generated/scored.
Original models unchanged; derived random-op seed attrs only. Kusal graph audit
confirms learned initializers/nonrandom nodes unchanged and deterministic graph
byte derivation; not yet speech/runtime determinism evidence. Both mechanical
quality suites6+6 passed. Any source failure disqualifies whole planned pack,
no retry/filter/substitution. Preserves complete native output and PCM transport;
no phonetic truth or human listening claim. Model gradient/export still pending.

### Native v1 shape failure retained; v2 recovery preregistration (2026-11-13)

First real inference revealed Kusal returns float32[1,1,1,T], not assumed
[1,1,T]. Stopped only new generator PID888489 after16records,0passed; original
trainer817144 untouched. All16complete engine tensors retained. v1 failure
receipt records exact plan/records hashes and unqualified status; never relabel
v1 as pass. No detector scores or development/acceptance generation occurred.

v2 plan supersedes transport shape handling only: accepts declared singleton
batch/channel axes before T; never crops samples. Reuses16saved original engine
tensors after derived-model/tensor hash authentication, without inference rerun.
Remaining requests run once. Original v1 plan/generator/output preserved. v2 binds
failed-plan/failure-receipt/prior-record hashes and its exact generator bytes;
no source/text/seed selection based on wake scores, which do not exist. Full
native mechanical gate still applies and any actual quality failure disqualifies
v2pack. Complete background/overlap/training admissions remain pending.

### Native synthesis live and original110000 preserved (2026-11-13)

v2 native generator PID889095/session63046 confirmed live; first27records27passed,
including16recovered original engine tensors without inference rerun. Partial
observation only, not whole-pack qualification. No dev/acceptance generated.
Original recovered110000 full record preserved in135fd0b, SHA256
2c947789717dcb92281e31b88f087cb6f8443506851820f55d3bfa8e6d964863.
0eligible; fallback.994 recall.6328703704 macro.6385416667,
hardnegative.059375 maxsource.0666666667, background1/1.2048112674h.
Zero-event upper correctly null, not a pass. Original120k sequence continues;
no INT8/development/acceptance or cross-arm conclusion.

### Training timing metadata extraction preregistered (2026-11-13)

Found a new-pack integration hazard before any model training: existing base
sampler centers positives on keyword_frame; using whole command-container end
would supervise command tails instead of wake prefixes. No such pack/training
has been launched. Current full-output container references remain diagnostics.

All4prospective VITS graphs contain /Ceil_output_0 token allocations and decoder
strides8/8/4. Separate extraction plan/script pins exact seeded generation and
metadata code before execution. Changes graph outputs only, never regenerates
or selects waveform: authenticates saved waveform, requires nonnegative integer
one-per-token allocations and sum*256==saved native length. Prefix is before
comma for commands or period for bare, includes preceding blank allocation.
16k mapping uses exact integer ceil. This is model-internal allocation, not
phonetic truth; no human speech latency claim. First fixed training request is
smoke only; full extraction waits for all2400native mechanical passes. A mismatch
fails timing admission rather than adjusting tokens. No original formal edits.

### Token allocation smoke passed and Mac getter clarification (2026-11-13)

Executed preregistered fixed first-training token-allocation smoke after62fdd30
push. Saved4090-hey-token-duration-smoke-v1.json:27token allocations sum69frames,
69*256=17664native samples exactly matches preserved original waveform. Prefix
before period ends14080native/10217transport samples; not full17664container end.
No waveform generated, wake scoring or model training. One bare Willow example
only: command mappings/all voices/full2400timing admission remain unverified.
Separate metadata sidecar does not rewrite original waveform-only duration=null
records. This is engine allocation, not acoustic phoneme truth.

Read Mac6861413399/d0c589b:13configuration implementation and two builds done;
zeroPCM API run still pending, not native speech-curve certification. Corrected
paired protocol getter range: setter requests .4.. .9999; actual getter finite
0<getter<1 with exact f32 bits and >=9significant digits, not forced into request
range/equality. Prime return coverage/controlDEFAULT remains required. New native
training synthesis observed280/280passes, still live and incomplete. No dev or
acceptance generated. Full negative inventories/new feature admission pending.

### New paired continuous background sources reserved (2026-11-13)

Fetched Mac d0c589b and authenticated its original12Earnings22 reservation
SHA b10b875cbbcfb09bd9b5c2aa41ccb31450b48bc9f7f7c3ce62c4547595135d53.
Downloaded pinned upstream metadata/README/license only; identical original
hashes. Deterministically exclude all12prior tickers, order remaining tickers
by frozen SHA namespace, select first12development/next12acceptance and fixed
one-file minimum per ticker. 24LFS pointer hashes/media SHA/size/URLs locked.
Metadata hours10.4827777778development and13.7075acceptance; descriptive only,
not decoded/scored exposure. No media yet downloaded. Ticker/source disjointness
is not physical-speaker independence; analysts may recur. No new human capture.
Original Mike/test6/7 and three consumed finals remain separate/unmodified.

prepare_hey_earnings_development_v1.py has ONLY development download/decode path.
Separate pushed prep lock authenticates all12sources, decoder/microfrontend
binary and package versions; exact transcript alias match fails whole readiness,
no easier substitution. Complete native float hash/duration, fullmono16kPCM16,
continuous frontend per call, final-onlypadding; no crops/gain/VAD/internalgaps.
All12complete and decoded>=8h required. History encoded/PCM overlap remains
separate before scoring. Original media/decoded/frontend stay local; license
explicitly covers transcript/text, not an identical media grant. No acceptance
media/transcripts opened. No model scores in this preparation.

System FFmpeg absent. Global pip install blocked by PEP668; used isolated
.local/wakeword/ffmpeg-tool-v1 imageio-ffmpeg0.6.0 instead, without changing
existing numerical/training packages. Binary7.0.2-static SHA locked; differs
from Mac9.0.2 so no cross-host PCM equality claim. Random320037sample fixture:
10s IO continuous frontend equals full-array frontend exactly,123finalpad,
1999x40features, original PCM hash preserved. Real-source decode pending.

### Original recovered115000 and native synthesis status (2026-11-13)

Preserved original115000 complete record SHA
0a40c0ea932b835bcad37868bc4a6f121d6db78c404b3e7ac5fd26ba3e19b681.
0eligible float points; fallback.994 recall.6333333333 macro.6588541667,
hardnegative.0511363636 maxsource.059375, background1/1.2048112674h.
Zero-event upper null; not pass, no candidate promotion. All284formal hashes
unchanged. Native generator live800/800mechanicalpasses, incomplete2400pack;
no synthetic development/acceptance generated or scored.

### Real continuous development preparation launched (2026-11-13)

After4481444push, prepPID894418/session21239confirmed live. Immutable completed
record prefix saved4090-hey-earnings-development-progress-v1.json: 4calls/3.212351701388889hours
full decoded/source/frontendgeometry passed at snapshot. Whole12cohort still
incomplete, overlap not admitted, no wake inference or scored FAR. Acceptance
media/transcripts unopened. Public media/decoded/frontend remain local.

### Complete new continuous development preparation and bounded overlap audit (2026-11-13)

PrepPID894418/session21239finished exit0. All12complete selected calls processed,
actual decoded10.483944826388887h (metadata10.4827777778h was descriptive).
Manifest SHA ff0519e735e1c9bb1205ec0ddc7422c7cca74cf1c07678f352b23dca70809841.
Zero exact alias matches across all original transcripts; whole-call native
sample duration,source bytes,mono16kPCM16,continuousfrontend/finalpad gates pass.
Tracked complete metadata receipt, no raw/decoded/frontend published. No wake
inference, scored FAR, threshold or candidate result. Acceptance untouched.

Independent read-only history audit reauthenticates all12originalMP3 LFS hashes,
WAV/frontend bytes and recomputes PCM hashes/sample counts. 36unique source/WAV/
PCM hashes compared to2157GitJSON/JSONL blobs from pre-reservation GPU064423c
and Macd0c589b plus4755accessible local metadata files, excluding only current
source/prep/audit self-records. No exact matches. Local inventory SHA
73be23a3ef9ebf730c9122e07d50cae1f4cc3b932f5d5538ae9ee4b9cc3eeb52.
Not proof against deleted/unrecorded files, excerpts/re-encoding or physical
speaker overlap; no old/final audio opened. Bounded evidence preserved and
source/scoring admission still false pending full study locks.

### Mac native threshold API evidence independently read (2026-11-13)

Read6861666653 and fetched aa4f47c917cad3a1a5bad7ad8bf51f37de2e5702. Local
saved-report audit authenticates rawJSON SHA4f0bd0995513441cd457fc46c61be5bfa7ec50921eeab5b200c86e1e0b40d5c0:
28zero streams,24successfulSET,896input calls,1764prime returns,zeroPCM CRC
1112058211; all request/getter decimal-f32bits/call accounting checks pass.
Each engine13distinct actual working points; DEFAULT and SET.644 remain distinct.
Willow DEFAULT.643999994,HiESP DEFAULT.629999995; SET.4 actual.396800011,
SET.644 actual.638848007,SET.9999 actual.991900802. These observed mappings are
not an assumed universal formula or QEMU getter substitution. Zero API actual
execution is no longer pending; speech/FAR curves and native>64event branch
remain separate. Our audit checks savedJSON only, not a second P4/UART replay
or whole-app/voice acceptance. No existing formal source changes.

### Matched training feature builder implemented, not yet run (2026-11-13)

build_hey_training_features_v1.py now prepares both explicit target arms using
complete serialized rendered PCM/frontend, shared binary and identical old
training-only ambient arrays. Fresh dev/acceptance arrays are never imported.
Exact plan can only be written after2400native/7200rendered and2400matching
timing records qualify; native/render generation/records hashes, unique IDs and
three variants are authenticated. Future plan binds every ambient NPY, code,
frontend binary and numerical versions before extraction. No feature plan or
actual new pack currently exists, and no training launch occurred.

Positive keyword_frame maps internal prefix-before-punctuation allocation plus
full synthetic RIRtail, not full command-container end. Complete prefix support
is checked for every original3..27positive jitter undercontext150; unsupported
rows invalidate entire pack without filtering. All waveform/feature frames
remain preserved; context crops are explicit training operations. Boundaries
remain generated allocations, not acoustic phonetic truth. HiESP remains a
common negative for both custom arms while text label is preserved. Reciprocal
target rows receive correct labels. Features-sidecar stores source byte/PCM/model
hashes and target identity for actual CPU admission.

3geometry tests passed: command end differs from prefix, all jitters include
supported prefix, long-prefix unsupported flag, source/role mismatch rejection.
2adapter tests still pass. Future actual CPU auditor now requests training-only
provenance, matching adapter; default legacy validator behavior unchanged.
Original trainer/pins untouched. Native job observed1400/1400passes, still live;
original training observed119000, not yet complete. Full new pack admission,
actual batch/gradient/export checks and tensor schedule lock remain pending.

### Original matched formal training/export completed (2026-11-13)

Both original arms completed120000updates, each selected5000 by unchanged
float rule and exported54440byte INT8 models. Original training.json bytes
preserved separately for both, no rewriting. Control model SHA
83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca;
recovered SHA0a39c5924c1e132bc2e535e0f116cf3b63c716b49b477fd99a82b1e16ef2f334.
Input/output tensors independently allocated/inspected [1,3,40]int8/[1,1]uint8,
output scale1/256 zero0; no new speech invoked. All284pins match.

Full120k paired source draws and paired network tensor hashes match, same
initialization inventory, all120k main batch hashes differ as intended. Original
audit preserved byte-for-byte; limit remains not all108rows every formal step
(the separate two-update fixture covers row preservation). Automatic sequence
PID719834advanced to control historical postprocessing, not restarted. Control
export audit passed. Recovered original postprocessor waits its original turn.

Selected5000float points were NOT fallbacks: control3eligible and recovered5.
Control threshold.99/macro.4125; recovered.98/macro.3661458333. These coarse
reused-float eligibility points are not final synthetic acceptance; zero-event
1.2048112674h upper95is2.4864743174/h. No efficacy/parity/promotion conclusion.
Last120000 recovered checkpoint itself0eligible/fallback.994/macro.6473958333,
recall.6356481481, maxsource.0552083333, background0/1.2048112674h. Original raw
record SHA966c29a0521dede3bac4817f68b19dde6324d6966b36cc7d9fe78f9c5f3b6138.
Collector historical receipts deliberately remain their original checkpoint-only
formal_run_complete=false; separate training-completion receipt now records
actual training/export completion with postprocessing/newdev/acceptance pending.
