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
