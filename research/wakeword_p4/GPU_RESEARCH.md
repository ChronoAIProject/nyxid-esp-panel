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
