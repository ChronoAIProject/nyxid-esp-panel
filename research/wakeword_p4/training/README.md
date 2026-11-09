# Reproduce the Hey Nyx training experiments

Read [BEST_PRACTICES.md](BEST_PRACTICES.md) for primary sources and the rejected
first candidate. This is training research, not a production wake-engine update.
All generated speech is AI-generated audio, not human recordings.

The continuing Hi ESP comparison goal and current evidence are tracked in
[PROGRESS.md](PROGRESS.md).

## Data used in the four-engine experiment

| Generator | Train positive / negative | Validation positive / negative | Test positive / negative |
| --- | --- | --- | --- |
| Piper LibriTTS-R | 3,000 / 3,000 | 480 / 480 | 480 / 480 |
| Kokoro ONNX | 60 / 400 | 12 / 80 | 12 / 80 |
| Qwen3-TTS VoiceDesign | 96 / 384 | 0 / 0 | 0 / 0 |
| OpenAI gpt-4o-mini-tts | 81 / 108 | 18 / 24 | 18 / 24 |

Total: **9,317 source clips**. Four waveform renderings per clip are correlated
variants, not additional independent speakers or trials. Piper has 904 source
speaker IDs, Kokoro has 28 English preset IDs, and OpenAI has 13 voice IDs.
Qwen uses 32 description conditions, which do **not** establish 32 independent
speaker identities; Qwen samples are training-only.

Source speaker/voice IDs are split before synthesis and augmentation. Derivative
voice clones must retain their carrier's split. Cross-engine physical speaker
identity is not established by preset names.

The experiment also uses speaker-disjoint LibriSpeech dev-clean negatives and
a hash-checked, CC0 environmental subset of MS-SNSD. See the manifest produced
by `backgrounds.py`. Environmental files are blob-disjoint across splits, but
the source does not establish independent original room/recording sessions.
Real user positive wake recordings and Chinese/TV/music negatives are missing.

## Environments and source pins

Use the parent research README to prepare the pinned Apache-2.0 microWakeWord
and Piper repositories and the trusted Piper checkpoint. Keep generated data
under an ignored/private directory, outside source-controlled files.

```sh
export WAKE_WORK="$PWD/.local/wakeword"
export WAKE_DATA="$WAKE_WORK/hey-nyx"
mkdir -p "$WAKE_DATA"
uv venv --python 3.12 "$WAKE_WORK/tf-env"
uv pip sync --python "$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/requirements-training.lock
uv venv --python 3.12 "$WAKE_WORK/kokoro-env"
uv pip sync --python "$WAKE_WORK/kokoro-env/bin/python" research/wakeword_p4/training/requirements-kokoro.lock
uv venv --python 3.12 "$WAKE_WORK/mlx-env"
uv pip sync --python "$WAKE_WORK/mlx-env/bin/python" research/wakeword_p4/training/requirements-mlx.lock
uv venv --python 3.12 "$WAKE_WORK/asr-env"
uv pip sync --python "$WAKE_WORK/asr-env/bin/python" research/wakeword_p4/training/requirements-asr.lock
```

The tested main runtime is Python 3.12 / TensorFlow 2.21 on CPU; synthesis uses
Piper MPS, Kokoro ONNX CPU, and Qwen MLX/Metal in separate environments. There
is no Docker or tensorflow-metal requirement. OpenAI synthesis is an API call;
its credentials are read from the environment or an external `--key-file`, never
written into fixtures or manifests. Only the preset public training phrases
are sent. There is no audible speaker playback in these generation scripts.

Qwen model pin: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit`, revision
`ffc6545dc9cb086950aa46c6cd3db490e6ece3e1`. Download with `hf download` and
`--revision` into a separate local model directory.

Kokoro's `model-files-v1.1` release assets are `kokoro-v1.0.onnx` and
`voices-v1.0.bin`; the generator verifies their published SHA-256 values before
loading. See `kokoro_samples.py`. Its ONNX speed tensor is cast according to
the model schema because kokoro-onnx 0.4.9 otherwise chooses the wrong dtype.
The corpus used the library's default silence trim; waveform checks do not
prove that every weak consonant is retained.

OpenAI's server model alias and sampling are not deterministic. A local corpus
manifest hashes the actual generated WAVs. The [official TTS guide](https://developers.openai.com/api/docs/guides/text-to-speech)
documents the 13 voices and speech API. The same voice ID never crosses this
experiment's train/validation/test split.

## Generate source clips

```sh
"$WAKE_WORK/tts-env/bin/python" research/wakeword_p4/training/generate.py \
  --upstream "$WAKE_WORK/piper" --output "$WAKE_DATA/piper" --per-class 3000
"$WAKE_WORK/kokoro-env/bin/python" research/wakeword_p4/training/kokoro_samples.py \
  --assets "$WAKE_WORK/kokoro-models" --output "$WAKE_DATA/kokoro"
"$WAKE_WORK/mlx-env/bin/python" research/wakeword_p4/training/mlx_samples.py \
  --engine qwen --model "$WAKE_WORK/qwen3-design" \
  --profiles 32 --batch 4 --output "$WAKE_DATA/qwen"
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/openai_samples.py \
  --output "$WAKE_DATA/openai" --workers 4
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/backgrounds.py \
  --output "$WAKE_DATA/backgrounds"
```

Generate the independent system-voice regression fixtures with
`mac_eval_samples.py --output <directory>`. These already informed v1 debugging;
new scores on them must be called regressions, not fresh untouched holdouts.

## Features, training and evaluation

Extract the human features once with the v1 `features.py --dataset --corpus
--output` pipeline and the verified LibriSpeech dev-clean archive described in
the results. Do not use legacy precomputed 20 ms/CC BY-NC features as if they
matched this experiment.

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/stream_data.py \
  --source "$WAKE_DATA/piper" --source "$WAKE_DATA/kokoro" \
  --source "$WAKE_DATA/qwen" --source "$WAKE_DATA/openai" \
  --backgrounds "$WAKE_DATA/backgrounds" --human-features "$WAKE_DATA/human-features" \
  --output "$WAKE_DATA/features-four"
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/train_stream.py \
  --upstream "$WAKE_WORK/microWakeWord" --features "$WAKE_DATA/features-four" \
  --output "$WAKE_DATA/model-four" --steps 40000 --eval-interval 10000
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/train_stream.py \
  --upstream "$WAKE_WORK/microWakeWord" --features "$WAKE_DATA/features-four" \
  --output "$WAKE_DATA/model-official64" --steps 20000 --eval-interval 10000 \
  --architecture official64 --threads 3
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/compare_candidates.py \
  --candidate "$WAKE_DATA/model-four" --candidate "$WAKE_DATA/model-official64" \
  --output "$WAKE_DATA/candidate-comparison.json"
```

Read the saved `selected` field before evaluating. The recorded four-engine run
selected `model-official64` from float validation. Evaluate that one candidate;
do not use test scores to choose between the two exports.

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/evaluate_stream.py \
  --model "$WAKE_DATA/model-official64/hey_nyx.tflite" --features "$WAKE_DATA/features-four" \
  --mac-eval "$WAKE_DATA/mac-regression" --output "$WAKE_DATA/evaluation-four"
```

Features use 16 kHz, 30 ms windows, 10 ms hops, 40 channels and one MicroFrontend
specification. Variants include genuine quiet, tiny microphone floors, colored
noise and real environmental noise, with synthetic room reflections. Full
sequences are retained so training can sample confusing speech at different
alignments. Source sampling uses square-root counts, limiting dominance by the
largest source without giving a tiny source equal weight.

Some TTS engines append long silence. For standalone wake clips without phone
timing, label metadata uses the last 10 ms RMS frame above -46 dB of peak plus
60 ms. **No PCM is cut.** This is an approximate offline label, not live speech
segmentation or an exact phonetic latency reference. Piper retains its own
generated phone-duration boundary.

The 16,225-parameter model uses a negative penalty of 20, batch 128 and continuous
validation. Checkpoint selection first requires <=0.5 observed ambient events/h
and <=1% confusing clip variants **for every validation source**, then maximizes
mean source recall. If nothing qualifies, the fallback is explicitly reported.
The exported INT8 streaming model is recalibrated on validation before the test
sets are read. P4/host numeric comparisons must use BUILTIN_REF, not XNNPACK.

The `--architecture official64` comparison uses the official notebook's layer
arguments with the same 180-feature research window. It was trained for 20,000
steps with `--eval-interval 10000 --threads 3`. This explores a different layer
configuration; it does not reproduce the upstream final-layer-duration/total
receptive-field recipe. The different step budgets also prevent attributing
the result solely to layer capacity.
The compact candidate runs for 40,000 steps. Candidate choice uses float
validation only and is saved before quantized test/regression scoring. Keep
the candidate comparison and both training histories, even if neither meets
the operating target.

Recorded metrics, artifact hashes and the acceptance decision are in
[RESULTS.md](RESULTS.md). Candidate comparison reproduces the original choice
from saved float validation histories and checks each model hash. It does not
read quantized evaluation, test audio or test scores.

These short negative exposures cannot certify <=0.5 false activations/hour.
Keep event counts, measured hours and confidence bounds. Fresh real household
and user-positive recordings plus full-app P4 integration are release gates.

## Continuing Hi ESP comparison

[PROGRESS.md](PROGRESS.md) records the active goal and current acceptance gaps.
The standalone [Hi ESP benchmark](../../wakeword_baseline/README.md) runs the
shipped engine on identical PCM, verifies every UART transfer by CRC, and
restores the original development app. Its paired Mac baseline reached 48/50
positive renderings and 0/190 shared negative renderings. These are correlated
synthetic regressions, not real-user recall. ESP-specific near words have
pronunciation uncertainty and are reported separately.

`mac_corpus.py --output` creates a new source with voice-family splits, excluding
the five old Mac regression voices. `edge_samples.py` adds Microsoft Edge TTS
as a sixth engine, using only public fixed phrases. Install
`requirements-edge.lock` in its own Python 3.12 environment and save a voice
catalogue before generation. The catalogue used by the experiment has 46
English presets; normal/multilingual variants of a named voice share a split.

```sh
"$WAKE_WORK/edge-env/bin/python" - <<'PY'
import asyncio, json, os
from pathlib import Path
import edge_tts
voices = asyncio.run(edge_tts.list_voices())
Path(os.environ["WAKE_DATA"], "edge-voices.json").write_text(json.dumps(voices))
PY
"$WAKE_WORK/edge-env/bin/python" research/wakeword_p4/training/edge_samples.py \
  --voices-json "$WAKE_DATA/edge-voices.json" --output "$WAKE_DATA/edge-corpus"
```

Append completed corpora with additional `stream_data.py --source` arguments.
Never overwrite previous feature/model directories. `--source-sampling uniform`
selects engines equally instead of the unchanged default square-root-size
sampling. Current controlled runs compare four- and five-engine data with:

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/train_stream.py \
  --upstream "$WAKE_WORK/microWakeWord" --features "$WAKE_DATA/features-five" \
  --output "$WAKE_DATA/model-uniform5" --steps 20000 --eval-interval 5000 \
  --architecture official64 --input-frames 203 --negative-weight 5 \
  --specaugment --align-stream --source-sampling uniform --threads 2
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/evaluate_stream.py \
  --model "$WAKE_DATA/model-uniform5/hey_nyx.tflite" --features "$WAKE_DATA/features-five" \
  --mac-eval "$WAKE_DATA/mac-regression" --output "$WAKE_DATA/validation-uniform5" \
  --validation-only
```

Validation-only calibration returns before reading testing/Mac audio and now
binds the selected threshold to the model SHA-256. The context/wide/max-pooling
diagnostics are in `../results/training-v3-contexts.json`; none has sufficient
validation recall to be accepted. Their test sets were not opened.

For longer continuous negatives, LibriSpeech **test-clean** is separate from
the earlier dev-clean corpus. Verify the archive against OpenSLR's official
`md5sum.txt`: `32fa31d27d2e1cad72775fee3f4849a9`. The tested archive SHA-256 is
`39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.
`continuous_negatives.py` verifies the archive checksum again, screens
transcript wake aliases, preserves a continuous frontend per speaker, and
exports both exact PCM and features for the two detectors. It generated 5.549 h
including 0.146 h of explicit 200 ms utterance separators (5.403 h of speech).
No transcript aliases were found in this archive. This source is evaluation
only; it is never silently added to training.

An audit found that the first continuous feature preparation padded each
utterance to a 10 ms hop, while its WAV contained only the explicit 200 ms
separators. The corrected preparer reads the complete WAV in whole-hop blocks,
with one frontend state and padding only the final incomplete hop. Existing
WAV/Hi ESP measurements remain valid. `rebuild_continuous_features.py --corpus
<old> --output <new>` rechecks WAV hashes, rebuilds features, and preserves the
exact benchmark manifest bytes in a new corpus. Use that corrected corpus for
matched comparisons. Whole-array versus blocked extraction was bit-exact on a
495.86 s file; the old feature stream had 17 additional frames. This does not
establish a cause for poor classifier recall.

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/continuous_negatives.py \
  --corpus "$WAKE_DATA/LibriSpeech/test-clean" --archive "$WAKE_DATA/test-clean.tar.gz" \
  --expected-md5 32fa31d27d2e1cad72775fee3f4849a9 \
  --output "$WAKE_DATA/continuous-test-clean"
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/evaluate_continuous.py \
  --model "$WAKE_DATA/model-uniform5/hey_nyx.tflite" \
  --calibration "$WAKE_DATA/validation-uniform5/threshold.json" \
  --corpus "$WAKE_DATA/continuous-test-clean" --output "$WAKE_DATA/continuous-uniform5"
```

Optional `--baseline` accepts a completed Hi ESP report for the exact same PCM,
requiring verified app restoration. Any result used to change a model makes
subsequent measurements on that audio a regression. Confidence bounds cover
the measured distribution, not all real household noise.

### Pretrained encoder experiment

The separate `transfer60` experiment reconstructs the ESPHome v2 Hey Jarvis
30/60-channel convolutional encoder from its quantized weights. Quantized
weights already contain folded batch normalization, so the float reconstruction
uses convolution biases and **does not apply batch normalization again**.
The source classifier is preserved only for numerical audits; Hey Nyx training
always resets it. An otherwise identical run without `--pretrained-model`
provides the random-initialization control.

Reviewed source: [micro-wake-word-models at 40ff33f](https://github.com/esphome/micro-wake-word-models/tree/40ff33f57f8fc6ad71a75ef085abbf742495b225),
`models/v2/hey_jarvis.tflite`, SHA-256
`21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77`.
Its pinned README, LICENSE and NOTICE explicitly cover the model weights under
Apache-2.0. Attribution: Copyright 2024–2026 Open Home Foundation,
https://www.openhomefoundation.org/. Preserve that notice with distributed
derivative artifacts. The source model's training utterances/speaker partitions
are not available here, so this experiment cannot claim that pretraining is
speaker-disjoint from our synthetic sources.

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/audit_transfer.py \
  --upstream "$WAKE_WORK/microWakeWord" --source-model "$PRETRAINED_JARVIS" \
  --features "$WAKE_DATA/features-five" --output "$WAKE_DATA/transfer-import-audit" \
  --mac-controls
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/training/train_stream.py \
  --upstream "$WAKE_WORK/microWakeWord" --features "$WAKE_DATA/features-six" \
  --output "$WAKE_DATA/model-transfer6" --steps 20000 --eval-interval 5000 \
  --architecture transfer60 --input-frames 149 --negative-weight 5 \
  --align-stream --source-sampling uniform --threads 2 --learning-rate 0.0001 \
  --pretrained-model "$PRETRAINED_JARVIS"
```

The import audit compares aligned float windows, float streaming and the exact
original INT8 artifact. It also tests three synthetic **Hey Jarvis** controls,
plus six Nyx/Mix negatives, to avoid checking only zero-output examples.
Float window/stream outputs agree to about `1.13e-6` on these controls; the
maximum dequantized-float/original-INT8 difference is about `0.043`. These
results validate this import, not Hey Nyx recognition. The one-step smoke also
completed training and streaming INT8 export. Full training/calibration and
board acceptance remain separate requirements.

Training-only duration diagnostics found that some OpenAI utterances span up to
2.89 s. A 149-frame model with a post-keyword crop offset cannot retain all of
those phrases. The next experiment uses `stream_data.py --lead-seconds 4` and
`train_stream.py --input-frames 335`, preserving every original-speed sample.
The longer lead keeps the frontend/model warm before the phrase. Because both
history and model context change, an improvement would not isolate one cause.
`--lead-seconds` defaults to the unchanged original 2 s rendering. Quiet and
noisy fixture reproduction was checked sample-for-sample after this extension.

Each new measured checkpoint is saved as `step-N.weights.h5` before selection.
`--initial-weights` can warm-start matching float weights in a fresh output
directory; this preserves the learned classifier and resets the optimizer.
Its SHA-256 is recorded. It is not an exact optimizer-state resume. Keep all
run histories and calibrate new exports on validation in fresh directories.

For new experiments use `run_training.py` with the same arguments as
`train_stream.py`. It requires a new output directory, copies the Python
sources and dependency locks into `output/source`, records source hashes and
the checkout revision in `launch.json`, then executes that private copy. This
prevents checkout edits from confusing TensorFlow's runtime source inspection.
One-step training, validation and INT8 export passed through this launcher;
this smoke is not a recognition result.

The optional `--hard-negative-fraction 0.5 --hard-negative-pool 4` experiment
scores larger **training-only** candidate pools. It takes half of each negative
batch from the highest-scoring candidates and half randomly from the remaining
pool. TTS and real ambient pools retain their original 48/48 balance. Mining
starts at step 1,000 by default; `--hard-negative-start-step 0` is suitable for
the controlled warm-start diagnostic. Positives, labels and PCM speed are
unchanged. Quantization calibration still samples the ordinary, unmined
training distribution. A two-step mined training/export smoke passed, and a
matched run without mining is required before attributing an improvement to
this option. No validation/test false wake is added to training.

`edge_samples.py --partials-only` and `kokoro_samples.py --partials-only`
generate isolated wake components and ordinary uses as a separate experiment.
The target is the **complete Hey Nyx phrase**: its full Nyx/Nix/Nicks aliases
stay positive, while `Hey` or `Nicks` alone does not contain that phrase.
Original generator defaults and voice partitions are preserved. Add these
separate corpora with `stream_data.py --source`; never overwrite prior data.

`training_negatives.py --archive <train-clean-100.tar.gz> --existing-features
<features> --output <new>` verifies OpenSLR's published MD5
`2a93770f6d5c6c964bc36631d331a522` before safe extraction. It selects up to 64
speaker IDs deterministically, rejects overlap with every existing human
speaker partition, and screens transcript wake aliases. These records are
**training only by default**. `--validation-speakers 16 --testing-speakers 16`
reserves additional disjoint groups from the same fixed permutation before
extraction or scoring. `--evaluation-corpus` also rejects overlap with an
existing continuous evaluation corpus. Do not append those held-out rows to a
training partition. The manifest retains CC BY 4.0 attribution, source hashes,
speech/separator exposure and final-hop feature padding. This is independent
of the test-clean continuous evaluation source.

`audit_negative_vocabulary.py` checks selected **validation-only** source
prefixes at an existing model-hash-bound threshold. It does not recalibrate or
read testing audio. Before new training, the rejected long335 model at its
separately frozen 0.7 diagnostic threshold triggered on 130/528 partial-word
renderings. Most were standalone `Nicks/Nix/Nyx`, confirming that this model
does not require the full phrase on those synthetic examples. This does not
prove that every long-audio false wake has the same cause. See
`../results/partial-vocabulary-before-training-v3.json`.

## Rejected/diagnostic paths

- OmniVoice's tested MLX configuration generated waveform-valid files, but all
  24 balanced pilot transcriptions were empty or inconsistent. Its 240 clips
  were excluded. This is evidence about this configuration, not every OmniVoice
  implementation.
- An initial MOSS script flattened stereo output and changed duration/speed.
  Those artifacts are invalid. The corrected script mixes the declared channel
  axis and asserts the model's sample count before resampling. The corrected
  20-clip pilot retained 12 waveform-valid clips; its positive phonetic quality
  remains unestablished and it is excluded from the four-engine run.
- `audit_speech.py` uses an independent offline tiny.en ASR engine, without a
  wake-phrase prompt. It can run a deterministic balanced pilot using `--limit`.
  Its outputs are a diagnostic quarantine, not trustworthy phonetic labels:
  it often decodes rare “Nicks/Nix” as “Next”, including in system voices.
  Never turn those ambiguous positives into negatives. The four-engine research
  run uses generator text labels; it does **not** claim every clip passed ASR or
  human listening. Raw audio and the diagnostic transcripts remain private.
- faster-whisper's filename decoder is incompatible with the installed PyAV 19
  API. Since these fixtures are already 16 kHz mono PCM16, the audit reads WAV
  samples directly and passes arrays, avoiding decoder/resampling changes.

No candidate from this directory is automatically flashed. For standalone P4
numeric probes, follow the parent README's app-only flash and guaranteed restore
procedure on the development board. The dining-room production board remains
on its normal published firmware/Hi ESP wake engine.
