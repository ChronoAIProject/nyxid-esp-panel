# Reproduce the Hey Nyx training experiments

Read [BEST_PRACTICES.md](BEST_PRACTICES.md) for primary sources and the rejected
first candidate. This is training research, not a production wake-engine update.
All generated speech is AI-generated audio, not human recordings.

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
