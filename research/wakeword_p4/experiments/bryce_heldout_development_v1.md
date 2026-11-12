Kristin / Norman held-out synthetic development v1
=================================================

This archive freezes 42 original native model outputs and 126 rendered WAVs.
It contains no wake-detector scores. Use it only as development data: never add
it to training or call it an independent final test. Freeze a separate scoring
protocol before inference, including model bytes, historical thresholds,
frontend/state reset rules, event/latency windows and background policy.

Each of two voices has three native stochastic draws of seven texts:
Hey Nyx, Hey Nyx followed by “check the volume”, Hi E S P with/without the same
command, and Hey Nick / Hey Next / Hey Mix. All quiet / noise25 / combined15
renders are retained. No wake score was used to pick a source or render.

The canonical data are the saved files. Cross-platform stochastic synthesis is
not promised to reproduce ONNX output bytes; download this archive for exact
replay. The ONNX model/config byte hashes and all generation settings are in
`data/plan.json`, and source/PCM hashes are in `data/manifest.json`.

Original models: `https://brycebeattie.com/files/tts/`, downloaded from
`https://sfo3.digitaloceanspaces.com/bkmdls/{kristin,norman}.onnx[.json]`.
The author describes training from scratch on LibriVox. Full reader/utterance
manifests and possible base-corpus overlap remain unknown. These are unused
named target-voice checkpoints, not proof of physical-speaker disjointness or
an independent TTS family: both use Piper architecture.

Native speed is unchanged (`length_scale=1`). Native float mono 22050Hz source
NPYs precede any rendering. Rational resampling 320/441 preserves wall-clock
duration within one 16k sample. Common voice/draw buffers preserve 2s lead,
every original source sample, the complete 960-sample synthetic RIR tail and
at least 1s trailing padding. Shared headroom normalization avoids clipping.

Timing uses the official Piper 1.8 `piper.patch_voice_with_alignment` helper,
which appends the `/Ceil_output_0` duration tensor to the ONNX graph output.
The generator checks that original nodes and initializers are byte-identical.
Each token duration times hop256 is saved alongside IDs and phone grouping.
`generator_prefix_end_sample` locates the final keyword token before the
comma/terminal punctuation; `full_generated_source_end_sample` locates the
whole utterance end. Do not substitute command end for keyword end in latency.
Generator timing is an automatic synthesis reference, not verified acoustic
phonetic truth. Decoder audio may straddle the reference. No listening or
human recording is required for this synthetic research.

“Hi E S P” is spelled out because these models phonemize “Hi ESP” as a word.
This was fixed before generation/scoring, not chosen using detector results.

The independent auditor reconstructs each waveform directly from saved native
arrays and the fixed recipe, then requires all 126 re-encoded WAVs to match
byte for byte. The ideal float64 floor quantizer differs from libsndfile by
at most 1LSB at conversion boundaries; the WAV reconstruction is exact.
It also checks hashes, untrimmed duration, phoneme-ID duration sums, config
mapping, full buffer geometry, PCM format and absence of saturation endpoints.

With the pinned Python 3.12.13 environment from `reproduce/requirements-lock.txt`:

```sh
python reproduce/audit_bryce_heldout_development_v1.py \
  --data data --models models \
  --generator reproduce/generate_bryce_heldout_development_v1.py \
  --output independent-audit-replayed.json
```

The audit is integrity evidence, not measured intelligibility, detection
accuracy, false-wake rate, P4 performance or long-call stability. Two voices
and correlated render/draw variants are not 126 independent speakers.

No microphone recordings, production-board data, NVS, credentials, dependency
caches, final corpora or ONNX binaries are included. There is no reason to
flash a device or change the current GPU training run to receive this asset.
