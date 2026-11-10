# GPU detector handoff to the P4

This is the artifact contract for a physical development probe. It does not
accept a model for release or change service permissions. The GPU track's
`teacher60` model and the Mac track's 80 ms embedding have different inputs;
their weights and cached features cannot be substituted for one another.

The latest inspected GPU candidate at `c0c4a50` is v19, model SHA-256
`4d4db51422775b400b6a5a101b5c01cc9177c95fa7fb61333795c0ca699f5151`.
Its published reports still reject overall acceptance. A future handoff must
name its own commit and hashes; do not assume this snapshot remains latest.

## Required files

Supply a downloadable archive or a shared directory containing:

- The original exported `hey_nyx.tflite` bytes.
- The matching original `training.json`, including architecture arguments,
  `context_frames`, initialization identity, source hashes and model hash.
- The original quantized validation report that fixes the selected threshold,
  score-window length, event latch/refractory and frontend identity. The v19
  report fixes threshold .95 and a five-score mean; no physical-test tuning.
- A synthetic, non-private exact-PCM fixture and expected **INT8 streaming**
  outputs, with its SHA-256, reset/warmup procedure and pinned host interpreter.
  Float-versus-quantized probabilities are a separate comparison and cannot
  establish host-versus-MCU numeric parity.

Keep original reports intact. Any normalized adapter metadata must preserve
and bind their hashes. Do not include account configuration, provider keys,
NVS/flash backups, private UART or room microphone recordings. Existing
published synthetic sources can be referenced by asset name and hash.

## Known metadata boundary

The existing Mac `prepare_runtime.py` expects `input_frames`, a calibration
`selected.threshold`, and reviewed architecture names `transfer60`,
`official64` or `wide128`. GPU v19 instead reports `context_frames: 150`,
`architecture: teacher60`, and the threshold in the quantized validation
report. Its arguments are:

```text
stride 3; first conv 30 filters, kernel 5;
pointwise filters 60,60,60,60;
repeat_in_block 1,1,1,1;
mixconv kernels [5],[9],[13],[21]; no residual connections.
```

The GPU report's validation phase is 1. This is a float-window/streaming
alignment fact, not permission to shift live PCM, duplicate frames or change
speech speed. Importing metadata must explicitly validate this topology and
frontend, preserve the original architecture name, and reject conflicting
model/training/calibration hashes. Renaming it `transfer60` to bypass the
existing allowlist would discard the distinction that needs checking.

## Physical probe sequence

1. Verify the downloaded model and report hashes before loading them. Inspect
   actual TFLite tensor types/shapes, quantization and operators; JSON metadata
   alone does not prove runtime compatibility.
2. Compare the actual 16 kHz, 30 ms window / 10 ms hop, 40-channel MicroFrontend
   configuration and scaling with the existing adapter. The 80 ms native FFT
   embedding frontend is unrelated.
3. Build and run the exact-PCM standalone P4 numerical probe on the explicitly
   identified USB development board, with an available original-app restore
   binary. Compare quantized output bytes and state resets before acoustic
   testing. In particular, do not infer fully-connected per-channel/rounding
   correctness from a float export probe.
4. Check full-app flash, internal memory, inference cadence and hardware audio
   byte balance. Retain the single existing I2S producer, processed call/preroll
   AEC/AGC and ordered FIFO. A smaller model file alone does not prove this fit.
5. Freeze an app/model-bound raw-input Hi ESP / candidate / Hi ESP acoustic
   plan before playback. Count every miss, near-word/ordinary false event and
   onset-ambiguous event. Keep source PCM speed and playback conditions fixed.
6. Restore and verify the original app, service and listener after each probe.
   Do not target the dining-room production panel. Human/room, continuous
   negatives, immediate speech, long-call/echo/reconnect and latency evidence
   remain necessary for overall acceptance.
