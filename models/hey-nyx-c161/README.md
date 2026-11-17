# Hey Nyx C161

Original INT8 model from GPU research commit `06fe353c`, without retraining or
re-export. Size: 60,952 bytes. SHA256:
`c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81`.

The profile preserves the existing board trial's threshold 0.91796875,
five-step smoothing and 309-frame context. Runtime uses 16 kHz PCM,
refractory 67 and continuous 81,600-sample startup priming. No model-specific
microphone owner, playback path, network endpoint or credentials are included.
See [voice behavior and limitations](../../docs/VOICE.md).
