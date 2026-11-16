# Chatterbox development diagnostic v1

Retain C161. The distinct pinned Chatterbox Turbo engine reveals another
coverage gap at the original .91796875 threshold, smoothing 5: Nyx 5/12,
including bare 0/3 and connected commands 5/9. Willow reciprocal inputs
activate 0/12, and the four everyday negative sentences across three seeds
activate 0/12. These tiny correlated sets do not establish a daily false
activation rate or human generalization. No candidate training or threshold
selection occurred.

All 36 locked requests completed with genuine exit 0 on RTX 4090; native
24 kHz watermarked float32 waveforms and full 16 kHz PCM16 are retained under
`.local/wakeword/chatterbox-development-v1`. Total generated duration is
58.64 seconds. All outputs were finite/nonempty; no samples clipped during
PCM conversion. This verifies containers, not pronunciation. One builtin
conditioning voice, three seeds, cannot stand in for three independent voices.

Replay also completed with exit 0. Each input used a fresh frontend/model,
81600 continuous zero samples before the whole waveform and no boundary
reset. Prime events were zero. Original UINT8 traces are published in
`4090-c161-chatterbox-traces-v1.tar.gz`, with per-member SHA256 in the result.
The replay lock binds every generated native waveform, PCM and metadata before
scoring. Events in the real-audio interval count as whole-container activations;
without phoneme timing they do not prove correct wake-prefix detection or
first-command retention.

The builtin default voice and these requests remain development-only, excluded
from training. Next: evaluate the corresponding Willow positive control using
its actual DEFAULT profile and matched continuous prime; investigate separately
conditioned training voices before proposing source expansion. Do not select
spelling, seed, threshold or generation retries from these results. Practical
P4 and resource feedback remains pending in Issue #18.

## Completed official reference

Official `wn9_heywillow_tts` DEFAULT completed all36 inputs with exit0.
Every report confirms getter bits `3f24dd2f`, firmware32256 plus host49344
continuous zero prime, intact source PCM CRC and full native512 returns.
Willow bare3/3, command9/9; Nyx reciprocal0/12; everyday negatives0/12;
prime events0. UART, return bytes and reports are archived with member hashes
in `4090-chatterbox-willow-reference-result-v1.json`; archive SHA256
`3734c354e6a6aa085eacabb7c068a3fba565a2338efd29833d2bd4d7bb9e6b4a`.

This source supplies a successful Willow positive control, unlike the earlier
Semaine diagnostic. It demonstrates a C161 coverage deficit on these respective
target phrases, not universal model superiority: words, model architecture,
frontend and native event rules differ, and this is one synthetic conditioning
voice. No additional background hours or real-world false activation rate is
inferred from the twelve short negative clips.

Training-source inventory checked all rows of the existing Piper13920,
Kokoro644 and Qwen360 manifests: none exceeds5seconds. They cannot directly
meet Turbo's >5second reference requirement. Generate new coherent long
training-only synthetic reference utterances from pinned existing engines;
do not loop short clips or recycle the evaluated builtin voice. Reusing
training synthetic conditioning identities creates engine diversity, not
independent human speaker evidence.
