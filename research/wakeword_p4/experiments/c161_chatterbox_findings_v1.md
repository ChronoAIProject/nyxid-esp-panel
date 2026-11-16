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
