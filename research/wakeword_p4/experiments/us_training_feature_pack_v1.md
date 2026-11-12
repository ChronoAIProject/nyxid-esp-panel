# US training feature pack v1: freeze before rendering or scoring

Hypothesis for a later fitting experiment: broad old-training negative separation
coexists with actual-float confusion on held-out synthesis identities; adding
fixed en_US synthesis coverage may improve generalization. This is not a
quantization-only or another temporal-only intervention. The three-preset pilot
is training-only. No held-out old/new development or final example is added.

Input is exactly all108 original rows, manifest SHA
350315d41263ef24d593fa46b1291ccc3ef21c95682840d65ccfcf1f1ea7369f,
with independent full hash/resample/PCM audit passing including float32 dtype.
No score-based selection, regeneration, trimming, context cropping of sources,
retry/replacement, or addition of observed development failure identities.
Original native float/resampled float/PCM preserved. Draws are not independent
speakers. Lessac/Ryan/LJSpeech are three synthesis presets, not certified
independent people/pretraining corpora.

Render exactly3 variants per row=324 attempts: original PCM waveform, Gaussian
noise25dB, and discrete synthetic RIR(delays0/160/480/960samples,
amplitudes1/.25/-.12/.06) plus Gaussian noise15dB. Each uses complete source,
including full RIR tail. Scale signal RMS to.08 before noise, then apply only
necessary global peak headroom gain min(1,.95/peak); these gains are explicit
augmentation, not repairs to the frozen original generation. Save augmented
PCM16 WAV, pre-quantization float64 arrays and frontend raw uint16 arrays.
Lead32000/tail16000samples. Seed first16 hex SHA256(original filename +
':variant:us-training-feature-v1'). Frontend consumes serialized PCM values,
not an unrecorded alternative float waveform. No silence removal.

Mechanical gates on every rendered attempt: finite float, mono/nonempty16k
PCM16, nonzero signal, no serialized endpoints, full output length, file/PCM/
array hashes. Verify and report exact pinned converter roundtrip. Generate raw
features using existing Linux frontend and hash its code/dependencies. Preserve
every attempt; no filtering based on audio quality or model output.

For context150, ledger-exclusive end e=ceil((32000+signal_samples)/160)-1,
crop raw[e-150:e]. Require begin=(e-150)*160<=32000 and
stop=(e+1)*160>=32000+signal_samples, e>=150 and e<=len(raw).
If a complete container does not fit, record unsupported without truncation or
relabeling; save all waveform/raw artifacts. Report all unsupported attempts
by preset/phrase/speed/draw/variant. This is a conservative process ledger,
not physical receptive field or phonetic truth. Store eligible complete crops
and labels in training.npz, rows carrying every source and augmentation link.
All four phrase/preset cells must remain nonempty to enable a future sampler.

No wake-model inference or training in this feature-pack stage. A later v44
plan must separately freeze slots/source mass/seeds and preserve original
remaining RNG/ambient/checkpoint/export behavior, with smoke audit before
formal fitting. Original inherited training quality gaps remain; a new passing
pack cannot certify all historical data. Promotion still requires the original
quality, positive/negative, INT8 and FAR upper95<=.5/h gates; v30 retained.
