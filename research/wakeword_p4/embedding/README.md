# Pretrained speech embedding experiment

This is an alternative to the rejected small MicroWakeWord classifiers. The
production/default wake word is still Hi ESP. The compact **Hey Nyx** head has
not established recognition acceptance. Standalone inference timing is not
whole-application acceptance.

## Frozen source and numerical reconstruction

Use the v0.5.1 openWakeWord release assets listed with SHA256 identities in
[`../results/embedding-feasibility-v3.json`](../results/embedding-feasibility-v3.json).
The Google embedding has 328,224 parameters. Its float graph is reconstructed
from the original weights and each operator's padding, strides, pooling and
activation options. Compare to the original interpreter **before** quantizing.
The full-window reconstruction agrees within 5.35e-5 on 28 training/control
sequences, but full-window INT8 costs about 606 ms on the P4 and is rejected.

Temporal-convolution caching retains the same weights and temporal context.
Eight new mel frames arrive per 80 ms call; four initial frames align the first
76-frame complete window. Compare only fully warmed outputs to the original
window graph: 105 checked outputs agree within 5.92e-5. Do not reset resource
variables using `reset_all_variables()` between recordings: it fails to clear
the first nine steps' resource history. Construct a new interpreter per clip.

Quiet calibration adds explicit -8/-4/0/2 histories to training-only PTQ
representatives. Streaming silence maps to -8; a whole loud clip's dB floor
would obscure this. The resulting frozen INT8 backbone has SHA256
`af0e12dd0cd6f368b0c2c24b2b5f4ff0df85af8175185d8e6bb6ef07e69e233a`.
Training-only embedding cosine mean/min are 0.9860/0.9676. This measures
embedding fidelity, not wake recall.

Use LiteRT `BUILTIN_REF` for host extraction and numeric goldens. XNNPACK and
`BUILTIN_WITHOUT_DEFAULT_DELEGATES` differ from the P4's reference-compatible
outputs by up to 36 and 9 byte counts in the audited fixture. Their speed is
not a reason to substitute them in this experiment.

## Exact integer activation folding

The float graph's 19 `LeakyRelu`/`Maximum(-0.4)` pairs are not ReLUs. The
`fuse_activations.py` helper runs each original fixed INT8 pair on all 256
possible bytes and writes a `NYX_INT8_LUT` custom node with the resulting map.
No learned weights or quantization scales change.

```sh
python research/wakeword_p4/embedding/fuse_activations.py \
  --model /private-run/embedding-stream-int8-quiet.tflite \
  --expect-pairs 19 --output /private-run/activation-lut
```

The P4 registration accepts exactly 256 opaque option bytes, one INT8 input
and one INT8 output with equal element counts. Keep the model buffer alive
for the interpreter lifetime. Invocation is:

```cpp
output[i] = table[int(input[i]) + 128];
```

The public helper reproduces the measured model bytes exactly, SHA256
`78816bed1f61f2ef5bfb2b83b056dd481849bc3fd11c9c8a29ae47281811435a`.
Exhaustive pair checks alone do not prove the complete model: the development
P4 also matches all 114 x 96 expected output bytes, including cold state.
Cached inference falls from about 79 ms to 70 ms per call with the pinned
ESP-NN 1.1.2 kernels. A tested newer ESP-NN commit is numerically exact but
slower at about 154 ms, so it is rejected.

Model-placement probes also match all114x96 bytes per case. Flash model with
internal arena costs70.41ms median; PSRAM model with internal arena69.85ms;
PSRAM model with PSRAM arena70.36ms. All use a96KiB tensor arena
(76,604bytes used) and a separate16KiB internal variable arena, on360MHz
P4/200MHz PSRAM. The native FFT scratch is allocated through the same
PSRAM-preferring heap policy. This allows an eventual whole-app test to
measure resource margin using the actual model expansion placement.

## Native frontend

`NativeMel.block()` in `native_mel.py` consumes exactly 480 preceding plus
1280 new mono samples at 16 kHz. Use the source graph's 512-point window,
257x32 mel matrix and 160-sample hop. The equations are windowed FFT,
magnitude squared, source mel weights, power log10, an 80 dB floor **per
streaming block**, then the source `/10+2` transform. Whole-recording flooring
is a different feature distribution and is not used for head training.

The P4 diagnostic implements a 512-point radix-2 FFT and the 229 nonzero
mel coefficients. Coefficients retain original float32 bits. On 33 fixed
training/control blocks it matches all 8,448 quantized source-graph inputs;
maximum float discrepancy is 6.80e-5. Median/max latency is 2.813/2.815 ms
with 6,148 bytes of scratch. The host helper also reproduces zero quantized
differences. Always regenerate goldens for the selected backbone's own
quantization scales.

## Classifier experiment and remaining acceptance

Freeze the backbone before training the wake head. Use existing source/voice
partitions, training and validation only. Each TTS clip has an original and
room/gain/noise version at its original speed, a 2.4-second lead and a
one-second tail. Human/environment features keep their original source
partitions. Heads sample only complete warmed histories, starting at frame
25 or later, so the first nine cold embeddings never enter their 16-frame
input histories.

The augmented variant has 24 reflection taps at delays80..2399samples, a
unit direct tap, Gaussian reflection weights with .2 standard deviation and
exponential decay500..1400samples. Normalize speech RMS to .04.. .20 of
full scale using a655-source-RMS floor. Add AR(1) noise with pole.5.. .96
at8..30dB SNR. Round/clip to signed16bit before the block frontend. Keep
per-source variant seeds, checksums and keyword-end metadata in private indexes.

The prespecified head is Conv2D24(3x96), ReLU, temporal MaxPool14, Dense16,
ReLU, Dense1 sigmoid. Fit 10k steps with seed726 and checkpoints
2,500/5,000/7,500/10,000; batch128 contains 32 positive, 48 TTS-negative and
48 continuous-negative histories. Sample TTS engines uniformly and continuous
negatives by duration. Negative loss weight is5. Learning rate is .001,
multiplied by .3 after5k and .1 after8k. Positive window endpoints are
.03.. .32 seconds after the existing keyword endpoint metadata; those
endpoints retain synthesis/alignment uncertainty.

Export fixed batch-one INT8 heads using 512 training histories and the
physical byte-range endpoints. Verify that input scale/zero are exactly
1/128 and0 before copying embedding bytes directly. The untrained interface
diagnostic is10,544bytes, with five operators and a mean float/INT8 output
error below.001. It is not an accuracy result.

Validation freezes a three-step score average, 80 ms cadence, a 25-step
refractory and rearming below half threshold. Use the prespecified
19-threshold grid. Prefer operating points with <=.5 continuous events/hour
and <=1% false clips in each engine/corpus subgroup, then maximize macro
recall. Freeze the checkpoint/threshold before existing paired/long
regressions. Newly reserved 6.455h English and3.126h Mandarin remain unread.
The exact grid, recipe and checkpoint tie order are recorded before fitting
in [`head-plan-v1.json`](head-plan-v1.json). `requirements.lock` records the
host package versions used for this experiment.

Remaining gates include recognition, human/acoustic comparison to Hi ESP,
actual whole-app flash/heap/CPU fit, immediate speech continuity, rearming,
quiet periods, long-call echo and reconnection. The 73 ms sum of standalone
frontend/backbone timings leaves little scheduling margin and must not be
reported as measured whole-app timing. The current raw and compressed model
sizes also require an actual unchanged-slot application build.

All device diagnostics verify the development board MAC and running slot,
flash the app only, then restore and verify the original app, selected service
and Hi ESP listener. Never run these tests on the dining-room production
board. Credentials, generated model binaries, PCM and private captures belong
outside git. No upstream model binaries are bundled here; release-asset
provenance still needs resolution before distributing a replacement.
