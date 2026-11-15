# Native sequence recognition also struggles with Joe

The existing Joe360 development cohort is difficult for the native Sherpa
GigaSpeech 3.3M keyword model as well as the compact models. The Mac run completed
all 360 original waveforms in 30.46 seconds and its original process exited 0.
It retained the GPU Jenny reference's HEY NICKS tokens and native defaults,
continuous 81600-sample prime, 480-sample input chunks and 0.66-second finish
padding. This was CPU reference inference, with no training or board action.

| Existing Joe inputs | Native HEY NICKS detections |
| --- | ---: |
| Target phrases | 3 / 72 |
| “Hey Nicks.” | 2 / 18 |
| “Hey Nicks!” | 0 / 18 |
| “…what time is it?” | 1 / 18 |
| “…turn on the light.” | 0 / 18 |
| Nearword false activations | 9 / 216 |
| Hi ESP inputs falsely recognized as Hey Nyx | 0 / 72 |

All three target detections occurred before finish padding: two with variant 2
(noise and RIR), one with variant 1 (noise). There were no target detections on
the quiet variant. The nine false activations were six “Hey Nick” and three
“Hey Nice”; four Nick events occurred in finish padding and remain counted.
There were no prime events. Complete records, including zero-event inputs and
finish events, are preserved in the
[original report](../results/mac-sherpa-existing-joe-original-v1.json.gz) and
[summary](../results/mac-sherpa-existing-joe-reference-v1.json).

The same native model previously detected 30/36 Jenny targets, while the compact
source-balanced model detected 0/72 Joe targets. These results expose a blind
spot across different representations. They do not determine whether the cause
is synthesized pronunciation, voice acoustics or the models' adaptation, and do
not establish that the generated text is correct phonetic ground truth. Native
event policies differ from the compact detector's, so these are practical
behavior observations rather than a strict decoder comparison.

Continue the original auxiliary CTC experiment and its existing full behavior
evaluation. Keep Joe as a reported development challenge; neither discard it
nor add it to training to repair this score. Representative human wake and
connected-command performance still need separate evidence. The 6 MB mixed
ONNX model remains a desktop reference, with no P4 conversion or qualification.

Reproduce from the already retained Joe render manifest and original public
model assets; all paths below are private/local examples:

```sh
uv venv --python 3.12 "$REFERENCE_WORK/env"
uv pip install --python "$REFERENCE_WORK/env/bin/python" \
  sherpa-onnx==2.0.1 sherpa-onnx-core==2.0.1 numpy==2.4.6 soundfile==0.14.0
"$REFERENCE_WORK/env/bin/python" \
  research/wakeword_p4/experiments/sherpa_existing_joe_mac_v1.py \
  --model-dir "$EXISTING_SHERPA_MODEL" \
  --manifest "$EXISTING_JOE_RENDER/replay-manifest.json" \
  --output "$REFERENCE_WORK/new-run"
```

The original model archive came from the official `kws-models` release and has
SHA256 `f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a`.
Only model/token assets were extracted. No new synthesis, recording, playback,
waveform processing or independent-final access occurred. The result is not a
new noise-background false-activation exposure or an independently qualified
wake model.
