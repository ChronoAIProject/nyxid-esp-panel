# Hey Nyx training: source review and experiment plan

Reviewed 2026-11-10. The phrase is **Hey Nyx**, pronounced **Hey Nicks**.
`Hey Nix` and `Hey Nicks` are positive spellings of the same sound.

## Recommended stack

Use the Apache-2.0 **microWakeWord** trainer and MicroFrontend on Apple Silicon,
export a streaming INT8 TFLite model, and run it with Espressif TFLite Micro on
the P4. Native Mac export and standalone P4 numeric compatibility are already
verified in the parent README. Recognition quality is a separate release gate.

The community [Apple Silicon trainer](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon/tree/390d255f224bf22709ff7b59f59166e4bcf0e7d1)
is a useful Mac-specific workbench: separate TTS/training environments, MPS/MLX
generation, multiple speech engines, reviewed device recordings, and detector
calibration. Its source was inspected, including `write_training_yaml.py`,
`make_features.py`, and `calibrate_detector.py`; the complete application has
not been executed here. Do not describe its README as proof of recognition
quality on our P4. GitHub reports no repository license for that pinned Mac
revision, so its code is not copied into this repository.

Its [NVIDIA sibling](https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker/tree/f5a1282dd2a796030d74f6506ece858bbc45ae7a)
also documents multiple engines and reviewed negatives. It is AGPL-3.0 and does
not require us to use Docker on the Mac. The additional wake-boundary model and
automatic recording/upload infrastructure are not necessary for this panel's
existing ordered wake FIFO.

## Practices supported by primary sources

| Practice | Source and implication for this experiment |
| --- | --- |
| Measure continuous false activations per hour, then optimize recall | [Official microWakeWord README and trainer](https://github.com/kahrendt/microWakeWord/tree/4665173cd35f1cff9a61e06fc427f124766c488e). Validation BCE or balanced clip accuracy alone does not select a usable always-listening detector. |
| Generate features over more context than the final model window | The official notebook explicitly allows PCAN/noise reduction to stabilize. Include true silence, quiet microphone floors, and audible background; avoid making all positives depend on one noise floor. |
| Cover streaming alignments | Official `SpectrogramGeneration.slide_frames` shifts feature windows. Sample negative phrases at many offsets, including partial words and before/after speech, rather than only at one phrase-end alignment. |
| Use diverse voices and real background/RIR augmentation | The official notebook uses background mixing, room responses, gain, jitter and other transforms. The Apple trainer recommends a multi-engine ensemble. Preserve consonants and inspect generated pronunciation. |
| Use an independent generation engine and real recordings for evaluation | [openWakeWord](https://github.com/dscripka/openWakeWord) discusses synthetic generation, augmentation, and realistic ambient evaluation. Its pretrained speech backbone means its synthetic-data success cannot be assumed for a tiny MCU model trained from scratch. Split speakers and sources before augmentation. |
| Calibrate the actual streaming quantized artifact | [ESPHome documentation](https://esphome.io/components/micro_wake_word/) explains threshold/window tradeoffs: higher threshold reduces false accepts but increases misses; a smaller sliding average reduces latency but can increase false accepts. A threshold cannot repair an inadequate classifier. |
| Collect reviewed real misses/false wakes for later retraining | The community trainer supports this loop. It is optional, and private audio must not silently become public training data or an automatic upload. |

The Apple trainer's example configuration uses 40,000 steps, batch 128, negative
class weight 20 and an ambient-aware model-selection metric. These are starting
points for that recipe, not universal requirements. The official microWakeWord
project itself warns that producing a usable custom model remains difficult.

## Other routes

[Espressif's official customization service](https://github.com/espressif/esp-sr/blob/master/docs/en/wake_word_engine/ESP_Wake_Words_Customization.rst)
offers TTS-only customization or larger human corpora. It is a paid, external
training route compatible with WakeNet. A public native Mac custom WakeNet
trainer was not found. No request has been sent to Espressif.

openWakeWord is useful for data and evaluation practices; its larger embedding
runtime is not the chosen MCU implementation. The panel keeps its original
16 kHz mono audio, 24 kbps Opus, 20 ms transport frames and speech speed.

## First trained candidate: rejected

The first experiment generated 7,920 Piper clips with disjoint source-speaker
splits, added LibriSpeech human negatives, trained a 16,225-parameter MixedNet
for 4,000 steps, and exported a 44,504-byte INT8 model.

At its validation-selected fallback threshold 0.994, it reached 92.81% recall on
synthetic held-out positive variants, but falsely accepted 8.65% of confusing
synthetic negative variants. An independent five-voice Mac `say` set reached
only 16% recall. It is **not accepted or installed**.

Zero events in about 1.12 hours of English audiobook negatives does not prove a
low production false-activation rate: its one-sided 95% Poisson upper bound is
about 2.68 events/hour. TV, music, Chinese background speech and actual user
positive recordings remain unmeasured.

Two diagnostic findings guide the next experiment:

1. A confusing phrase activates both the floating model and the quantized
   streaming model. Export/quantization alone does not explain the false wake.
2. Independent TTS scores change greatly when only a tiny noise floor is added.
   The first data renderer added noise even to its nominally clean examples.
   Background/PCAN distribution is a plausible contributor, not an established
   single root cause.

The v1 test and Mac set have now informed development. Further results on them
are regression measurements, **not fresh untouched test results**.

## Next training experiment

- Preserve v1 artifacts and metrics. Never overwrite a failed run with a better
  result under the same experiment name.
- Train with true quiet and several microphone/noise levels, longer frontend
  context, real environmental background, and multiple streaming alignments.
- Add a second TTS engine and keep voice identities disjoint across splits.
- Increase negative diversity and weighting; mine confusing windows from the
  training split only. Do not train on validation/test false wakes.
- Select checkpoints against continuous validation negatives plus positive
  recall; freeze detector metadata before reporting regression/test results.
- Record actual exposure and confidence bounds. An initial target of <=0.5
  false activations/hour and >=95% recall is a project gate, not an achieved
  metric. Six hours with zero events is only enough for an approximate 0.5/hour
  one-sided 95% upper bound for the measured audio distribution.
- A production release also requires real-user positive recordings, household
  negatives, and full-app P4 resource/AEC/reconnect testing. A standalone probe
  or synthetic metric cannot replace those checks.

## Data and licensing

Use compatible raw sources and one 10 ms feature hop. Legacy precomputed
microWakeWord negatives are CC BY-NC 4.0 and describe a 20 ms hop; the community
trainer's entire stock-data recipe is not automatically suitable for an
unrestricted model release. LibriSpeech/LibriTTS-R use CC BY 4.0. Check the
individual background and room-response terms, record source pins and hashes,
and keep generated/private audio outside git.

The [Kokoro ONNX implementation](https://github.com/thewh1teagle/kokoro-onnx)
is MIT and its model is Apache-2.0; it is a possible independent local generator.
[MS-SNSD](https://github.com/microsoft/MS-SNSD/tree/fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742)
has mixed dataset terms: its Freesound noises are documented as CC0, while its
DEMAND noises are CC BY-SA 3.0. Its source-code MIT license does not override
the audio licenses. Restrict background selection by source and retain hashes.
