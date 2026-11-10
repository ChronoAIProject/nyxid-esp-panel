# Custom wake-word research: Hey Nyx

Validated on 2026-11-10 using an Apple Silicon Mac and a development ESP32-P4 revision 1.3. This is a standalone numeric probe and native training/sample-generation research, not a replacement production wake engine. The production default remains **Hi ESP**.

The selected phrase is **Hey Nyx**, pronounced **Hey Nicks**. `Hey Nix` and `Hey Nicks` are acoustic equivalents, not hard negatives. Candidate near-word negatives include `Hey Nick`, `Hey Next`, and `Hey Mix`, alongside ordinary conversation, Chinese speech, television, music, and room noise.

Actual Hey Nyx training, multi-engine sample generation, failed candidates and
reproduction steps are now recorded in [training/README.md](training/README.md).
The completed four-engine experiment and why its model is unsuitable for
production are documented in [training/RESULTS.md](training/RESULTS.md).
The compatibility smoke model in the table below remains a separate random-data
experiment; do not confuse it with a trained recognizer.

## What was actually verified

| Stage | Evidence | Limit |
| --- | --- | --- |
| Native Mac training | Python 3.12.13, TensorFlow 2.21.0; upstream MixedNet, 16,225 parameters, three training batches | Random data; this model cannot recognize Hey Nyx |
| Streaming export | INT8 input `[1,3,40]`, UINT8 output `[1,1]`, 44,368-byte TFLite file | Compatibility only |
| Native sample generation | Torch/torchaudio 2.8.0, Piper MPS fork; twelve 16 kHz mono PCM16 Hey Nyx samples | Small synthetic smoke check; no human pronunciation assessment or held-out speakers |
| P4 inference | Pretrained Okay Nabu and Mac-exported model; 976 input steps each, reference and frontend-substituted runs | Standalone task, not full GUI/AEC/Wi-Fi integration |
| Dynamic score range | Okay Nabu reference scores cover 0–255 using twelve synthetic Okay Nabu fixtures | Not a detector accuracy estimate |
| Board recovery | Original development app restored after every probe, local service reconnected, existing wake listening | Dining-room production board was not modified |

No OpenAI key, paid training service, Docker, or tensorflow-metal is needed for these checks. The original production audio format and speed were not changed.

## P4 numeric and performance results

The final experiment used ESP-IDF 5.5.4, CPU 360 MHz, esp-tflite-micro `1.3.3~1`, esp-nn `1.1.2`, and esp-micro-speech-features `1.2.3`. See `results/p4-probe.json` and `results/fixture-manifest.json`.

- Okay Nabu: 25,788-byte tensor arena used; median inference about 621 µs, p95 628 µs, initial maximum about 2.1 ms.
- Mac random-data model: 18,492-byte tensor arena used; median about 417 µs, p95 about 419 µs, maximum about 1.2 ms.
- Each run allocates a 64 KiB arena plus a separate 4 KiB resource-variable arena. Used tensor bytes do not describe total runtime memory. The final main task stack had 13,844 bytes free out of 16 KiB.
- Frontend: 198 frames from a two-second deterministic chirp/noise fixture; about 58 ms total processing. One of 7,920 raw values differs by one unit: frame 3/channel 39, P4 496 versus Mac 495. Corresponding INT8 input is 63 versus 62. **The frontend is not bit exact across these builds.**
- Feeding those board-produced features for the first 66 steps does not change any final model score in this fixture. Remaining inputs are shared host-generated vectors, including synthetic speech features; the speech PCM itself was not processed by the board frontend in this experiment.
- Against Mac **BUILTIN_REF** kernels, both models have zero output differences across all 976 steps in both runs.

### Failed checks and the reference correction

The initial strict frontend comparison failed on the single raw difference above. A second check confirmed that this also crosses one INT8 quantization boundary. Compiling the same embedded frontend sources on the Mac produced the same values as pymicro-features, so this was not explained by simply using a different frontend repository. The exact arithmetic/library cause has not been isolated. The final probe explicitly reports the difference and accepts at most one raw/quantized unit; it also compares resulting model outputs. This bound is numeric evidence for these fixtures, not proof of equivalent wake accuracy.

The first 256 low-score inputs were insufficient to expose the desktop delegate difference. Adding speech fixtures produced 47 Okay Nabu output differences, maximum 10/255, against Mac's default XNNPACK interpreter. Running the same inputs on the Mac with canonical BUILTIN_REF kernels reproduced that exact difference count and maximum. With BUILTIN_REF golden outputs, the P4 then matched every output. `prepare.py` therefore explicitly disables XNNPACK. Do not relax the score comparison to hide a delegate mismatch.

## Reproduce native Mac training and export

Use a separate workspace for downloads and generated/private data. Paths below are examples; they contain no device credentials.

```sh
export WAKE_WORK="$PWD/.local/wakeword"
mkdir -p "$WAKE_WORK"
uv venv --python 3.12 "$WAKE_WORK/tf-env"
uv pip sync --python "$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/requirements-smoke.lock
git clone https://github.com/kahrendt/microWakeWord.git "$WAKE_WORK/microWakeWord"
git -C "$WAKE_WORK/microWakeWord" checkout 4665173cd35f1cff9a61e06fc427f124766c488e
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/mac_smoke.py \
  --upstream "$WAKE_WORK/microWakeWord" --output-dir "$WAKE_WORK/mac-output"
```

`mac_smoke.py` trains random data, exports a streaming model, and exercises it locally. A passing result does not mean a keyword was trained. `requirements-smoke.lock` records the tested native environment, without tensorflow-metal.

## Reproduce native Piper samples

Keep TTS in a separate virtual environment so its dependencies do not change the TensorFlow environment.

```sh
uv venv --python 3.12 "$WAKE_WORK/tts-env"
uv pip sync --python "$WAKE_WORK/tts-env/bin/python" research/wakeword_p4/requirements-tts.lock
git clone --branch mps-support https://github.com/kahrendt/piper-sample-generator.git "$WAKE_WORK/piper"
git -C "$WAKE_WORK/piper" checkout b5b36e266576693ea47db129fbfe6bbc06164762
git -C "$WAKE_WORK/piper" apply --unidiff-zero "$PWD/research/wakeword_p4/piper-torch28.patch"
curl --fail --location --output "$WAKE_WORK/piper/models/en_US-libritts_r-medium.pt" \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
"$WAKE_WORK/tts-env/bin/python" research/wakeword_p4/tts_smoke.py \
  --upstream "$WAKE_WORK/piper" --output-dir "$WAKE_WORK/hey-nyx"
```

The small patch explicitly loads the upstream full-object checkpoint with `weights_only=False` on Torch 2.8. `tts_smoke.py` verifies its SHA-256 **before** deserialization. Do not apply this loading mode to arbitrary checkpoints. The phonemizer produces identical phones for Hey Nyx/Nicks/Nix. The twelve samples use the generator's initial speaker blends, not a balanced or independent speaker dataset.

For the optional numeric probe speech fixtures, run the same script with `--phrase 'Okay Nabu.'` and a separate empty output directory. These fixtures exercise a known reference detector; they do not train Hey Nyx.

## Reproduce the standalone P4 probe

```sh
"$WAKE_WORK/tf-env/bin/python" research/wakeword_p4/prepare.py \
  --smoke-model "$WAKE_WORK/mac-output/smoke-streaming-int8.tflite" \
  --speech-dir "$WAKE_WORK/okay-nabu"
cd research/wakeword_p4
source "$IDF_PATH/export.sh"
idf.py build
```

Omit `--speech-dir` for the original 256-step low-score check, which provides weaker coverage. The public Okay Nabu model download is pinned and hash checked. Generated headers, model binaries, downloads and private recordings are excluded from git.

**Do not run `idf.py flash` on a provisioned panel.** This standalone project's partition layout differs from the panel firmware. The experiment used app-only flashing to the verified development board's active `ota_0` offset `0xe0000`; bootloader, partition table, NVS, filesystem and OTA metadata were retained. That offset must not be assumed for another layout or active slot.

The actual flash/restore procedure was:

1. Save the existing known-good app binary and its ELF SHA, read the board's `/status`, verify idle voice, active OTA slot and USB MAC. Use the development board only.
2. With no other UART owner, flash only the probe app to that verified active app offset using esptool `write_flash`; never erase all flash or write NVS.
3. Capture 115200-baud UART output; reset once if necessary to capture the startup. Collect `PROBE_FRONTEND`, `PROBE_MODEL` and `P4_WAKE_PROBE_DONE`. UART capture has a 30-second deadline.
4. In a `finally` block, close UART, flash the original app back to the same offset, and verify `/status.app_sha256`, local service `up`, unchanged voice service selection and wake `listening`.
5. Preserve failed results as well as the final result. Do not leave the standalone numeric probe installed.

## Production integration still to do

Use microWakeWord on Mac for training and Espressif TFLM on P4 for inference. WakeNet9 weights cannot be replaced with a TFLite file: these are different runtimes. The existing WakeNet engine must be replaced or selected at build time, not simply supplemented without accounting for flash.

The fixed development app is 6,811,664 bytes in a 6,881,280-byte OTA slot (69,616 bytes free). Its compressed Hi ESP weights occupy 183,729 bytes. The standalone probe links about 106,863 flash bytes in TFLM/frontend/ESP-NN archives, plus its models, registrations, C++ support and test code. These archive figures are not a final firmware delta. Build and measure the full replacement on both P4 revisions before claiming it fits. No partition-layout or OTA compatibility change is needed for the research probe.

Feed the new frontend from the **existing single continuous 16 kHz I2S → AEC → AGC producer**. Accumulate 160-sample frontend hops and run the `[1,3,40]` streaming model every three feature frames. Keep the existing ordered wake FIFO and call lifecycle. Do not add a second mic reader, replay the same preroll twice, change Opus speed/format, locally segment user speech, mute the entire playback period, or filter transcripts to conceal echo. Keep streaming state per engine instance and reset/rearm it deliberately.

Before enabling Hey Nyx by default:

1. Build a substantially larger positive set with broad independent speaker/voice coverage, keyword followed immediately by a request, volume/distance variations, and room impulse/noise augmentation. Listen to samples; waveform validity and phoneme spelling alone are insufficient.
2. Split by source speaker/generator/room before augmentation. Reserve human recordings and separate environments for evaluation. Include Nyx/Nicks/Nix as equivalent positives rather than contradictory labels.
3. Use ordinary speech, near words, Chinese speech, background and Bot speaker audio as negatives. Keep training, threshold tuning and final evaluation separate. Report missed wakes and false activations per hour on continuous held-out audio, including confidence intervals and actual hours.
4. Keep one feature specification across all splits: 16 kHz, 30 ms window, 10 ms hop, 40 channels, 125–7500 Hz, matching noise reduction/PCAN/log settings and quantization. The legacy precomputed dataset card describes a **20 ms** hop: do not silently mix it with this 10 ms frontend.
5. Validate full GUI, camera, Wi-Fi, AEC, long calls, short Bot replies, reconnects, wake immediately followed by speech, volume/mute and automatic/hands-on modes. Measure stack, internal RAM, CPU and flash in the real app, not just this probe.

The readily downloadable `kahrendt/microwakeword` negative feature collection is marked **CC BY-NC 4.0**. Do not treat it as an unrestricted production training asset. Prefer appropriately licensed raw speech/background sources and extract consistent features; keep attribution and split provenance. The Piper LibriTTS-R model card lists CC BY 4.0 for its dataset. Check model, generator and data terms before distributing trained artifacts. No new model weights or recorded speech are published by this research directory.

## Primary references and pins

- [microWakeWord trainer/notebook](https://github.com/kahrendt/microWakeWord/tree/4665173cd35f1cff9a61e06fc427f124766c488e)
- [Piper Mac MPS branch](https://github.com/kahrendt/piper-sample-generator/tree/b5b36e266576693ea47db129fbfe6bbc06164762)
- [Piper LibriTTS-R model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/libritts_r/medium/MODEL_CARD)
- [ESPHome P4 streaming integration](https://github.com/esphome/esphome/tree/1add061d9430c9e170c8b79a3a8d3ba9d2cc8730/esphome/components/micro_wake_word) (uses a larger P4 task stack)
- [Reference models and explicit Apache 2.0 weights license](https://github.com/esphome/micro-wake-word-models/tree/40ff33f57f8fc6ad71a75ef085abbf742495b225)
- [Espressif TFLite Micro](https://github.com/espressif/esp-tflite-micro)
- [Legacy feature dataset card](https://huggingface.co/datasets/kahrendt/microwakeword/blob/main/README.md)

Upstream downloads and all generated audio remain outside committed source. Numeric result summaries intentionally omit network names, device addresses, identities, credentials and raw histories.
