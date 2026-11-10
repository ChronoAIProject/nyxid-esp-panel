# Hi ESP developer PCM replay

Uses the same unmodified ESP-SR 2.6.1 `wn9_hiesp` model, classic ESP32 library,
normal emulated 4MB PSRAM, IDF5.5.4 and official Xtensa QEMU as the adjacent
`qemu_esp32_probe` project. Follow its build instructions, using this project,
`wakenet-esp32-replay-project`, `wakenet-esp32-replay-build` and ELF
`nyx_wakenet_esp32_replay.elf`. Set the copied sdkconfig external model parent
as documented there. Merge flash with the generated flash_args; no board needed.

The transport uses Xtensa QEMU simcall5(open),3(read),6(close). IDF's OpenOCD
semihost functions use an incompatible ABI, as its openocd_semihosting.h notes.
Only the file transport is custom: model creation, thresholds, frontend,
detect and state machine use official APIs. Read-only `input.pcm` is opened
relative to QEMU's dedicated run directory under `-semihosting`.

Generate six fresh developer probes with existing **training** voice IDs:

```bash
python3 research/wakeword_p4/training/generate_hiesp_probe.py \
  --assets .local/wakeword/kokoro-models \
  --voice-partitions .local/wakeword/hey-nyx/kokoro/generation.json \
  --output .local/wakeword/hiesp-pcm-probes-fresh
```

Replay one marked record, supplying the installed nm path:

```bash
python3 research/wakeword_p4/qemu_esp32_replay/run_replay.py \
  --qemu .local/wakeword/qemu-xtensa-release/qemu/bin/qemu-system-xtensa \
  --elf .local/wakeword/wakenet-esp32-replay-build/nyx_wakenet_esp32_replay.elf \
  --nm /absolute/path/to/xtensa-esp32-elf-nm \
  --flash .local/wakeword/wakenet-esp32-replay-build/qemu-flash.bin \
  --manifest .local/wakeword/hiesp-pcm-probes-fresh/manifest.json \
  --record af_heart-0.wav --threshold 0.63 --output /tmp/nyx-hiesp-fresh
```

Runner checks WAV/decoded-PCM SHA, exact16k mono PCM16, source count, sequential
512-sample calls, explicit final zero padding, firmware CRC32 against host zlib,
and completion stage104 after destroy/deinit. No AFE or model replacement is
performed. Input header is `<IIfI`: magic0x4e595852, original sample count,
requested threshold, reserved0; remaining bytes are little-endian PCM16.
Only the explicitly marked developer manifest is accepted by this runner.
Firmware handles `WAKENET_CHANNEL_VERIFIED=-1` separately from a positive event.
Requested0.63 yields setter success1 but getter0.62496001, so both are recorded.
Do not assume setter/getter numeric equality or compare thresholds between engines.

Events are timestamped at the **end of the consumed audio chunk**. No wall-clock
QEMU speed is treated as device latency; TTS file end is not phonetic alignment.
Two independent fresh-boot runs of af_heart-0 reproduced its event at sample44544.
All six waveforms passed CRC. Two Hi ESP prompts triggered once each; two Hey
Nicks prompts and two ordinary sentence prompts did not trigger Hi ESP. Nyx v19
on the identical original PCM triggered only its own two prompts. This establishes
a developer protocol, **not** a matched-false-wake benchmark or product quality.

Failed diagnostics are retained: initial threshold equality assumptions, treating
channel-verified as an error, and a monitor prompt startup race. The runner now
consumes the initial monitor prompt fully before requesting the completion word.
Sources/hashes, successful reports and representative failed UARTs are published.
No final set was read. Next work requires pronunciation screening, longer
continuous negative streams, validation-only operating-point calibration, and
only then frozen independent comparison.

## Continuous existing development negatives

`training/prepare_hiesp_negative.py --source SOURCE --features FEATURES --output FRESH`
checks the existing source audit, manifest, cache and feature hashes, selects only
validation, verifies one original WAV per speaker/session and original duration,
then copies the unchanged WAV and emits a development-negative-validation manifest.
It reads other split metadata only, never their audio. For DiPCo use source
`.local/wakeword/dipco-development-v1b`, features
`.local/wakeword/dipco-human-features-v1`, record `S05_U04.CH1.wav`.
Replay with the same runner, adding `--timeout-seconds 1800` for the longer stream.
Firmware prints a progress line every60 seconds of consumed audio, without
resetting frontend/model state. The default120s host deadline remains for probes.

`training/compare_hiesp_negative.py` takes the replay manifest, original features,
completed Hi ESP report and repeated `--nyx-report` directories' report.json paths.
It requires original PCM equality with the existing Nyx frontend stream after only
final10ms alignment, identical exposure, feature and score-trace hashes, and
reproduced Nyx events. It reports each native engine's trigger count and Poisson
bound, without selecting a new threshold or claiming matched general FAR.

The full2716.15625-second S05 validation recording completed: Hi ESP0 events,
Nyx v19 at0.95 also0, v23 at0.94140625 had2. Original PCM and both final padding
lengths(60 samples) matched. Zero events still gives upper3.97055/hour, not0.5/h.
This is reused development evidence, not independent acceptance.
