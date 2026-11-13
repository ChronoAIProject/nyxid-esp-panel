# Native official Hey speech pilot: frozen development diagnostic

The [protocol](official_hey_speech_pilot_v1.json) and
[preparer/auditor](official_hey_speech_pilot_v1.py) are committed before generation
or inference. This follows successful [zero-input initialization](official_hey_zero_smoke_v1.md),
but adds actual synthesized speech through the native P4 detector.

The two already consumed macOS development voices Samantha and Daniel each
generate three literal phrases, with bare and connected-command forms: 12 files.
The Nyx label deliberately uses the documented `Hey Nicks` alias; `Hi E S P`
spells the letters. Text intention does not establish correct pronunciation.
Both official arms consume the same shuffled, full waveforms. There is no
speaker playback, microphone capture, time stretch, silence trimming, gain or
augmentation. Native sample rates are converted to 16 kHz without changing speed.

The native replay keeps pinned SR2.6.1, original model bytes, DET_MODE_95,
create/zero-prime/full-PCM/destroy lifecycle and default thresholds. Getter
observations must remain .644 for Willow and .630 for Hi ESP. They are not a
shared confidence scale or evidence of equal population false-wake rates.
Cross phrases are diagnostic negatives; no common confuser or background corpus
is included. Preserve every failure and raw event, including events in leading
silence and multiple events. Do not retry a failed transport as an unreported
replacement, regenerate missing detections or select thresholds using this pilot.

Each file retains the complete native AIFF, including its original silence,
plus exactly 28800 leading and 16000 trailing zero samples after resampling.
Container endpoints are not keyword-end labels, especially for command forms;
no true wake latency is computed. Counts mean intended-text container detection
on two consumed synthetic sources, not natural-user recall or independent
acceptance. These voices/files never return to training or fresh acceptance.

## Execution

Use the existing inference environment with NumPy, SciPy and SoundFile. Preparation
writes the exact protocol and implementation before invoking `say -o` and records
their hashes, installed voice inventory identity, native source identity and
conversion details. Keep all generated files outside git.

```sh
python research/wakeword_p4/experiments/official_hey_speech_pilot_v1.py prepare \
  --output "$PRIVATE_SPEECH_FIXTURES"
```

Replay Willow first, then Hi ESP, with the existing
[guarded runner](../../wakeword_baseline/README.md), authenticated resource-enabled
probes and `--baud 115200 --require-resources --engine MODEL`. Each arm must finish
and restore successfully before the next starts. Reverify the development chip,
active slot, original app, idle state and retained restore image; production
dining-room panel is excluded. Use explicit app-only restoration only for the
already offline development configuration, preserving its selected service.

```sh
python research/wakeword_p4/experiments/official_hey_speech_pilot_v1.py audit \
  --fixtures "$PRIVATE_SPEECH_FIXTURES" \
  --build-report "$PRIVATE_RESOURCE_BUILD_REPORT" \
  --willow-report "$PRIVATE_WILLOW_RUN/report.json" \
  --hiesp-report "$PRIVATE_HIESP_RUN/report.json" \
  --expected-restored-elf "$ORIGINAL_ELF_SHA" --output "$PRIVATE_PILOT_AUDIT"
```

The audit reconstructs every PCM from its retained native source, checks WAV/PCM
hashes, transport CRC, exact sample/call counts, runtime chunk/getter, concrete
probe identity and complete restoration. It retains all per-clip results and
reports any-event and post-leading-silence-event counts separately. Detector
compute timing includes native silence and padding; it is not microphone
end-to-end latency or whole-app scheduling evidence.

## Relation to the GPU experiment

This is a Mac-owned transport/development diagnostic, not a replacement for the
new official Willow / custom same-phrase Willow / matched custom Nyx comparison.
That experiment needs a separate frozen matched training/data/calibration and
acceptance protocol before scoring. Existing formal GPU training and historical
final boundaries remain intact.

GPU's working historical reference is classic ESP32 QEMU, whose firmware selects
`hiesp` and explicitly sets its threshold; requested .630 produced getter
.62496001 there. It is distinct from native P4's .630 observation. GPU's P4 QEMU
probe never reached detection because its emulator lacked a PIE instruction.
Neither path establishes Willow detection or numeric equality on P4. Preserve
the old canonical runner and final checks when preparing an explicitly selected
new official model; authenticate actual selected model and getter rather than
silently relabeling historical Hi ESP output.

No recognition or whole-app acceptance is claimed by preparation. Results and
execution errors, if any, are recorded separately after the frozen pilot.
