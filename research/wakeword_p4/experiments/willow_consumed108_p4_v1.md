# Official Willow on the consumed Semaine reference inputs

Before native execution, freeze a DEFAULT-only P4 run of all 108 already
consumed Semaine clips used for the completed Muse test: Hey Willow, Hey Nicks
and Hey Muse, 36 each, original manifest order and unchanged PCM/container bytes.
Two adjacent fresh create/63-zero-prime/complete-input/destroy lifecycles per
clip give 216 executions. Preserve all carriers, variants and synthesis settings.
No new speech, source selection, gain/rate/trim change, background exposure,
threshold sweep or fresh-development/acceptance access is involved.

This closes a concrete reference-runtime gap: the original official Willow
Semaine curves came from classic ESP32 QEMU, whereas the prior P4 Willow speech
grid used twelve different Samantha/Daniel inputs. The completed P4 Muse run
does not qualify the Willow model. The same-input P4 run checks whether the weak
Semaine result also occurs with the unchanged P4 model/runtime. It cannot on its
own establish full QEMU/P4 equivalence or general speaker quality.

Reuse the exact already-built Willow threshold probe from
`results/official-hey-threshold-build-v1.json`: firmware SHA
`7b688777785d53c9c3e739a56014f2c79db84c6fe3f0347aee93b4a7bf31357b`,
expanded model 291,040 bytes, factory `wn9_heywillow_tts`, DET_MODE_95,
ESP-SR 2.6.1 / ESP-IDF 5.5.4. Authenticate its binary, ELF identity and single
embedded original compressed model against that historical build report;
do not rebuild or claim its historical C source equals a later source revision.
The DEFAULT getter must remain `3f24dd2f`; no setter is called.

`run_willow_consumed108_mac_v1.py` reuses the established Muse UART reader and
complete native validators, and the existing full-slot transaction. Its separate
reducer checks every sample/call/CRC/raw event, getter and repeat, retaining
timings and heap observations. The original Muse helpers, firmware and reports
are unchanged. A final same-baud BAUD/READY acknowledges final destruction.
UART is 921600, no tcdrain/flush; only complete board acknowledgements establish
consumption. Preserve failures and the original run handle; no automatic repeat.

Before any write, require fresh USB metadata/MAC, console-to-HTTP identity,
idle firmware 0.6.27 and explicit active slot geometry. The transaction backs up
and verifies the full slot, partition table and OTA metadata; after any probe
write attempt it restores and independently reads back all original bytes.
Require original app/service/agent/settings, idle voice, enabled/listening Hi ESP
and no serial owner afterwards. Readiness is explicitly app-only because the
original local backend is absent. No network settings, credentials or dining-room
production board operations are part of this test.

Freeze and publish `results/official-willow-consumed108-protocol-v1.json` and
the execution source before the native run:

```sh
python3 research/wakeword_p4/experiments/run_willow_consumed108_mac_v1.py freeze \
  --build-report research/wakeword_p4/results/official-hey-threshold-build-v1.json \
  --probe-bin "$ORIGINAL_THRESHOLD_BUILD/build-willow/nyx_wake_baseline.bin" \
  --source-manifest "$ORIGINAL_CONSUMED_PACKET/study/hey-development-replay-v1.json" \
  --wave-root "$ORIGINAL_CONSUMED_PACKET/audio" \
  --protocol NEW_PROTOCOL.json
```

Run with the same arguments, action `run`, the published protocol, a new private
output directory and all explicit `--port`, `--expected-mac`, `--status-url`,
`--restore-app`, `--expected-app-sha256`, `--expected-restore-sha256`, `--slot`,
`--app-offset`, `--slot-bytes`, `--partition-table-offset` parameters. Use the
existing IDF Python environment containing pyserial and esptool. Do not run
`idf.py flash` or flash any standalone partition table/bootloader.

The pre-inference reducer review uses twelve authenticated retained P4 Willow
records to construct 24 paired parser cases. It exercises the boot-prefix sync,
positive/zero events and exact repeats; seven corruptions are rejected (wrong
engine, getter, CRC, event sample clock, final acknowledgement, extra protocol,
and fixture length). This is parser evidence, not 24 new model executions.
Its result is `results/official-willow-consumed108-reducer-review-v1.json`.
Native completion, saved-UART reduction and independent restoration are still
required; a preparation or parser pass is not a recognition result.
