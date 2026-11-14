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
That review was preparation, not a recognition result. The native execution
below subsequently completed using the protocol/source published in `ec31178`.

## Completed native result

The original hardware transaction completed with exit 0; it was not restarted
when observation timed out. All 216 executions completed: 29,990 input detect
calls and 13,608 zero-prime calls, with exact repeated event records. Worst
standalone detect time was 2,947 us. Each repeat detected Hey Willow in 4/36
clips, Hey Muse in 0/36 and Hey Nicks in 0/36. No threshold setter was called.

The separate saved-UART reduction exactly equals the transaction's reduction.
`compare_willow_p4_qemu_consumed108_v1.py` authenticates every original WAV/PCM
hash, sample count and source/variant/carrier/synthesis metadata against the
original QEMU execution at GPU commit `c43bdcb`. All 216 positive-event coordinate
lists, including empty lists, match QEMU DEFAULT exactly. This subset therefore
does not exhibit a P4/QEMU positive-event difference that explains Willow's weak
recognition. It does not establish arbitrary-stream equivalence, identical
cross-architecture model weights, human quality, or performance at other
thresholds. The two repeats are not independent speakers. Weak Willow recall
does not lower the goal, and Muse's earlier 36/36 target result still lacks
background FAR and independent acceptance.

The original full 6,881,280-byte app slot was restored and read back exactly,
as were the original partition table and OTA metadata. A separate post-transaction
check passed all 20 checks, including fresh HTTP firmware/app identity, valid
idle ota_0, unchanged service/agent/audio/display/update settings, enabled and
listening Hi ESP, and no serial owner. The original local voice endpoint remains
unavailable; this is app-only restoration, not voice lifecycle acceptance.

Public evidence in `results/`:

- `official-willow-consumed108-hardware-uart-v1.log.gz`: complete original UART,
  raw SHA256 `505c2b59ebf059266b22edd94fe956e202d314be580285d2d4bebc4b7126f71f`.
- `official-willow-consumed108-hardware-reduction-v1.json.gz`: complete independent
  reduction, raw SHA256 `2f8dfcea6446dbd1d7fe7881464ffc304c7e03ce51023b34bf98aba15a33cefe`.
- `official-willow-consumed108-p4-qemu-comparison-v1.json`: all 216 per-case
  comparisons and six literal/repeat aggregates.
- `official-willow-consumed108-independent-restoration-v1.json`: sanitized
  independent restoration checks and exact backup/readback hashes.
- `official-willow-consumed108-publication-v1.json`: complete file/hash bindings.

Reproduce from the retained UART without touching hardware or generating audio:

```sh
WAKE_REPLAY_DIR=$(mktemp -d)
gzip -dc research/wakeword_p4/results/official-willow-consumed108-hardware-uart-v1.log.gz \
  > "$WAKE_REPLAY_DIR/uart.log"
python3 research/wakeword_p4/experiments/run_willow_consumed108_mac_v1.py reduce \
  --protocol research/wakeword_p4/results/official-willow-consumed108-protocol-v1.json \
  --capture "$WAKE_REPLAY_DIR/uart.log" --output "$WAKE_REPLAY_DIR/reduction.json"
python3 research/wakeword_p4/experiments/compare_willow_p4_qemu_consumed108_v1.py \
  --gpu-ref c43bdcb \
  --protocol research/wakeword_p4/results/official-willow-consumed108-protocol-v1.json \
  --capture "$WAKE_REPLAY_DIR/uart.log" --output "$WAKE_REPLAY_DIR/comparison.json"
```

Both reproduced outputs were checked byte-for-byte against the published files.
No microphone/speaker operation, new source, background exposure, fresh
development or acceptance access, network change or production-board operation
was part of this experiment. No candidate is qualified or promoted.
