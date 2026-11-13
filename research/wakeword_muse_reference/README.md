# Separate official Hi/Hey Muse P4 reference

This standalone project evaluates the installed ESP-SR 2.6.1 wn10_himuse
four-member layout. The existing WakeNet9 probe and production panel remain
unchanged. The factory is queried for actual chunk/rate/channel/word metadata.
No microphone, speaker, Wi-Fi, NVS or filesystems are used by the probe.

DET_MODE_95 is fixed; no threshold setter is called. Each of five independent
create/63-zero-prime/complete-input/destroy loops uses internally generated
PCM16 zeros. Actual input lengths are 1, chunk-1, chunk+1, 16000 and 16001 samples;
only the final chunk is padded. All prime/input returns, thresholds before/after
prime, exact sample/call/CRC counts, timings and phase heap snapshots are retained.
Reported heap during priming includes the explicitly reported temporary probe
buffers. A zero-input witness cannot establish recognition, FAR, phrase latency
or whole-device compatibility.

Use isolated copies of this project, dependencies.lock and the same six locked
managed components used by the WakeNet9 reference build. Pin ESP-IDF 5.5.4,
ESP-SR 2.6.1 and the original development P4 revision configuration. Build with
separate build/sdkconfig paths:

    idf.py -B "$MUSE_BUILD" -D SDKCONFIG="$MUSE_SDKCONFIG" build

The local 3 MiB partition CSV is a build capacity setting only. Never flash its
partition table or bootloader to a provisioned panel. The guarded run.py verifies
the actual USB MAC, original app/restore SHA, live idle state and actual flash
partition entry before writing only the explicitly named active app slot. It
restores in finally, verifies app/slot/service/agent/wake listener and records
app-only versus online readiness. Use fresh output directories. A polling gap
does not authorize restarting a live run.

Before native inference, publish exact source, model-member, compressed-model
and firmware identities and the protocol. Recheck original app restoration
independently afterwards. Preserve every failure and original UART log privately;
publish only sanitized result frames and hashes, never original app/model images
or private device/network settings.

Record the isolated build and exact protocol, then commit/push the source and
both resulting JSON files before running:

    python prepare.py --build-root "$MUSE_ROOT" \
      --inventory ../wakeword_p4/results/official-hey-artifact-inventory-v1.json \
      --output "$MUSE_LOCK"

The root layout is project/, build/, build-private.log. The recorder verifies
all original inventoried model/header/library bytes, all four unpacked members,
the compressed payload embedded once in the application, actual compiler/
configuration/symbols and absence of compiler warnings.

Use the pinned IDF Python with pyserial/esptool and independently verified values:

    python run.py --port "$MUSE_PORT" --expected-mac "$MUSE_MAC" \
      --status-url "$MUSE_STATUS_URL" --expected-app-sha256 "$MUSE_APP_ELF" \
      --restore-app "$MUSE_RESTORE" --restore-sha256 "$MUSE_RESTORE_SHA" \
      --probe-bin "$MUSE_PROBE" --probe-sha256 "$MUSE_PROBE_SHA" \
      --protocol "$MUSE_LOCK/protocol.json" --build-report "$MUSE_LOCK/build.json" \
      --slot "$MUSE_SLOT" --app-offset "$MUSE_OFFSET" --slot-bytes "$MUSE_SLOT_BYTES" \
      --restore-readiness app-only --output "$MUSE_NATIVE"

All recorder-bound source hashes are verified again by the runner. It also
checks that the serial port has no owner before reading the USB chip/partition.
Use online readiness only when the original application was already online.
