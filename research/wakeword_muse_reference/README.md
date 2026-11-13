# Separate official Hi/Hey Muse P4 reference

This standalone project evaluates the installed ESP-SR 2.6.1 wn10_himuse
model. Its four raw files contain both chip variants. The pinned official
`copy_wn10_model(..., 'esp32p4')` prepares only the p2 files as `wn10_data` and
`_MODEL_INFO_`; the pinned official `pack_models` then packages those two files.
The existing WakeNet9 probe and production panel remain
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
ESP-SR 2.6.1, ESP-DL 3.3.14 and the original development P4 revision configuration. Build with
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
all original inventoried model/header/library bytes and both SDK preparation
helpers against upstream b9dc43a329a68de2f72147808fc9ea29d2ace71e. It verifies
the two unpacked runtime members byte-for-byte against the original p2 files,
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

The first v1 native attempt (source d3354d0) reached READY, then aborted with
`can not find wn10_data in model wn10_himuse`; no CASE completed. It had packed
all four raw files without target preparation. Its 817708-byte expansion was a
two-chip inventory container, not the prepared P4 layout. That failed build,
protocol, UART evidence and app-only restoration remain preserved separately.
The corrected protocol/build/native schemas are v2; use a fresh v2 build root.

The correctly packed v2 still failed during creation: the precompiled WakeNet
caller allocates 144 bytes for `dl::Model`, while ESP-DL 3.3.15 requires 152 and
writes its added `m_built` member at byte 144. A heap assertion and MCU reboot
followed, before any CASE. The separate v3 pins ESP-DL 3.3.14, whose target
object size is 144, and compiles an assertion for this ABI. It preserves all
model bytes, preparation and zero cases. Use a fresh v3 build root; v2 evidence
and the 3.3.15 lock remain in the frozen history. This runtime pin applies only
to this standalone probe. Native execution must still verify the fix.

After the guarded run, extract only detector frames and independently compare
their complete order, zero-sample CRCs, raw returns and getters with the report:

    python audit.py extract --uart-private "$MUSE_NATIVE/uart-private.log" \
      --output "$MUSE_NATIVE/frames.json"
    python audit.py audit --frames "$MUSE_NATIVE/frames.json" \
      --report "$MUSE_NATIVE/report.json" --protocol "$MUSE_LOCK/protocol.json" \
      --build-report "$MUSE_LOCK/build.json" --output "$MUSE_NATIVE/audit.json"

This audit performs no inference or device access. A successful saved-frame audit
does not establish speech recognition or an independently working voice session.
