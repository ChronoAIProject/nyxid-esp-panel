# Investigating retained memory across voice calls

Use the isolated USB development panel and its own NyxID profile and credentials.
Keep the production panel out of this experiment. Read-only long-term sampling is
already supported by `tools/monitor_stability.py`; its records exclude captions,
audio, network names, account identifiers and credentials.

The production-source cleanup is isolated in
[PR #15](https://github.com/ChronoAIProject/nyxid-esp-panel/pull/15): only the two
DTLS implementations and their ownership declaration change. It was extracted
onto the production main branch and built locally with ESP-IDF 5.5.4, including
the normal P4 rev1.3 OTA/factory packaging checks. The hardware evidence below
comes from the development research firmware with the same DTLS fix; the newly
built production-profile image was not flashed during this investigation.

## Reproduction used on 2026-11-15

1. Sample `/status` once a second for 90 seconds with no call. Compare the start
   and end idle values, allowing transient network allocations.
2. Open six calls through the panel daily service (`POST /voice/open`, a fresh
   UUID `request_id` and `microphone_muted:true`). Hold each connected call for
   15 seconds, stop it, unmute and wait for wake listening, then sample 30 seconds
   idle. Only stop test-owned calls. Use an already authorized local test caller;
   do not install that caller's key on the panel.
3. Compare the same idle/listening state. Internal/DMA recovery alone does not
   imply PSRAM recovery. This test retained 143332 bytes across six calls before
   the lifecycle fix, despite internal memory returning to 219119 each time.
4. Investigate both call teardown and subsequent wake rearm. A trace ending at
   `call_task` cleanup misses allocations made while the listener is recreated.

Follow-up isolation with the DTLS context fix applied:

- Five microphone mute/unmute cycles, with no provider connection, recovered
  their PSRAM baseline within ordinary transient variation.
- Two muted provider calls with wake listening disabled still retained 45596
  bytes in total (23644 after the first, 21952 more after the second).
- A continuous system trace across two complete calls and listener rearms had
  no overflow. It showed one live replacement frontend/AEC, rather than two.

Thus listener reconstruction is not established as the residual leak source.
The confirmed 1840-byte DTLS context leak is separate: normal peer
close calls `dtls_srtp_deinit` as final owner, without subsequently freeing it.
Certificate pre-generation also needs crypto cleanup even though it has no
SRTP session. Both IDF source variants now use the same final-owner rule; only
the IDF 5/P4 variant has been built and exercised so far.

The next diagnostic compared allocated PSRAM blocks after both rearms, using
`heap_caps_walk`. It found new untraced blocks of 16896 and 4480 bytes, matching
TLS record buffers. `dtls_srtp_reset_session` called `mbedtls_ssl_setup` again
when changing roles, overwriting its old input/output buffer pointers. The role
change now frees and initializes the existing SSL context/config before setup.
Certificates and the peer's callback context retain their existing owners.
Repeated normal-firmware validation is recorded separately in the result file.

No TTS generation, Mac speaker fixture, raw microphone recording, new PCM speed,
or inference-score agreement is needed for this memory check. It does not test
recognition, AEC quality or first-utterance delivery.

## Quiet-call timeout is not a connection failure

The follow-up muted call connected in 5.32 seconds and exchanged media until
NyxID's existing 180-second user-idle limit. The authenticated session-status
endpoint confirmed `end_reason: idle` and 179 observed media seconds. The panel
did not reset and returned to unmuted wake listening; subsequent read-only
monitoring covered more than four minutes of that idle state.

This does **not** pass the originally planned 30-minute conversation check. A
silent 30-minute test is incompatible with the unmodified idle policy. Do not
disable that policy or manufacture user activity to obtain a passing result.
Long-conversation and acoustic/AEC acceptance still need representative use.
The sanitized evidence is in
`../results/mac-quiet-call-idle-stability-v1.json`.

Judge usability from actual wake detection, commands, false wakes, timing and
stability. Exact heap recovery is not required: use matched-state memory trends
to find sustained resource loss. CPU/GPU or FP32/INT8 score agreement is not an
acceptance gate.

## Longer normal-firmware lifecycle follow-up

The follow-up planned 32 muted calls, with 15 seconds connected and 20 seconds
of unmuted wake listening after each hang-up. Save the original output volume
and temporarily set it to zero to avoid repeated ringback; restore it in cleanup,
including on interruption or failure. Check the development panel identity and
only stop the call owned by the test. Do not open the serial bridge for this
test, because doing so can reset the panel.

Thirty calls connected, exchanged outgoing media, hung up and restored wake
listening. The 31st attempt returned HTTP 429 before provider creation: the
existing NyxID admission limit permits five new voice sessions per minute and
30 per rolling hour. Read-only database inspection confirmed 30 closed sessions,
all `user_ended`, and zero live owner slots. No limiter, credentials or session
records were changed to manufacture a completed 32-call result.

Across the completed calls the median connection time was 3.94 seconds and no
reset was observed. The former approximately 21 KB-per-call loss did not recur;
first-to-last idle PSRAM drift was 8128 bytes across 30 calls. This small drift
remains an observation, not an exact-recovery failure or proof that every cache
or lifetime has been resolved. Continue the existing passive monitor; do not add
more numerical-agreement gates or repeat this run just to obtain matching heaps.

Cleanup restored volume 80 and unmuted wake listening. Independent status reads
confirmed advancing capture/detection counters and the same normal firmware.
The voice state retained the 429 error, with no active or busy call; the ordinary
rolling-hour admission limit still applies to this local test caller. The result
is in `../results/mac-normal-call-lifecycle-v1.json`, with all selected original
state samples retained in the adjacent gzip. These short muted calls do not
qualify recognition, first-utterance delivery, long conversations or AEC.

### Recovery after the ordinary admission window

A separate single-call check was run after all 30 original sessions had aged
beyond the rolling hour. Start from the retained 429 error with the wake listener
running; keep the same firmware and ordinary admission policy. With greetings
disabled, save the output volume, temporarily set it to zero and open one fresh
test-owned call with the microphone muted. Observe eight seconds connected,
stop that call, unmute and restore the saved volume. Do not retry in a loop or
modify quota/session data. Keep this result separate from the incomplete 32-call
run above.

The new call connected in 4.90 seconds. Capture and outgoing media advanced;
hang-up cleared the error to `idle` and restored wake listening. Independent
status reads verified advancing capture/detection, volume 80, an unmuted
microphone, no active/busy call and the same boot. The selected original states
and script changes are in `../results/mac-rate-limit-recovery-v1.json`.
This verifies recovery after admission reopens, without extending the evidence
to acoustic wake quality, first-utterance delivery or long conversations/AEC.

## Four-hour observation of normal firmware

The existing passive monitor retained 482 selected samples from 02:00:33 to
06:02:33 UTC on 2026-11-15: four hours and two minutes, with the same normal
application and boot. After the initial installation transition there were no
additional boot boundaries, monitor errors or monitor gaps; the largest sample
interval was 38.91 seconds. The original installation boot boundary and preceding
120.64-second observation gap remain in the report as baseline events.

The first ten samples predate the monitor's audio-counter fields. The remaining
472 report both counters, starting at 02:05:43 UTC; no expected-running counter
stall was observed. Missing early counters remain unreported, rather than being
counted as a stall or as demonstrated progress. The current matched-state monitor
reports no retained-loss alert. Memory values remain descriptive.

The interval includes the short calls and control-fault tests above, the admission
rejection and separate successful recovery, followed by passive wake listening.
It does not represent four hours of conversation or uninterrupted idle. The
`four_hour_followup` in `../results/mac-normal-progress-stability-observation-v1.json`
references all selected original samples in
`../results/mac-normal-progress-four-hour-samples-v1.json.gz`.

For a later snapshot, read the existing `metrics-*.jsonl` files from the first
normal-image sample, retain subsequent transition/error records, and export the
same allowlisted sample fields without the device fingerprint. Reuse
`tools/monitor_stability.py`'s `audio_progress` handling for missing counters and
call/listener transitions. Preserve baseline events separately from later events;
do not open the serial bridge or start another monitor to obtain the snapshot.

### Longer passive follow-up

The same monitor continued through 15:41:28 UTC: 13 hours and 41 minutes from
the first normal-image sample, with 1,634 samples and the same app and boot.
There were no later boot boundaries, monitor errors or gaps. Capture and
detector counters were present in 1,624 samples, with no expected-running stall;
the first ten still remain unreported. The current monitor reported no sustained
memory-loss alert. No firmware, call, audio or serial action was added to collect
this follow-up.

The `long_followup_20261115` in the existing observation report references
`../results/mac-normal-progress-long-samples-20261115.json.gz`. Earlier snapshots
are preserved. This extends passive stability evidence after the original short
call tests; it does not establish a long conversation, wake accuracy,
first-utterance delivery or AEC quality.

The live app descriptor was also checked over read-only HTTP as
`301f013ed5bba2583dcddab85780700deb5544244a4f82fd4af97b880450a42f`.
The retained build manifest maps this image to the aligned-prefix wake model
`3ea7b067f1000516c8dfe9c6005bccdbe7360a2c26fc3d7790b70e5d26e8d3b7`.
It is the existing research image, not frozen v30 or any newly published GPU
candidate. This identity check does not qualify the model's behavior.

## Control failure and recovery with the lifetime fixes

The normal development image also completed three muted provider calls with no
speech fixture or recording. For each fault, match only the authenticated test
call's `_id`, observed `generation` and `live_slot:true` in the local Mongo
`assistant_voice_sessions` collection. Temporarily replace `control_socket_id`
with a unique diagnostic marker and set `control_socket_until` to now plus the
fault duration. Require exactly one matched document. Do not change provider
ownership, generation, authentication, service grants or other calls. Expire only
that matching marker in cleanup. The full setup is described under
"Reconnection fault test" in `../../../docs/DEBUGGING.md`.

- An eight-second control lease interruption recovered the original call in
  9.04 seconds, with the same generation and one live owner session.
- A forty-second interruption exceeded the client's existing grace period. The
  client released its failed call in 26.74 seconds; the server closed it with
  `end_reason: client_lost`, confirmed final usage and released the live slot.
- A fresh call then connected in 4.12 seconds, exchanged media and hung up
  normally. The unmuted wake listener resumed, without an observed board reset.

These were explicit remote-open calls. They exercise reconnect, failed-call
cleanup and subsequent connection after the DTLS fixes, but do not establish
wake FIFO preservation, first-utterance delivery or acoustic echo cancellation.
The sanitized result is `../results/mac-control-fault-recovery-v1.json`.

## Temporary IDF diagnostic

Apply `trace-call-and-rearm.patch` only in a disposable development checkout.
It records allocation addresses, sizes and call sites, never allocation contents.
The patch is absent from normal firmware.

`trace-two-call-cycles.patch` extends one trace across two complete calls, so
expected new listener state can be distinguished from a surviving older instance.
`trace-and-walk-two-call-cycles.patch` additionally snapshots used PSRAM blocks
at both rearms and prints new blocks absent from the allocation trace. Its walker
only copies addresses/sizes to bounded preallocated arrays; it never allocates or
logs from the heap-walk callback. Inspect both snapshot and trace overflow flags.

Use a separate SDK configuration with:

```ini
CONFIG_HEAP_TRACING_STANDALONE=y
CONFIG_ESP_SYSTEM_USE_FRAME_POINTER=y
CONFIG_HEAP_TRACING_STACK_DEPTH=6
CONFIG_HEAP_TRACE_HASH_MAP=y
CONFIG_HEAP_TRACE_HASH_MAP_SIZE=1024
```

Retain the normal development origin, model profile and original partition table.
Build the `app` target and flash only the existing app slot. Frame-pointer tracing
can enlarge the bootloader beyond the old slot; this diagnostic does not require
replacing it or moving partitions. Keep a normal app/ELF backup before flashing.

The patch allocates a persistent 4096-record trace buffer in PSRAM and the IDF
hash map uses internal RAM. Treat that diagnostic overhead separately from normal
firmware. Check `overflow=0`. Symbols in prebuilt libraries may stop unwinding;
resolve the captured PCs against the exact diagnostic ELF with `addr2line`.

The IDF 5.5 revision used here has an important coverage gap: standalone tracing
wraps `heap_caps_malloc_base`, aligned allocation, realloc and free, but not the
new `heap_caps_calloc_base`. The latter calls malloc within the same translation
unit, bypassing linker wrapping. Mbed TLS external-memory calloc allocations
were therefore absent even in a nonoverflowing trace. Absence from that trace
does not establish that all connection allocations were freed; the heap walk
exposed the missing buffers. This is why an actual repeated-call memory check
remains necessary after a source-level fix.

Do not use `heap_trace_dump_caps()` for a large UART dump on this target: its
critical-section output triggered an interrupt watchdog during this investigation.
The patch copies individual records via `heap_trace_get()` and prints them with
interrupts enabled and short yields. That diagnostic reset is separate from normal
firmware stability evidence.

The bundled media-library tracer alone is insufficient: the peer library mixes
media allocation wrappers and direct `cfree`, which can leave false unmatched
records. Its source also contains an allocation-size 860 debug trap and does not
copy variable-length backtrace data when compacting records. The system heap
tracer avoids relying on those reports as proof of leaks.

Keep UART logs private: normal boot/network logs may contain local configuration.
Export only selected allocation records and aggregate status values. Opening the
WCH USB serial bridge can reset this board even with DTR/RTS deasserted; close the
reader for passive uptime monitoring. Restore a normal firmware after diagnosis.
