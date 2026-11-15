# Investigating retained memory across voice calls

Use the isolated USB development panel and its own NyxID profile and credentials.
Keep the production panel out of this experiment. Read-only long-term sampling is
already supported by `tools/monitor_stability.py`; its records exclude captions,
audio, network names, account identifiers and credentials.

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
