# Actual idle wake/AEC rearm stability observation

The USB development board ran its unchanged 0.6.27 panel app. No firmware,
backend, credentials, saved settings or production board was changed. There
was no Mac playback, audio recording or call. This checks idle AEC/audio-lease
rearm, not WakeNet recreation, acoustic quality or custom Hey Nyx acceptance.

`tools/check_wake_rearm.py` completed 50 actual mute/unmute cycles, exit 0.
All 100 UART responses alternate muted/unmuted, all 50 cycles reached the
corresponding HTTP wake states, and all three settled samples per cycle show
listening restored. App identity, settings and activity counters were unchanged;
there was no in-cycle reboot. The complete 538 sanitized samples are retained,
followed by 15 passive samples over 44.39 seconds. The tool restored the
unmuted listener and closed UART; a later HTTP check at uptime 539 seconds
confirmed the same boot and listener state. The launchd monitor continues.

| Metric | Cycle 2 | Cycle 50 | Difference (bytes) |
| --- | ---: | ---: | ---: |
| Internal free | 278731 | 278731 | 0 |
| DMA free | 239119 | 239119 | 0 |
| Internal largest block | 126976 | 126976 | 0 |
| DMA largest block | 126976 | 126976 | 0 |
| PSRAM free | 19524400 | 19521924 | -2476 |

These are three-sample medians, not minima or allocator traces. Other cycles
include temporary drops followed by recovery; all rows and all 50 cycle medians
are preserved. Startup allocations make the initial-to-final increase misleading,
so it is not evidence that every allocation was released. The short followup
PSRAM median is 19521920 bytes. A retained difference remains unexplained.

The IDF v5.5.4 console default retains 32 history strings and frees the oldest
when full (`esp_console.h`, `linenoiseHistoryAdd`). The app uses that default.
History growth can explain part of the early retention and plateau around
16 two-command cycles; this source observation does not prove the cause of the
full PSRAM difference. No leak fix is inferred from this short run.

The first pilot completed zero cycles and exited 1: opening the USB adapter
caused a `rst:0x1 (POWERON)` before the console command could complete, despite
setting DTR/RTS false before opening. It was not restarted because observation
timed out. The new run explicitly allowed one adapter-induced startup, waited
30 seconds, rechecked UART/HTTP identity and settings, and established its new
baseline. Its captured startup also says POWERON. These two pre-test boots
must be separated from the earlier unexplained passive-monitor uptime reset.
The original pilot results and both raw UART files stay private; no node or
network identifiers are published.

Reproduction is documented in [STABILITY.md](../../../docs/STABILITY.md).
Results: [summary](../results/mac-wake-idle-rearm-50-v1.json) and
[all sanitized samples](../results/mac-wake-idle-rearm-50-samples-v1.json.gz).
This is a bounded lifecycle check. Long-call/reconnect behavior, prolonged
retained-memory trends and a qualified Hey Nyx model remain unverified.
