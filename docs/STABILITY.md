# Long-term board stability observation

`tools/monitor_stability.py` observes an existing board through read-only LAN
`GET /status`. It does not open a serial port, reboot or flash the board, start
calls, record audio, or change the board's configuration. Use the USB development
board for research; the dining-room production board is a separate device.

## Setup and background operation

Python 3 uses only its standard library. Store the configuration outside the
repository with permissions 600, and its parent directory with permissions 700:

```json
{
  "status_url": "http://BOARD_LAN_ADDRESS/status",
  "expected_slug": "VERIFIED_DEVELOPMENT_BOARD_NODE_SLUG",
  "interval_s": 30,
  "state_dir": "/PRIVATE_ABSOLUTE_PATH/NyxPanelMonitor/stability"
}
```

Verify the intended device's node identity against the existing USB/preflight
record before configuring it. An IP address alone is insufficient: DHCP can
assign it to another board. A mismatched identity is recorded as
`identity_mismatch`, without retaining that response or its heap values. If the
board moves networks, update this private address while preserving the intended
identity. Switching boards requires a deliberate new identity configuration.

For the currently installed Mac monitor, the private config is
`~/Library/Application Support/NyxPanelMonitor/config.json`; its state directory
is `~/Library/Application Support/NyxPanelMonitor/stability`. No device addresses
or credentials are checked into this repository.

```sh
python3 tools/monitor_stability.py sample \
  --config "$HOME/Library/Application Support/NyxPanelMonitor/config.json"
python3 tools/monitor_stability.py install \
  --config "$HOME/Library/Application Support/NyxPanelMonitor/config.json"
```

`install` performs an identity-checked online preflight, copies the script into
the state directory and creates
`~/Library/LaunchAgents/ai.nyx.panel-stability.plist`. The macOS launch agent runs
at login, restarts on process exit, and uses background priority. Do not run a
second `run`/`sample` process against the same state directory while it is active.

This is observation while this Mac is awake, logged in, and able to reach the
board. It cannot provide continuous coverage during sleep, logout, power-off,
or network absence. Gaps are recorded; no power-management settings are changed.

## Read, stop, resume, or update

The report command reads saved state without querying the board:

```sh
python3 tools/monitor_stability.py report \
  --config "$HOME/Library/Application Support/NyxPanelMonitor/config.json"
launchctl print "gui/$(id -u)/ai.nyx.panel-stability"
```

Check `updated_utc` and `last_success_utc` as well as `health`: a saved `online`
report can be stale if the Mac or process stopped. `recent_samples`, minima and
trends describe the current contiguous six-hour observation window, rather than
the board's entire lifetime. `observed_restarts` counts observed uptime resets,
not every possible restart hidden inside a monitoring gap. A voice `error`
state can reflect an unavailable local backend and is separate from memory loss.

Stop background operation while keeping all evidence:

```sh
launchctl bootout "gui/$(id -u)/ai.nyx.panel-stability"
```

Resume the installed version:

```sh
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/ai.nyx.panel-stability.plist"
```

To update the installed script, stop it first, remove only the owned plist above,
then run `install` again from the updated checkout. Preserve the config, state
and logs. Installation refuses to overwrite an existing plist. It does not
automatically follow repository changes. Recent history restores on restart;
changed phase fields naturally separate old and new samples.

## Measurements and interpretation

Every 30 seconds by default, the monitor records total free heap, PSRAM free,
DMA free/largest block, and internal free/largest block. Largest-block loss can
indicate fragmentation even when total free memory looks healthy. It also
records uptime, firmware/app SHA, voice/wake/media/display/OTA state, image
allocation sizes, packet/FIFO/wake/camera counters, and network-connected status.
The HTTP response is limited to 64 KiB with a four-second network timeout.

The sanitized allowlist excludes Wi-Fi names/passwords, addresses, account/Bot/
session identifiers, captions/messages, image contents, and raw HTTP responses.
A device identity fingerprint keeps histories distinct without logging the slug.
Local `metrics-YYYY-MM-DD.jsonl` files are compressed after two days and removed
when older than 30 days; unrelated files are untouched. `status.json` is written
atomically. `service.log` and `service-error.log` describe the monitor process.
None of these local logs are automatically uploaded or committed.

Trends compare matching phases within one observed boot/app and uninterrupted
sampling window. Images and voice/media modes are separate phases. The first
five minutes of board uptime are excluded from trend analysis. Five-minute
medians need at least eight buckets spanning at least 30 minutes. A median
pairwise slope and the last three buckets must both show retained loss before a
suspicion appears in `suspected_losses`. Current diagnostic thresholds are:

| Metric | Retained loss | Negative slope |
| --- | --- | --- |
| Internal/DMA free or largest block | >= 4 KiB | >= 1 KiB/hour |
| PSRAM free | >= 64 KiB | >= 16 KiB/hour |

These are investigation triggers, not an allocator proof or release gate.
Caches, unrepresented workloads, brief allocations between samples and network
variation limit inference. A short stable sample does not establish absence of
leaks. The tool stores suspicions locally; it does not automatically notify a
person or restart the board.

For a suspected loss, inspect matching idle samples before/after actual voice,
reconnect, camera and image operations, then reproduce the responsible cycle.
Record firmware and the full trend interval. Long-call/AEC acceptance still
needs real call evidence; idle heap observation cannot substitute for it.

## Bounded idle wake/AEC rearm check

`tools/check_wake_rearm.py` temporarily mutes and unmutes an already idle,
unmuted development board. It checks the USB VID/PID/serial, then matches the
UART node identity against the private HTTP monitor config before mutating it.
Each cycle must reach `muted`, return to `listening`, and retain the same app,
boot and settings. Unexpected call, wake, camera or OTA activity aborts the test.
It restores the unmuted listener while the identified board remains idle.
The background read-only monitor continues running independently.

```sh
python3 tools/check_wake_rearm.py \
  --config "$HOME/Library/Application Support/NyxPanelMonitor/config.json" \
  --port "$DEV_PORT" --usb-serial "$DEV_USB_SERIAL" \
  --vid "$DEV_USB_VID" --pid "$DEV_USB_PID" \
  --cycles 50 --output "$PRIVATE_REARM_RUN"
```

The output directory must not exist. It contains sanitized heap/status samples
and **private raw UART**, which may contain device/network details and must not
be committed or uploaded. No audio is played or retained, calls are not opened,
and no NVS settings are written. Three settled heap samples per cycle describe
retained memory; transient allocations and leaks in other paths remain outside
this check. Mute/unmute recreates idle AEC and rearms the audio lease; it does
not recreate an unchanged WakeNet model or exercise a call/reconnect lifecycle.

Some adapters reset the board when the serial port opens even with DTR/RTS set
false before opening. The default test rejects this. For a deliberately chosen
development-board run, `--allow-adapter-reset` captures startup for 30 seconds,
verifies UART/HTTP identity and unchanged settings, and establishes the new
baseline before any cycles. Any later reboot still aborts. Keep the port open
through the entire run; classify the initial adapter-induced boot separately
from unexplained restarts in the passive monitor.

Run the monitor's privacy, lifecycle, trend and retention regressions with:

```sh
python3 -m unittest discover -s tests -p test_stability_monitor.py -v
```
