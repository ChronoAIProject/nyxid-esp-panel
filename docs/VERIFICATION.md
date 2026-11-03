# Hardware acceptance, 2026-11-03

The acceptance target is the attached ESP32-P4 v1.3 / 16 MiB board, 32 MiB
PSRAM, OV5647, JD9365, GT911 and ES8311. Raw logs, registration credentials,
flash backups and camera images remain private outside the repository, under
the operator's dated `nyxid-esp-panel/20261103` output directory. Artifact names
below identify those records without publishing the device's surroundings or
credentials.

The migration is installed as v0.1.14. Build, display/navigation/Camera, reset
and actual A/B OTA checks pass. Media transitions also complete, but strict
control responsiveness and intermittent C6/Wi-Fi stability remain unresolved.

## Build and recovery

- Private repository: `ChronoAIProject/nyxid-esp-panel`.
- ESP-IDF v5.5.4, legacy P4 revision profile, 16 MiB flash; dependency paths are
  portable and the managed-component patches fail closed.
- All 11 host test executables pass, including independent HMAC fixtures with
  hex-decoded signing keys, binary chunk validation and WebSocket fragmentation.
- A separate clean defaults build passed (`build-clean`, v0.1.2); GitHub Actions
  also builds the actual firmware, not just the host tests. Runs `41536695098`,
  `41540218209`, `41541223809` and `41542434054` succeeded. OTA/Flash-fix
  source `0ca728c` passed both jobs in run `41547615184`. Final v0.1.14 source
  `4fdd57d` passed both jobs in run `41551346822`.
- v0.1.14 builds locally: application `0x66bbf0` bytes within each
  `0x690000` OTA slot (148496 bytes spare). Binary SHA256:
  `f4611883bbc063f548055c5dfe2471bab3199e34f5eb19bae3322ccb83e2f235`.
- Complete pre-migration backup: 16 MiB, SHA256
  `b25d05f4e50b90eba16b969bc82a1e516518fa64a07decba5350e1db51f09005`.
- Before reset acceptance, current same-board NVS was separately saved at its
  migrated offset `0x9000`, length `0xd2000`; SHA256
  `36948c8615fc4aa25751bc217b43de1baa56c9a8889b41994ad22e676f57b1d9`.
  This contains the new identity; the original vendor NVS must not replace it.

## Measured behavior

| Check | Evidence / result |
| --- | --- |
| Project boot | Serial identifies `nyxid-esp-panel`; display, touch, ES8311 and C6 initialize. Missing SD card is logged and the desktop still boots. |
| Nyx registration/auth | Runtime UART provisioning, WSS certificate verification, authenticated node registration, heartbeats and actual signed proxy requests succeed. |
| Text/JPEG/CJK | Actual Nyx calls display the Hello Lexa image and `Hello Lexa · 你好，世界`; the 800×1230 app-screen JPEG was decoded and visually inspected. |
| Invalid JPEG / oversize | HTTP 400 / 413; exact screenshot remains unchanged. |
| Expiry / clear / navigation | Clear removes prior content; final v0.1.14 observes a 10-second message during the Nyx request, then returns to the exact baseline screenshot. Unsupported native targets are rejected. The concurrent observation avoids mistaking response latency for an expiry failure (`acceptance-final-v014/results.json`). |
| Three Camera cycles | Final v0.1.14 completes three actual open/snapshot/Back cycles with decodable 320×240 camera JPEGs; uptime remains continuous (`acceptance-final-v014/results.json`). |
| Native Settings/WLAN | Pages render with actual connected/scanned networks. The GUI stack is internal to allow native NVS writes. Repeated navigation found paused Settings retained its WLAN subpage; v0.1.14 closes subpages on pause, rebuilds the root on resume and explicitly resolves remote `settings`. Two full cycles of Settings→WLAN→Settings→WLAN→info→provisioning→home now pass on v0.1.14 without a reset (`acceptance-final-v014/results.json`). |
| QR provisioning | An actual generated QR JPEG passes through the firmware QR decoder and provisioning path, schedules reboot and persists configuration. This checks decoding, not camera optics. |
| H.264/WHIP | Final v0.1.14 completes a 142.6-second sequence with a 30-second LAN client deadline: MediaMTX/ffmpeg decode initial, resumed and final 640×480 H.264 frames; Camera suspends the publisher and Back establishes a new receiver session. Eight actual Nyx status calls succeed; uptime is continuous (`media-diagnostic-v014/results.json`). Earlier v0.1.5 completed a strict 126.8-second / 17-request run (`media-acceptance-v015/results.json`). |
| Publishing control latency | Final v0.1.14 diagnostic Nyx latency is 0.639–19.495 seconds. The 10-second LAN-deadline run failed; this is functional media/transition evidence with known control latency, not a strict stability pass. Earlier v0.1.5 measured 0.408–7.357 seconds. Some RTP loss remains; the scene is very dark. |
| Stream geometry/timing | Sensor 800×800 is center-cropped to 640×480 at (80,160), avoiding incomplete PPA scaling borders. Encoded frames are paced with backpressure and elapsed-time PTS; UART reports about 150 frames/10 seconds. |
| Network interruption | While publishing, deliberate station disconnect recovers Wi-Fi, authenticated Nyx and a new WHIP receiver session in 11.13 seconds. An actual Nyx request and newly decoded receiver frame pass (`stream-recovery-v015.json`). |
| Configuration validation | Quoted/CJK device label and backslash location roundtrip correctly; FPS 256 returns 400 without persisting. Disabled MQTT accepts clearing its old test fields (`config-acceptance-v014.json`). |
| Settings persistence | On v0.1.13, temporary label/location and brightness 63 survive reboot (`persistence-v013-retry.json`). |
| Identity-preserving reset | v0.1.13 authenticated reset restores configuration defaults (including brightness 80), retains Wi-Fi and the Nyx identity, and reconnects successfully (`persistence-v013-retry.json`). See the intermittent C6 startup limitation below. |
| Full P4 NVS wipe / restore | Explicit UART wipe boots the desktop/LAN without Nyx identity, with `credentials_unset`; screenshot inspected. Saved NVS is restored with esptool hash verification; all six config getters equal the pre-test baseline and authenticated Nyx works (`persistence-v013-retry.json`, v0.1.13; also `persistence-final.json`, v0.1.9). C6 station settings survive this P4-only wipe. |
| Raw OTA | v0.1.13 binary moves `ota_1`→`ota_0` and validates after 30 seconds. Fourteen actual Nyx status requests during upload all succeed in 0.415–8.758 seconds; the reboot has its expected connection gap (`raw-ota-concurrent-v013.json`, `raw-ota-final-v013-status.json`). The v0.1.14 UI fix is then installed through raw OTA, switching `ota_0`→`ota_1` and reaching valid/Nyx-up (`ota-0.1.14-status.json`). |
| Raw digest mismatch | Wrong SHA256 returns 422 and leaves the active boot slot unchanged. |
| URL OTA | Actual background update switches `ota_0`→`ota_1`, boots v0.1.13 and validates after 30 seconds. Wrong SHA fails without reboot/slot switch; concurrent update returns 409. During download/write, 20 actual Nyx status calls succeed in 0.489–1.235 seconds (`url-ota-v0113.json`). |

The current baseline has the QA WHIP target cleared, auto-start false and MQTT
disabled with its test credentials removed. Final native
display/navigation/Camera regression passes on v0.1.14.
The complete media/Camera/resume sequence passes with a 30-second LAN client
deadline; the stricter 10-second deadline and intermittent transport failures
remain known limitations. The final actual LCD JPEG was decoded and visually
inspected (`final-display.jpg`).

## Repeating acceptance

`tools/accept_hardware.py --camera --native-pages` performs actual Nyx/LAN
text/JPEG/rejection/expiry checks, two Settings/WLAN/info/provisioning navigation
cycles and three Camera/snapshot/Back cycles. It checks uptime to catch resets.
It does not simulate hardware. `tools/accept_media.py` uses a real receiver,
requires a new receiver source after Camera release, decodes H.264 and measures
Nyx responsiveness. See README and each tool's `--help` for arguments.

`tools/provision_agent.py` imports encrypted node-agent credentials without
printing them. Opening this board's USB-UART pulses reset, so do not open a
second monitor or reset the board during OTA/pending startup validation.

## Limits of the evidence

One v0.1.13 identity-preserving-reset boot failed before GUI/services, with
C6 SDIO `failed to get CIS data` / `Failed to read data`, followed by repeated
slave resets (`reset-v013-private.log`). Restoring the verified same-board NVS
and restarting recovered operation. A complete repeat of reboot, both resets
and backup restoration passed on the identical binary
(`persistence-v013-retry.json`). The cause of that intermittent transport
startup failure remains unresolved; the repeat does not establish its removal.
This is distinct from the Flash watchdog failure fixed and retested during OTA.

The ordinary Nyx HTTP proxy used above still forwards JSON/Base64. The separate
machine-ingress `proxy_upload_v1` receiver is bounded to 64 KiB JPEG pushes and
has host protocol/signature tests; its machine-ingress end-to-end path has not
been exercised against the deployed server. LAN JPEG and raw OTA carry bytes
directly. Do not infer binary machine-ingress acceptance from ordinary proxy
success.

The first v0.1.13 media run decoded H.264 and completed 12 Nyx status
calls, but its Camera snapshot request exceeded the 10-second client timeout
(`media-acceptance-v013/results.json`). It did not pass the full sequence;
the serial-backed retry (`media-acceptance-v013-retry/results.json`) later
stopped on one Nyx HTTP 504 after six successful status requests. The missing
request did not appear in the board log, and the board remained online with
continuous uptime; this does not identify whether the loss was in Wi-Fi,
the Internet path or the server. Neither attempt passed the full sequence.
The v0.1.14 strict run likewise
decoded H.264 and completed successful Nyx calls before a LAN status request
exceeded its 10-second client deadline (`media-acceptance-v014/results.json`).

Later LCD reads also timed out despite serial logging a successful response
send. A short LAN ping check had 20% loss and 987–1990 ms responses, while
the gateway had no loss and 5–11 ms responses. A deliberate station reconnect
restored Wi-Fi/Nyx without reboot; the next eight board pings had no loss and
7–180 ms responses. These short samples
(`network-diagnostics-v013.json`) show a transient link problem; they do not
establish its root cause or long-term removal. Native
WLAN logs on v0.1.14 show strong signal (RSSI −26 to −29 dBm), so these
measurements alone do not identify the cause as weak signal. The final
30-second-deadline diagnostic completes the media/preview/resume sequence
without a reboot, including all eight Nyx requests, but its worst Nyx request
takes 19.495 seconds. That result does not cancel the earlier strict failures.

The camera scene is extremely dark/covered. Physical camera QR scanning, image
quality, high-motion streaming, physical gestures/audio perception and a
multi-hour soak are not verified. H.264 has some packet loss; the short run does
not establish zero loss or production media stability. Ethernet, a secondary
Wi-Fi AP and optional real device-code endpoints were not physically exercised.
The native video player requires SD media, which was absent during acceptance.
