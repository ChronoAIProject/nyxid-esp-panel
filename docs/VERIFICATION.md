# Hardware acceptance, 2026-11-03

The acceptance target is the attached ESP32-P4 v1.3 / 16 MiB board, 32 MiB
PSRAM, OV5647, JD9365, GT911 and ES8311. Raw logs, registration credentials,
flash backups and camera images remain private outside the repository, under
the operator's dated `nyxid-esp-panel/20261103` output directory. Artifact names
below identify those records without publishing the device's surroundings or
credentials.

## Build and recovery

- Private repository: `ChronoAIProject/nyxid-esp-panel`.
- ESP-IDF v5.5.4, legacy P4 revision profile, 16 MiB flash; dependency paths are
  portable and the managed-component patches fail closed.
- All 11 host test executables pass, including independent HMAC fixtures with
  hex-decoded signing keys, binary chunk validation and WebSocket fragmentation.
- A separate clean defaults build passed (`build-clean`, v0.1.2); GitHub Actions
  also builds the actual firmware, not just the host tests. Runs `41536695098`,
  `41540218209`, `41541223809` and `41542434054` succeeded.
- Candidate v0.1.12 builds locally: application `0x66bb40` bytes within each
  `0x690000` OTA slot (148672 bytes spare). Binary SHA256:
  `45b05e93d9dc2f43b7d758fd83c5e54d55ae0bd7b1a6a742c3c19f664f18d72c`.
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
| Expiry / clear / navigation | Clear removes prior content; a 2-second text expires back to the exact baseline screenshot; unsupported native targets are rejected. |
| Three Camera cycles | Three actual open/snapshot/Back cycles pass with decodable 320×240 camera JPEGs (`acceptance-v012b/results.json`). |
| Native Settings/WLAN | Pages render, with actual connected/scanned networks. Exit exposed a PSRAM-stack NVS assertion; GUI stack is now internal. Final regression pending. |
| QR provisioning | An actual generated QR JPEG passes through the firmware QR decoder and provisioning path, schedules reboot and persists configuration. This checks decoding, not camera optics. |
| H.264/WHIP | MediaMTX receives/ffmpeg decodes 640×480 H.264; 126.8 seconds of publishing with 17 successful Nyx requests, then/including Camera preview suspension and a new receiver session on resume (`media-acceptance-v015/results.json`). No reboot or HTTP timeout in that run. |
| Publishing control latency | Nyx requests during the above run take 0.408–7.357 seconds. Some RTP loss remains; the scene is very dark. |
| Stream geometry/timing | Sensor 800×800 is center-cropped to 640×480 at (80,160), avoiding incomplete PPA scaling borders. Encoded frames are paced with backpressure and elapsed-time PTS; UART reports about 150 frames/10 seconds. |
| Network interruption | While publishing, deliberate station disconnect recovers Wi-Fi, authenticated Nyx and a new WHIP receiver session in 11.13 seconds. An actual Nyx request and newly decoded receiver frame pass (`stream-recovery-v015.json`). |
| Configuration validation | Quoted/CJK device label and backslash location roundtrip correctly; FPS 256 returns 400 without persisting. Disabled MQTT accepts clearing its old test fields (`config-acceptance-v014.json`). |
| Settings persistence | Temporary label/location and brightness 63 survive reboot. |
| Identity-preserving reset | Authenticated reset restores configuration defaults (including brightness 80), retains Wi-Fi and the Nyx identity, and reconnects successfully. |
| Full wipe / restore | Explicit UART wipe boots an unprovisioned desktop/LAN with `credentials_unset`; screenshot inspected. Saved NVS is restored with esptool hash verification; all six config getters equal the pre-test baseline and authenticated Nyx works (`persistence-final.json`, v0.1.9). |
| Raw OTA | Repeated authenticated binary updates switch between actual A/B slots; expected version boots, credentials persist and startup marks the pending image valid after 30 seconds. |
| Raw digest mismatch | Wrong SHA256 returns 422 and leaves the active boot slot unchanged. |
| URL OTA | Candidate validation pending. Download and wrong-digest rejection pass, but earlier candidates reset during flash erase/write; do not treat 202 acceptance as success. |

The current baseline has the QA WHIP target cleared, auto-start false and MQTT
disabled with its test credentials removed. Final display/media regression and
candidate URL OTA verification remain open in this working record.

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

The ordinary Nyx HTTP proxy used above still forwards JSON/Base64. The separate
machine-ingress `proxy_upload_v1` receiver is bounded to 64 KiB JPEG pushes and
has host protocol/signature tests; its machine-ingress end-to-end path has not
been exercised against the deployed server. LAN JPEG and raw OTA carry bytes
directly. Do not infer binary machine-ingress acceptance from ordinary proxy
success.

The camera scene is extremely dark/covered. Physical camera QR scanning, image
quality, high-motion streaming, physical gestures/audio perception and a
multi-hour soak are not verified. H.264 has some packet loss; the short run does
not establish zero loss or production media stability. Ethernet, a secondary
Wi-Fi AP and optional real device-code endpoints were not physically exercised.
The native video player requires SD media, which was absent during acceptance.
