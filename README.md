# nyxid-esp-panel

An ESP32-P4 panel that combines the Waveshare ESP-Brookesia desktop with NyxID
remote control, pushed text and images, camera snapshots, provisioning, and
Home Assistant H.264/WHIP publishing.

The hardware target is the attached legacy ESP32-P4 v1.3 board: 16 MiB flash,
32 MiB PSRAM, ESP32-C6 over SDIO, a Waveshare 10.1-inch 800×1280 JD9365 touch
display, an OV5647 camera, and ES8311 audio.

Implementation is governed by [SPEC.md](SPEC.md). Feature migration and hardware
acceptance are tracked there; an imported source file alone is not a completed
feature.

## Provenance

- UI/BSP baseline: `waveshareteam/ESP32-P4-Platform`, commit
  `6f78fa810ec505048c1941e3d09cefc9f29129bd`, `firmware/brookesia`.
- Service migration: `voice-presence`, commit
  `caaf44a1473d1806b2249f0a9eb3d1fdfdd9cc57`,
  `src/voice_presence_ha_camera`, version 0.3.1.
- Retains the GUI close-event lock fix discovered during the baseline flash.

Device credentials, device tables, flash backups, generated configuration and
local build output are excluded from version control.

