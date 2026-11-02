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

## Build

Use ESP-IDF **v5.5.4**, its Python environment and the P4 toolchain:

```sh
. /path/to/esp-idf/export.sh
idf.py build
cmake -S tests/host -B build-host
cmake --build build-host -j8
ctest --test-dir build-host --output-on-failure
```

The default configuration targets P4 silicon v1.x, 16 MiB flash, 32 MiB PSRAM
at 200 MHz and a 360 MHz CPU. Do not use a rev3-only image on the v1.3 board.
Host signing tests require OpenSSL development headers (`libssl-dev` on Linux,
`brew install openssl` on macOS). The dependency lock and fail-closed managed
component patches are part of the build. Run `idf.py reconfigure` after changing
component source lists. CI builds the same hardware profile.

The 16 MiB layout retains 840 KiB NVS, two **6.5625 MiB** application slots and
2 MiB SPIFFS. The native MP3 player and Opus/PCM service decoders are linked;
service audio encoding is disabled because publishing is video-only.
the unused SquareLine demonstration app is excluded. Bundled music contains
BGM 1, For Elise and Something; BGM 2 and Waka Waka are omitted for flash space.
Native music, drawing, calculator, Settings, camera and spectrum apps remain.

## Back up and migrate an existing board

Do this once when moving from a vendor/legacy partition table. Keep backups
private and use the backup of **this exact board**:

```sh
python -m esptool --chip esp32p4 --port "$PORT" --baud 921600 \
  read_flash 0 0x1000000 /private/path/before-migration-16mb.bin
python tools/extract_runtime_nvs.py /private/path/before-migration-16mb.bin \
  /private/path/runtime-nvs.bin
python -m esptool --chip esp32p4 --port "$PORT" --baud 921600 write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x2000 build/bootloader/bootloader.bin \
  0x8000 build/partition_table/partition-table.bin \
  0x9000 /private/path/runtime-nvs.bin \
  0xdc000 build/ota_data_initial.bin \
  0xe0000 build/nyxid-esp-panel.bin \
  0xe00000 build/storage.bin
```

This preserves runtime NVS while moving its offset. The installed vendor
runtime NVS was at `0x3b000`; an older layout used `0x9000`. The extractor reads
the actual table and refuses truncation. C6 Wi-Fi driver settings remain on the
C6 and are read through the shared station driver. Afterwards, use OTA or
`idf.py -p "$PORT" flash` without writing a replacement NVS image.

To restore the exact pre-migration firmware and settings:

```sh
python -m esptool --chip esp32p4 --port "$PORT" --baud 921600 \
  write_flash 0 /private/path/before-migration-16mb.bin
```

## Provision NyxID

Register a Nyx node and a custom service using the NyxID CLI. Route that service
via the new node with endpoint `http://127.0.0.1:80`, auth method `none`, and a
unique service slug. Node authentication belongs to the WSS connection; no
service credential is embedded in firmware. A device without credentials still
boots and provides its LAN recovery interface.

The UART console runs at **115200 baud**:

```text
wifi <ssid> <password>
nyx-key <node_auth_token>
nyx-node <node_uuid>
nyx-secret <64-character_signing_secret>
nyx-slug <service_slug>
nyx-status
reboot
```

`nyx-status` masks the token. Do not collect an unredacted serial transcript
while typing credentials: the console echoes input. Node credentials can also
be provisioned from QR codes using `nyxprov://nyx?token=...&node=...&secret=...` or
`nyxprov://full?ssid=...&psw=...&token=...&node=...&secret=...` (URL encode values).
Use native Settings → WLAN for Wi-Fi, or remote screen `scan` for QR setup.
The optional device-code client is disabled by default; mock mode is explicit.

Alternatively, install `pyserial` and `cryptography` in a local Python
environment and import an encrypted node-agent registration:

```sh
python tools/provision_agent.py --config-dir /private/node-config \
  --port "$PORT" --slug "$SLUG"
```

## API and transport

Nyx exposes the compatibility routes in
[`local_api.c`](components/panel_services/local_api.c): status, text/JPEG/clear,
screens and snapshots, stream start/stop/status, LCD/device/Wi-Fi/LAN/MQTT
configuration, reboot, identity-preserving reset and SHA256-checked URL OTA.
MQTT remains disabled. WHIP publishes H.264 video only.

LAN recovery exposes `/status`, `/snapshot.jpg`, `/lcd/snapshot.jpg`,
`GET /lcd/screen`, raw `POST /lcd/image`, `POST /lcd/clear`, stream operations,
and stream/MQTT configuration. Other compatibility routes require Nyx.
Raw JPEG requests use `Content-Type: image/jpeg`; the initial limit is 64 KiB.
Example with the NyxID CLI:

```sh
nyxid proxy request "$SLUG" /lcd/text -m POST -d '{"text":"Hello Lexa"}'
nyxid proxy request "$SLUG" /lcd/image -m POST \
  -H 'Content-Type:image/jpeg' -d @hello-lexa.jpg
curl -H 'Content-Type: image/jpeg' --data-binary @hello-lexa.jpg \
  "http://$PANEL_IP/lcd/image"
```

LAN JPEG transfers carry bytes directly. Legacy Nyx JSON/Base64 remains
compatible. The new receiver implements NyxID `proxy_upload` v1 signed metadata
and `NYXM` binary frames (kind 8, 16-byte UUID, 64-bit big-endian sequence,
end flag; kind 9 abort). It validates ordering, length, identity, timeout and a
64 KiB bound before adopting an image. It advertises `proxy_upload_v1` only with
a signing secret. NyxID's ordinary HTTP proxy currently buffers/base64-encodes
requests; its binary upload dispatch is limited to machine ingress. Supporting
the wire format alone does not make ordinary proxy requests use binary.

Signed legacy proxy requests and binary metadata are checked with HMAC-SHA256,
a 60-second clock window and a bounded replay guard. NTP must be synchronized.
Legacy credentials without a signing secret authenticate their WSS session but
do not enable binary uploads. Remote `home` opens Nyx Panel; `settings`/`wifi`
open native Settings, `camera`/`scan` use the shared camera, and
`info`/`operator`/`device_bind` show device status/provisioning. `back` invokes
native navigation; unsupported targets return an error.

## OTA

Nyx `POST /system/ota` takes `{"url":"...","sha256":"..."}` and promptly returns
202 accepted. A worker downloads and verifies the image, then reboots. Poll
`/status` → `ota.busy`, `ota.stage`, `ota.last_error` for progress/failure;
acceptance alone does not mean the update succeeded. Concurrent updates return
409. The HTTP(S) server must provide `Content-Length`; up to three redirects
are followed. Verify the expected firmware version and valid slot after reboot. For a raw LAN
image, send `POST /system/ota/image` with `Authorization: Bearer <node_token>`,
`X-SHA256: <hex_digest>` and the application binary as the body. The raw route
streams into the inactive slot and rejects a missing/mismatched digest before
changing the boot slot. URL and raw OTA share an update lock. Boot health checks
wait 30 seconds for UI/services before validating a pending image for rollback.

See [docs/API.md](docs/API.md) for the route inventory and
[docs/VERIFICATION.md](docs/VERIFICATION.md) for measured results and open checks.
The migration contract and completion gates are in [SPEC.md](SPEC.md).
