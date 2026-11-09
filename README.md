# nyxid-esp-panel

An ESP32-P4 panel that combines the Waveshare ESP-Brookesia desktop with NyxID
remote control, pushed text and images, on-demand camera capture, provisioning
and standalone two-way voice with a selected NyxBot.
Continuous H.264/WHIP publishing was retired in v0.3.0.

The hardware target is the attached legacy ESP32-P4 v1.3 board: 16 MiB flash,
32 MiB PSRAM, ESP32-C6 over SDIO, a Waveshare 10.1-inch 800×1280 JD9365 touch
display, an OV5647 camera, and ES8311 audio.

Implementation is governed by [SPEC.md](SPEC.md). Feature migration and hardware
acceptance are tracked there; an imported source file alone is not a completed
feature.

See [debugging and physical acceptance](docs/DEBUGGING.md) for local/production
profiles, preserving USB updates, UART, natural acoustic fixtures, repeatable
voice scenarios, service discovery, camera/display diagnosis and cleanup.
Custom **Hey Nyx** wake-word research, native Mac reproduction steps and P4
numeric results are in [the standalone research notes](research/wakeword_p4/README.md).
This experiment does not replace the production **Hi ESP** detector.

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
python -m pip install -r tools/requirements-build.txt
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

The Python requirements pin Component Manager 2.4.11, which fixes false
missing-Kconfig errors when the solver considers newer, unselected dependency
versions. The ESP-IDF image's older manager can fail even with the checked-in
dependency lock. Install the build requirements before the first configure.

The 16 MiB layout retains 840 KiB NVS, two **6.5625 MiB** application slots and
2 MiB SPIFFS. The native MP3 player remains available. Camera service startup does not allocate
a WebRTC player, H.264 pipeline or service audio codecs.
The unused SquareLine demonstration app is excluded. Bundled music contains
BGM 1, For Elise and Something; BGM 2 and Waka Waka are omitted for flash space.
Native music, drawing, calculator, Settings, camera and spectrum apps remain.

## Native menus and NyxID sign-in

Nyx Panel is the message/image app. Network, NyxID connection and device status
live in native Settings; the content app has no duplicate system menu.
Images fill the full 800×1280 canvas, preserving their aspect ratio with a
center crop. The latest text floats near the bottom with a transparent
background and a 2-pixel outline, including Chinese bitmap glyphs. `color`
sets the lettering and the existing `bg` field now sets the outline color.
Text wraps; oversized text is clipped to the canvas's safe area. Up to 20
messages are retained so an expiring caption restores the previous unexpired
one. There is no timestamp or title over the image. The native bottom-edge
navigation gesture remains available to exit the app.

- **Settings → WLAN → Scan Wi-Fi QR** opens the shared camera scanner. Standard
  `WIFI:` personal WPA/WPA2/WPA3 and open-network codes, escaped fields and
  Unicode SSIDs are supported, along with legacy `nyxprov://wifi` codes. The
  panel shows the SSID and asks before connecting. Failed/cancelled connections
  restore the previous network; successful connections are saved without reboot.
- **Settings → NyxID → Sign in / choose panel Key** displays an Agent Key login
  code and QR. On your phone, select an existing Key or create one for the panel.
  The panel stores its own child credential. Publish Device voice on the intended
  Bot in NyxID, grant that Voice connection to the panel Key, then refresh and
  select the Bot in Settings. The provider key stays on NyxID.
- **Settings → NyxID → Set up remote control connection** separately provisions
  a WSS node and daily/management services. Approve the account device code on
  your phone, then restart using the page's button. The temporary account session
  is logged out; its access/refresh tokens are never stored in NVS.
- **Settings → Device status** shows firmware, network, NyxID, media and update
  status. Scanning returns to the Settings page that opened it.

Existing provisioned panels keep their identity on upgrade and do not start a
new login automatically. See [onboarding details](docs/ONBOARDING.md) and the
[upstream feature audit](docs/NYXID_FEATURE_AUDIT.md) for the protocol, recovery
behavior and capabilities intentionally left for later.

## Voice and service permissions

Tap the microphone icon to open voice controls. Calls support **Hold to talk**
and **Auto conversation**, with echo cancellation, adjustable output gain and
separate user/Bot caption colors. The controls start collapsed and no microphone
audio is uploaded at boot. Local **Hi ESP** wake-word detection is enabled by
default and can be disabled in the voice panel. Say the wake word and continue
speaking: the panel records during connection and sends that speech before live
audio, without switching microphone readers or sending an opening greeting.
Remote daily-service calls can open two-way voice, queue
an opening notification and control microphone mute or hangup. Voice connects
directly from the board to NyxID and the provider; it needs no Mac bridge.
The existing panel Key discovers active Bot Voice services through `/api/v1/keys`.
Voice HTTP and WebSocket traffic uses the selected connection's dedicated
`/api/v1/proxy/nyxbot-voice/{service_id}` route on the configured NyxID server.
The saved selection retains the exact service ID and Bot binding across reboots.

The panel Key's grant to one Bot's Voice is independent of that Bot's grant to
the panel's services. The daily service permits camera/display, persistent
background, calls, volume/gain, input mode and brightness. Management also
permits Bot selection, credentials, networking and system operations. Grant
each service separately. Camera capture can return and display the same JPEG;
the Bot can use its existing daily grant to inspect and show that photo.

This firmware's Agent Key voice and typed camera-tool discovery require the
NyxID changes in [NyxID #2169](https://github.com/ChronoAIProject/NyxID/pull/2169).
The server must enable `assistant:voice-agent-key`, publish/grant the intended
Voice connection, and support the dedicated service proxy route. Physical audio
baseline validation used local NyxID; the service transport migration requires
matching backend and hardware validation.
Merging firmware does not deploy or enable the backend feature in production.
See [docs/VOICE.md](docs/VOICE.md) for setup, isolated local profiles and audio
behavior, and [docs/API.md](docs/API.md#daily-and-management-services-v0615)
for remote operations.

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

Open **Settings → NyxID → Set up remote control connection** for account device
login and automatic node/service setup. To import an existing node instead,
register a node and custom service
using the NyxID CLI. Route that service
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
screen device_bind
nyx-login start
nyx-login status
nyx-login cancel
reboot
```

`nyx-status` masks the token. Do not collect an unredacted serial transcript
while typing credentials: the console echoes input. Node credentials can also
be provisioned from QR codes using `nyxprov://nyx?token=...&node=...&secret=...` or
`nyxprov://full?ssid=...&psw=...&token=...&node=...&secret=...` (URL encode values).
Use native Settings → WLAN for Wi-Fi, or remote screen `scan` for QR setup.
The account device-login flow in Settings provisions actual control-node
credentials. Hardware client device-code/bootstrap identities are different and
are not accepted as WSS credentials.
UART `factory-reset` erases P4 NVS including Nyx identity; the C6 station driver
has separate storage and can retain its Wi-Fi settings. Back up current NVS
before using it. The authenticated reset API preserves both Wi-Fi and Nyx.

Alternatively, install `pyserial` and `cryptography` in a local Python
environment and import an encrypted node-agent registration:

```sh
python tools/provision_agent.py --config-dir /private/node-config \
  --port "$PORT" --slug "$SLUG"
```

## On-demand Camera service

The panel's existing authenticated NyxID service exposes:

- `GET /camera/snapshot.jpg`: take a fresh 800×1280 JPEG without navigating the LCD.
- `GET /camera`: availability, capture counts, duration and last error.
- `GET /openapi.json`: Camera operation descriptions for remote callers.

```sh
nyxid proxy request "$SLUG" /camera --output json
nyxid proxy request "$SLUG" /camera/snapshot.jpg --stream > photo.jpg
```

The CLI's `--stream` flag preserves binary stdout for this single HTTP response;
it does not start continuous camera streaming. Without it the CLI treats JPEG
bytes as text. The board starts CSI only for the request, waits for a fresh
frame after a one-second cold exposure warmup, encodes the JPEG, and releases
CSI, buffers and encoder before responding.
A second overlapping request gets 503 while the camera is busy. Failed capture
never returns an earlier image. Native Camera and QR scanning keep preview alive
only while their page is open; taking a photo during preview preserves that page.

Capture matches the `p4-brookesia` system Camera profile: RGB565, two capture
buffers, 400 kHz SCCB, and OV5647's native 800×1280 / 50 FPS sensor mode. JPEG
encoding occurs only on request (quality 50, bounded to 512 KiB). QR decoding
uses a separate 200×320 sample to keep its memory and CPU work bounded.

Display initialization retains the original 10.1-inch JD9365 timing and DSI
configuration. Full-frame double buffering replaces partial refresh. A checked
patch builds a local copy of the IDF 5.5 DPI driver and reports actual DMA
scanout selection to LVGL; it never edits the installed SDK or substitutes
periodic VSYNC for framebuffer retirement. See [display details](docs/DISPLAY.md).

`/snapshot.jpg` remains an authenticated alias. Unauthenticated LAN can read it
only while native preview is already open; the new Camera routes require NyxID.
The OpenAPI document is served through the same authenticated service; automatic
Nyx MCP tool discovery requires a separately configured, server-reachable schema.

## API and transport

Nyx Panel opens automatically at boot. It shows the long-term default image
whenever no temporary image is active. `POST /lcd/default-image` saves a JPEG
through NyxID in NVS; `GET` reports its metadata and `DELETE` restores the embedded
NyxBot image. Temporary image expiry and clear reveal the default; reboot
and app updates preserve a saved image. See [API details](docs/API.md).
At startup, the selected Bot's name and optional introduction appear for ten
seconds over the image. If no custom default image is saved, the same notice
suggests asking the Bot to set one. The notice then hides automatically;
the default image remains device-wide and does not change with Bot selection.
The system uses NyxID's black surfaces and purple controls, with a shared
[brand palette](components/panel_common/nyx_theme.h).

Nyx exposes the compatibility routes in
[`local_api.c`](components/panel_services/local_api.c): status, text/JPEG/clear,
screens and snapshots, LCD/device/Wi-Fi/LAN/MQTT configuration, reboot,
identity-preserving reset and SHA256-checked URL OTA. MQTT remains disabled.
Legacy stream start/config writes now return HTTP 410; status reports disabled.
Old stored WHIP targets and auto-start flags are ignored.

LAN recovery exposes `/status`, `/snapshot.jpg`, `/lcd/snapshot.jpg`,
`GET /lcd/screen`, raw `POST /lcd/image`, `POST /lcd/clear`, stream operations,
and stream/MQTT configuration. Other compatibility routes require Nyx.
Raw JPEG requests use `Content-Type: image/jpeg`; the initial limit is 64 KiB.
Example with the NyxID CLI:

```sh
nyxid proxy request "$SLUG" /lcd/image -m POST \
  -H 'Content-Type:image/jpeg' -d @nyxbot.jpg
nyxid proxy request "$SLUG" /lcd/text -m POST \
  -d '{"text":"NyxBot · 你好，世界","color":"#ffffff","bg":"#0d0d0d","align":"center","duration_ms":0}'
curl -H 'Content-Type: image/jpeg' --data-binary @nyxbot.jpg \
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
`info`/`operator`/`device_bind` open native Settings status/NyxID pages. `back` invokes
native navigation; unsupported targets return an error.

## OTA

Production firmware checks stable GitHub Releases automatically when idle.
See [firmware releases and automatic OTA](docs/UPDATES.md) for publishing,
hardware profiles, remote configuration and rollback behavior.


Nyx `POST /system/ota` takes `{"url":"...","sha256":"..."}` and promptly returns
202 accepted. A worker downloads and verifies the image, then reboots. Poll
`/status` → `ota.busy`, `ota.stage`, `ota.last_error` for progress/failure;
acceptance alone does not mean the update succeeded. Concurrent updates return
409. The HTTP(S) server must provide `Content-Length`; up to three redirects
are followed. The complete image is buffered in PSRAM, bounded by the inactive
slot capacity and available memory, and its digest is checked before flash is
erased. After closing the download connection, writes erase one 4 KiB sector at
a time and yield between writes. Verify the expected firmware version and valid
slot after reboot. Runtime Flash reads, writes and erases share a recursive
mutex, including NVS/status reads and mapped image verification; this prevents
the overlapping Flash access observed during status polling on the legacy P4.
For a raw LAN
image, send `POST /system/ota/image` with `Authorization: Bearer <node_token>`,
`X-SHA256: <hex_digest>` and the application binary as the body. The raw route
streams into the inactive slot and rejects a missing/mismatched digest before
changing the boot slot. URL and raw OTA share an update lock. Boot health checks
wait 30 seconds for UI/services before validating a pending image for rollback.

## Hardware acceptance

With a logged-in NyxID CLI and Pillow, run the real display, native Settings/WLAN
and three-cycle Camera acceptance against a provisioned panel:

```sh
python tools/accept_hardware.py --ip "$PANEL_IP" --slug "$SLUG" \
  --image /private/hello-lexa.jpg --output-dir /private/panel-acceptance \
  --camera --native-pages
```

On v0.2.0, add `--onboarding` to check scan/back navigation and a real login
request followed by cancellation. It never approves a login. Optional
`--wifi-qr-image /private/wifi-qr.jpg` verifies decoding and SSID confirmation
without connecting to that network. Actual phone approval, physical camera
scanning and successful Wi-Fi switching require separate checks.

`tools/accept_camera.py --ip "$PANEL_IP" --slug "$SLUG" --output-dir /private/camera-acceptance`
checks repeated authenticated photographs, concurrent control/capture calls,
resource release and native preview compatibility. Keep its images private.
`accept_media.py` refuses to run because continuous publishing is retired.

See [docs/API.md](docs/API.md) for the route inventory and
[docs/VERIFICATION.md](docs/VERIFICATION.md) for measured results and open checks.
The migration contract and completion gates are in [SPEC.md](SPEC.md).
