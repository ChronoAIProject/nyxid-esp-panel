# NyxID ESP Panel — migration specification

Status: implementation in progress. Date: 2026-11-03.

## 1. Outcome

Create `ChronoAIProject/nyxid-esp-panel` and ship one firmware that retains the
working Brookesia desktop, Settings, touch, camera, music and audio support while
bringing the previous HA camera firmware's useful remote-device functionality
into that desktop. The panel must boot without a compiled-in device registry,
accept NyxID control, display the requested text and JPEG, serve diagnostics and
snapshots, provision its credentials, and publish H.264 to a configurable WHIP
receiver. Writing this document or merely copying the old code is insufficient.

The initial repository is private. Secrets and per-device registration are
runtime data. Every migrated capability has an acceptance gate below.

## 2. Baselines and hardware

UI source: Waveshare commit `6f78fa810ec505048c1941e3d09cefc9f29129bd`.
Services: voice-presence commit `caaf44a1473d1806b2249f0a9eb3d1fdfdd9cc57`, v0.3.1.

The primary acceptance board has P4 silicon v1.3, 16 MiB flash, 32 MiB PSRAM at
200 MHz, CPU at 360 MHz, C6 SDIO pins CLK18/CMD19/D0–D3=14–17/reset54,
JD9365 800×1280 DSI display, GT911 touch, OV5647 CSI camera and ES8311 audio.
Use ESP-IDF 5.5.x with the explicit legacy silicon profile. A separate rev3_x
configuration is permitted, but it cannot substitute for a verified v1.3 build.

The existing factory image is backed up separately on the operator's machine.
Never commit that backup or restore a raw NVS image from another device.

## 3. Required migration inventory

| ID | Capability | Required resulting behavior |
| --- | --- | --- |
| R01 | Brookesia desktop | Preserve launcher, Settings, touch, clock, brightness, volume, music, calculator and drawing apps; missing SD media must not prevent boot. |
| R02 | Nyx Panel app | A native launcher app shows sticky pushed JPEGs and the last 20 timestamped text messages, including CJK. Remote display pushes open this app. Clear removes both content types. Duration expiry is supported. |
| R03 | HTTP compatibility | Preserve the old route inventory and response status semantics for `/status`, LCD routes, snapshots, stream routes, configuration and system operations. URL OTA is asynchronous: 202 acknowledges acceptance; later download/digest errors are reported in `/status.ota`. Raw image digest rejection remains 422. LAN keeps its documented narrow recovery subset; authenticated Nyx exposes the full set. |
| R04 | NyxID connection | Runtime NVS credentials, authenticated WSS registration, heartbeats, reconnect with bounded backoff, service registration, and a truthful connection state. No UI work or long HTTP work in the socket callback. |
| R05 | Image transport | LAN accepts JPEG bytes directly. Retain JSON/Base64 Nyx compatibility and add negotiated binary-upload handling with bounded chunks; never advertise unsupported capabilities. Invalid, oversized or incomplete transfers cannot replace the displayed image. |
| R06 | Provisioning | Wi-Fi setup through Brookesia Settings; saved primary/fallback network configuration, Nyx key/node/slug provisioning through UART and QR, QR feedback, provisioning mode and unknown-device boot without a registry halt. Preserve optional device-code support only against real configured endpoints; mock mode is a development option. |
| R07 | Networking | One network initialization path shared with Settings; restore the board's existing Wi-Fi NVS, reconnect after loss, publish link/IP events, and retain Ethernet support on boards providing it. |
| R08 | Camera preview | Native Camera app uses the shared media owner. Repeated open, preview, swipe/back, reopen and close release resources and never race LVGL's event stack. |
| R09 | Snapshots and QR | Serve a current camera JPEG in preview mode and an LCD JPEG; snapshot requests and QR decoding share camera ownership. Publish mode retains the old explicit snapshot-unavailable response. |
| R10 | HA media | Configurable WHIP URL, H.264 FPS/bitrate, start/stop, auto-start, network retry, status and errors. Camera preview/scan and WHIP capture are mutually exclusive and transitions are serialized. Audio publishing stays disabled, matching v0.3.1. |
| R11 | Persistent config | Preserve existing namespaces for Nyx, stream, LCD, device, LAN and Wi-Fi where possible; validate input before committing. Secrets are masked in status and logs. |
| R12 | System operations | Authenticated reboot, reset preserving identity where documented, explicit full reset, image/URL OTA with SHA256 verification, and real update partitions. Never report successful OTA when no writable update slot exists. |
| R13 | Diagnostics | Firmware/BSP identity, heap, active media mode, Wi-Fi/Ethernet IP, Nyx state, stream state, screen state and meaningful last errors; UART status and provisioning commands. |
| R14 | Disabled MQTT compatibility | Keep configuration/status compatibility but leave the old MQTT bridge disabled, as it was in v0.3.1. Do not invent an active MQTT input surface. |
| R15 | Reproducibility | Standalone repository, pinned dependency lock, hardware profile defaults, documented build/flash/backup/restore and CI that builds the actual firmware and runs host protocol/state tests. |

## 4. Architecture and ownership

The existing `main` C++ application remains responsible for BSP I2C, audio,
display and Brookesia startup. A `panel_services` component owns backend state,
HTTP, Nyx, provisioning and stream orchestration. It exports narrow C interfaces
to native `NyxPanel` and `Camera` apps. Backend code never constructs a second
LVGL display or replaces the Brookesia screen stack.

### GUI

Use one Brookesia app/screen lifecycle and the BSP recursive LVGL mutex. Worker
threads submit bounded messages/content to the GUI adapter; decode or network
work happens outside the GUI lock. The GUI adopts completed content under the
lock and releases previous buffers only after image widgets stop referencing
them. The shared `App::notifyCoreClosed` lock fix is retained. The GUI task uses
an internal RAM stack because native Settings callbacks write NVS; an external
stack triggers the P4 flash cache-disable assertion on WLAN exit.

Remote `/lcd/screen` names map to native apps/pages: home→Nyx Panel,
settings→Settings, camera→Camera, scan→QR scanner, info/operator→device status,
wifi→WLAN, device_bind→provisioning, back→Brookesia navigation. An unsupported
target returns an explicit error; no fake navigation success or separate legacy
screen stack is allowed.

### Media

One owner initializes CSI and allocates capture buffers. Backend snapshots,
camera UI, QR scanner and WHIP request transitions through the serialized media
state machine. The baseline Camera's independent video-driver/dummy-display
pipeline must be adapted to that owner, not run alongside the old capture stack.
Reuse the BSP's initialized audio/codec handles instead of opening a second I2S
controller. No blocking camera transition while holding the GUI mutex.

### Network and Nyx

Share ESP netif, event loop and Wi-Fi initialization with Brookesia Settings.
Use existing saved station credentials as a fallback when legacy config is
absent. Link/IP changes drive Nyx and WHIP state. All socket receive buffers,
pending requests and display messages have fixed caps and clear ownership.
Control traffic/heartbeats remain responsive during JPEG transfers.

Binary uploads follow the actual NyxID protocol, including open/chunk/end,
request identity, ordering, length validation and cancellation. The old 64 KiB
JPEG limit remains the initial default until larger bounded transfers pass
hardware testing. This does not require Base64 on the new binary path. Legacy
JSON transport remains available for existing servers.

### Identity and configuration

Credentials come from NVS or explicit provisioning. Import neither
`devices_data.h` nor `secrets.h`. Derive an unregistered device label from the
hardware MAC; it must still boot into the desktop and accept provisioning.
Preserve the installed board's Wi-Fi configuration during normal upgrades.
Factory-reset operations must distinguish identity-preserving reset from full
wipe. A normal build and flash must not silently reset unrelated settings.

### Flash layout and OTA

The vendor's factory-only layout cannot provide functional A/B OTA on 16 MiB.
Choose explicit NVS, otadata, phy, two aligned OTA application slots and a
resource partition within the real chip size. Verify final image sizes against
both slots. Record the migration from the installed factory layout and back up
NVS before changing offsets. Fresh firmware validates its startup before
marking a pending update valid; failure retains an available rollback image.
Runtime Flash APIs share an internal RAM recursive mutex. Image verification
through a Flash mapping also holds that mutex, so status/configuration reads
cannot overlap OTA flash operations on the legacy P4.

## 5. Implementation sequence

1. Establish the independent repository, provenance, this SPEC and feature gates.
2. Integrate backend component builds and native GUI/media/network adapters;
   remove credential tables and duplicate initialization.
3. Bring up LAN diagnostics, pushed text/image, native navigation, snapshots,
   QR/UART provisioning and saved configuration.
4. Bring up Nyx WSS/registration/reconnect, negotiated binary transfers, WHIP
   ownership and real OTA. Keep the full migration inventory in scope.
5. Run host tests and firmware builds, flash the attached board with a backup,
   validate end-to-end behavior and publish reviewable commits/results.

Steps may overlap, but no feature is complete merely because its source builds.

## 6. Verification gates

| Gate | Required evidence |
| --- | --- |
| G01 Repository | Remote exists under the requested organization/name; committed source, SPEC, provenance and instructions are pushed; no device secrets or generated binaries tracked. |
| G02 Host tests | Existing protocol/Base64/mailbox/link/stream tests pass in the standalone tree. Add meaningful cases for binary fragmentation, oversize/truncation/cancellation and ownership transitions where implementation changes require them. |
| G03 Firmware | Clean ESP-IDF build with the committed lock/profile; 16 MiB legacy revision image headers and partition bounds verified. CI runs the same build. |
| G04 Hardware baseline | Fresh boot logs identify the new project/version and show display, touch, audio and C6 transport. Observe sustained operation and repeat Camera open/close at least three times without a panic. |
| G05 Display/API | Send `Hello Lexa` and a real JPEG through the supported API, inspect LCD screenshot, check CJK, clear, invalid JPEG, oversize rejection, navigation and timer expiry. Screenshot must show actual rendered output. |
| G06 Nyx | Provision this device, observe connected node/heartbeat, call status/text/image through Nyx, interrupt/recover connectivity, and verify the advertised transport path against the server actually used. |
| G07 Media/QR | Camera snapshot decodes; LCD snapshot decodes; QR provisioning consumes an actual QR image; WHIP receiver obtains H.264 frames; preview↔publish transitions recover without leaked buffers or duplicate CSI initialization. |
| G08 Persistence/OTA | Reboot retains settings; verified OTA moves to a real other slot and boots the expected image; mismatched digest is rejected. Destructive full-reset testing requires an isolated saved configuration and restoration. |
| G09 Documentation | Route inventory, provisioning procedure, build/flash commands, verified results and outstanding limitations match current code/hardware. |

Do not claim success for an external receiver, physical screen interaction,
Nyx capability, or OTA slot using only compilation or mocks. If an external
dependency is unavailable, retain that gate as incomplete and continue all
independent implementation and verification.

## 7. Current progress

- [x] Inspected both source baselines and the attached hardware profile.
- [x] Defined the migration inventory, ownership constraints and acceptance gates.
- [ ] G01 repository and initial commits.
- [x] G02 standalone host tests (11 executables, including upload/fragmentation/signing fixtures).
- [ ] G03 firmware build and CI.
- [ ] G04 hardware baseline.
- [x] G05 display and API acceptance (actual Nyx JPEG/text/CJK, inspected LCD JPEG, rejection/expiry checks).
- [ ] G06 Nyx end-to-end acceptance.
- [ ] G07 media and QR acceptance.
- [ ] G08 persistence and OTA acceptance.
- [ ] G09 final operational documentation.

Measured evidence and remaining checks are maintained in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md). Raw device logs, registration
credentials and flash backups remain private outside this repository.
