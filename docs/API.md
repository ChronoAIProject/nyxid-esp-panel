# Panel API

Nyx routes below run only after authenticated WSS. Signed requests are checked
when a signing secret is configured. LAN intentionally exposes a recovery subset.
JSON endpoints use `Content-Type: application/json`; JPEG endpoints use
`image/jpeg`. The JPEG push cap is 65536 bytes, measured before decoding.

| Method | Path | LAN | Body / behavior |
| --- | --- | --- | --- |
| GET | `/status` | yes | Firmware, ELF hash, network, Nyx, actual native screen, media, heap and OTA slots/state |
| GET, POST | `/nyx/login` | no | Sanitized account device-login status; POST starts login and opens Settings; no secret device code or tokens returned |
| POST | `/nyx/login/cancel` | no | Cancel login; a late account-session delivery is revoked before further registration |
| GET, POST | `/wifi/qr` | no | Scanned SSID and confirmation/connection state; POST confirms the scanned network; passwords stay hidden |
| POST | `/lcd/image` | yes | Raw JPEG; fills the 800×1280 canvas with an aspect-preserving center crop; optional `?duration_ms=N`, 0 means sticky; 202 accepted, 400 invalid JPEG, 413 oversized |
| POST | `/lcd/text` | no | `{"text":"Hello Lexa","size":"medium","color":"#ffffff","bg":"#121a2b","align":"center","duration_ms":0}`; latest caption floats over the image, `bg` is the 2-pixel outline color; retains 20 messages for expiry restoration |
| POST | `/lcd/clear` | yes | Clears image and message history |
| GET | `/lcd/snapshot.jpg` | yes | Current LVGL screen encoded as JPEG |
| GET | `/lcd/screen` | yes | Actual `target`, native stack `depth` and `err` |
| POST | `/lcd/screen?target=NAME` | no | `home`, `settings`, `wifi`, `camera`, `scan`, `info`, `operator`, `device_bind`, `back`; unknown target 404; accepted navigation runs on the GUI timer |
| GET | `/camera/snapshot.jpg` | no | Fresh 800×1280 JPEG on demand; LCD unchanged; CSI/buffers/encoder released before response unless native preview is open; 503 busy/unavailable |
| GET | `/camera` | no | Mode, dimensions, capture counts, last duration/error; no automatic capture |
| GET | `/openapi.json` | no | Camera service OpenAPI with `camera_take_photo` and `camera_get_status` |
| GET | `/snapshot.jpg` | preview only | Nyx: alias for on-demand capture. LAN: requires native preview already open, otherwise 503 |
| GET | `/stream/status` | yes | Compatibility: state `disabled`, auto_start false |
| POST | `/stream/start` | yes | 410 `continuous_streaming_removed`; replacement `/camera/snapshot.jpg` |
| POST | `/stream/stop` | yes | Idempotent 200, state `disabled` |
| GET, POST | `/config/stream` | yes | GET reports disabled; POST returns 410. Legacy NVS settings cannot activate streaming |
| GET, POST | `/config/lcd` | no | `idle_mode`, `brightness`, `rotation`; physical display orientation follows the BSP profile |
| GET, POST | `/config/device` | no | `label`, `location` |
| GET, POST | `/config/wifi` | no | `ssid_1`, `psw_1`, `ssid_2`, `psw_2`; GET excludes passwords; POST defaults to `dry_run:true`; false tests connection before saving |
| GET, POST | `/config/lan_server` | no | `enabled`, `port`; changes require reboot |
| GET, POST | `/config/mqtt` | yes | Compatibility config; password is redacted; bridge remains disabled |
| POST | `/system/reboot` | no | Reboots after sending response |
| POST | `/system/factory_reset` | no | Resets config while preserving Wi-Fi and Nyx identity, then reboots |
| POST | `/system/ota` | no | `url` and required `sha256`; 202 accepts a background download, 409 busy; poll `/status` → `ota.stage`/`last_error`; only a verified image switches the boot slot |
| POST | `/system/ota/image` | yes, bearer required | Raw app binary, `Authorization: Bearer <node token>` and `X-SHA256`; digest mismatch 422; success 202 then reboot |
| POST | `/_dev/qr/inject_jpeg` | no | Open `scan` first; real QR decoding stages Wi-Fi for confirmation. Busy/pending scans reject replacement. No camera-optics validation |

Display pushes open Nyx Panel. Image and caption lifetimes are independent;
zero duration is sticky. Only the latest unexpired caption is visible. It wraps
inside 40-pixel side margins with a 56-pixel bottom inset; longer text is
clipped within the safe area. `align` accepts `left`, `center`, or `right`.
`small` uses the default Latin font; `medium` uses the bundled Chinese font.
No opaque text panel, timestamp or status bar covers the picture. Clear removes
the picture and the retained captions. Image uploads remain limited to
800×1280 source pixels and 64 KiB; prepare larger pictures before uploading.

The explicit UART `factory-reset` command wipes P4 NVS, including Nyx identity.
The C6 Wi-Fi driver has separate storage and can retain its station settings.
Save a private backup first. `nyx-status` masks credentials. `nyx-key`, `nyx-node`,
`nyx-secret` and `nyx-slug` save NVS values; reboot applies the WSS credentials.
`wifi-reconnect` disconnects the station to exercise the saved-network reconnect
path without changing its configuration.

LAN raw JPEG and raw OTA carry binary bytes. Ordinary Nyx HTTP proxy calls still
use the server's JSON/Base64 forwarding. Negotiated machine-ingress
`proxy_upload_v1` uses signed metadata plus `NYXM` binary chunks. The receiver
validates request identity, sequence, size, completion, cancellation, replay and
expiry; it only permits POST `/lcd/image`. It does not advertise machine v2.

URL OTA requires HTTP 200 with a positive `Content-Length` within the inactive
slot size. It follows at most three HTTP redirects and verifies HTTPS server
certificates. It buffers the complete bounded image in PSRAM, checks SHA256,
closes HTTP, then erases/writes flash in 4 KiB steps. Allocation failure is
reported as a failed update. A failed download/hash leaves flash untouched;
poll `ota.stage == "failed"` and `ota.last_error` after an accepted request.

Camera capture uses a single owner shared with native Camera and Wi-Fi QR.
It uses the system Camera's 800×1280 RGB565 / native 50 FPS capture profile and
two capture buffers. Responses are quality-50 JPEGs bounded to 512 KiB.
Concurrent capture requests fail with 503 rather than queueing capture jobs.
Errors return JSON and never a stale JPEG. Capture startup waits at most about
5.6 seconds for its first frame, then up to 2.5 seconds for a fresh frame including
a one-second cold exposure warmup; hardware
encoding has a 1-second timeout and cleanup waits up to 1.5 seconds for the pump.
Network/proxy latency is additional. `/camera` and `/status` do not wait for CSI.
For CLI JPEG output use `nyxid proxy request "$SLUG" /camera/snapshot.jpg --stream`;
this flag copies raw response bytes from one request and does not enable video.
The schema endpoint alone does not install dedicated Nyx MCP tools.
