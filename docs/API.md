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
| GET | `/lcd/default-image` | no | Fallback image metadata: `source` (`stored`/`builtin`), `persistent`, dimensions and `jpeg_bytes` |
| POST | `/lcd/default-image` | no | Raw JPEG, up to 65536 bytes and 800×1280 pixels; validates and saves in NVS before returning 202; survives reboot/app updates; leaves an active temporary image visible |
| DELETE | `/lcd/default-image` | no | Deletes the stored fallback and restores the embedded brand background; leaves temporary images and captions intact |
| POST | `/lcd/text` | no | `{"text":"NyxBot","size":"medium","color":"#ffffff","bg":"#0d0d0d","align":"center","duration_ms":0}`; latest caption floats over the image, `bg` is the 2-pixel outline color; retains 20 messages for expiry restoration |
| POST | `/lcd/clear` | yes | Clears temporary image and message history, revealing the persistent fallback |
| GET | `/lcd/snapshot.jpg` | yes | Actual DMA-selected 800×1280 framebuffer encoded as JPEG, including system overlays; no object-tree redraw |
| GET | `/lcd/screen` | yes | Actual `target`, native stack `depth` and `err` |
| POST | `/lcd/screen?target=NAME` | no | `home`, `settings`, `wifi`, `camera`, `scan`, `info`, `operator`, `device_bind`, `back`; unknown target 404; accepted navigation runs on the GUI timer |
| POST | `/camera/capture` | no | Take one photo, show the same JPEG for two minutes and return it; preserves the fallback and voice controls |
| GET | `/camera/snapshot.jpg` | no | Fresh 800×1280 JPEG on demand; LCD unchanged; CSI/buffers/encoder released before response unless native preview is open; 503 busy/unavailable |
| GET | `/camera` | no | Mode, dimensions, capture counts, last duration/error; no automatic capture |
| GET | `/openapi.json` | no | OpenAPI for the requesting service; management operations are absent from the daily contract |
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
expiry; it only permits POST `/lcd/image` and `/lcd/default-image`. It does not advertise machine v2.

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

## Startup and fallback image

The panel opens Nyx Panel on every boot, before turning on its backlight. With
no temporary push, it displays the stored default image, or the embedded NyxID
NyxBot image if none is stored. Temporary image TTL expiry and `/lcd/clear`
restore this fallback. Captions float over either image and keep their own TTL.
Invalid uploads leave both the displayed image and stored blob intact. Image
decoding and NVS writes happen outside the GUI lock. One NVS blob provides
atomic replacement across power loss. Factory reset removes the custom image.

This is the live Panel page, with no timed splash or network wait. Fallback art
decodes alongside hardware initialization. Networking and clock sync run in the
background; `/status.time_valid` reports whether the clock can validate signed
requests, independently of opening Settings.

Use the authenticated NyxID proxy to configure it remotely:

```bash
nyxid proxy request "$DAILY_SLUG" /lcd/default-image -m POST \
  -H 'Content-Type:image/jpeg' -d @background.jpg
nyxid proxy request "$DAILY_SLUG" /lcd/default-image
nyxid proxy request "$DAILY_SLUG" /lcd/default-image -m DELETE
```

Resize/recompress first; JPEGs must fit both byte and dimension limits. Signed
Nyx binary uploads support `/lcd/image` and `/lcd/default-image`. Persistent
image configuration is excluded from unauthenticated LAN recovery routes.

## Daily and management services (v0.6.15)

Onboarding creates two ordinary NyxID UserServices for the same node:
`<board-slug>` (daily) and `<board-slug>-admin` (management). Each can be granted
independently to Keys and Bots with the existing service permission model.
A normal API Key also needs access to the node under NyxID's existing node scope;
that node grant does not replace the required service grant.
Existing daily service IDs stay stable. A Bot with daily access receives no
management access automatically. The management service includes daily actions.

The board checks the exact service slug supplied by the authenticated NyxID
control connection before entering a handler. Unknown or missing slugs are denied.
The ordinary v1 request HMAC does not cover service identity; identity metadata
comes from the authenticated control connection (TLS in production). It is never
taken from caller HTTP headers. LAN retains its own fixed recovery subset and
exposes no new voice controls.

Daily permits status, service discovery, temporary and persistent display images,
screenshots, on-demand camera, disabled-stream status/stop, Bot list refresh,
call controls, hardware volume, playback gain, saved input mode and brightness.
`/voice/config` (including Bot selection), `/config/*` except `/config/audio`,
navigation, login/QR setup, audio diagnostics and `/system/*` require management.
The dedicated daily mode and brightness endpoints accept only their named field;
unknown, duplicate, combined, out-of-range or fractional fields are rejected.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/voice` | Call state, selected Bot, input mode and saved preference, local mic mute, volume/gain, session ID, remote request stage, audio counts and peaks |
| POST | `/voice/open` | `{"request_id":"UUID","text":"Optional spoken greeting","microphone_muted":false}`; opens automatic two-way voice and the controls; 202 asynchronous, duplicate 200, busy/conflicting identity 409 |
| POST | `/voice/say` | `{"request_id":"UUID","text":"Notification"}`; requires active call, queues bounded speech |
| POST | `/voice/microphone` | `{"muted":true}`; input mute only, leaves playback available |
| POST | `/voice/start` | Starts with saved input mode and opens controls |
| POST | `/voice/stop` | Asynchronous hangup; poll until `busy:false` |
| POST | `/voice/hold?pressed=1` | Hold-mode microphone gate; 0 releases |
| GET | `/voice/agents` | Bots currently discovered using this board's own Key; includes each exact Voice `service_id` |
| POST | `/voice/agents/refresh` | Refresh authorized Bots asynchronously |
| GET, POST | `/voice/config` | Management: POST exactly one `agent_id`, `input_mode` (`automatic`/`push_to_talk`) or `playback_gain` (integer 1..8) |
| GET, POST | `/config/audio` | Daily: POST exactly one `volume` (integer 0..100) or `playback_gain` (integer 1..8); persisted for this profile |
| POST | `/voice/input-mode` | Daily: exactly one `input_mode` (`automatic`/`push_to_talk`); requires idle, preserves selected Bot |
| GET, POST | `/lcd/brightness` | Daily: exactly one integer `brightness` (0..100); preserves rotation and idle behavior |
| GET, POST, DELETE | `/lcd/default-image` | Daily: read metadata, save persistent JPEG fallback, or restore the bundled NyxBot picture |

`/status.voice` mirrors live voice state; `/status.nyx.management_slug` identifies
the separately grantable management service. State includes network/display/media,
heap, firmware and OTA diagnostics without credentials.

Remote open uses the selected Bot and the panel's own login credential. It uses
an automatic per-call mode without replacing the saved mode. No call starts at
boot. `microphone_muted:true` permits a notification with input muted.
Speech uses NyxID's authenticated provider sideband, never a fake user transcript
or client-supplied provider frame. Text is nonblank UTF-8, at most 300 bytes, with
no control characters. Unknown/duplicate fields and trailing JSON are rejected.
The last 8 remote request IDs plus content hashes persist before side effects;
matching retries do not open a microphone or repeat speech, including after
reboot. A failed/uncertain attempt requires a new ID for an intentional new try.
The server additionally deduplicates per call, bounds its queue and expires
unsent notifications after 30 seconds. `queued`, `dispatched`, `accepted` report
actual processing stages; `accepted` is not an audible-playback receipt.
