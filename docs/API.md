# Panel API

Nyx routes below run only after authenticated WSS. Signed requests are checked
when a signing secret is configured. LAN intentionally exposes a recovery subset.
JSON endpoints use `Content-Type: application/json`; JPEG endpoints use
`image/jpeg`. The JPEG push cap is 65536 bytes, measured before decoding.

| Method | Path | LAN | Body / behavior |
| --- | --- | --- | --- |
| GET | `/status` | yes | Firmware, ELF hash, network, Nyx, actual native screen, media, heap and OTA slots/state |
| POST | `/lcd/image` | yes | Raw JPEG; optional `?duration_ms=N`, 0 means sticky; 202 accepted, 400 invalid JPEG, 413 oversized |
| POST | `/lcd/text` | no | `{"text":"Hello Lexa","size":"medium","color":"#ffffff","bg":"#121a2b","align":"left","duration_ms":0}`; last 20 messages |
| POST | `/lcd/clear` | yes | Clears image and message history |
| GET | `/lcd/snapshot.jpg` | yes | Current LVGL screen encoded as JPEG |
| GET | `/lcd/screen` | yes | Actual `target`, native stack `depth` and `err` |
| POST | `/lcd/screen?target=NAME` | no | `home`, `settings`, `wifi`, `camera`, `scan`, `info`, `operator`, `device_bind`, `back`; unknown target 404; accepted navigation runs on the GUI timer |
| GET | `/snapshot.jpg` | yes | Current preview JPEG; publish mode returns 503 |
| GET | `/stream/status` | yes | Publisher state, target, profile and last error |
| POST | `/stream/start` | yes | Starts configured video-only WHIP publisher asynchronously |
| POST | `/stream/stop` | yes | Stops publisher asynchronously |
| GET, POST | `/config/stream` | yes | `target_whip_url`, `auto_start`, `fps`, `bitrate` |
| GET, POST | `/config/lcd` | no | `idle_mode`, `brightness`, `rotation`; physical display orientation follows the BSP profile |
| GET, POST | `/config/device` | no | `label`, `location` |
| GET, POST | `/config/wifi` | no | `ssid_1`, `psw_1`, `ssid_2`, `psw_2`; GET excludes passwords; POST defaults to `dry_run:true`; false tests connection before saving |
| GET, POST | `/config/lan_server` | no | `enabled`, `port`; changes require reboot |
| GET, POST | `/config/mqtt` | yes | Compatibility config; password is redacted; bridge remains disabled |
| POST | `/system/reboot` | no | Reboots after sending response |
| POST | `/system/factory_reset` | no | Resets config while preserving Wi-Fi and Nyx identity, then reboots |
| POST | `/system/ota` | no | `url` and required `sha256`; 202 accepts a background download, 409 busy; poll `/status` → `ota.stage`/`last_error`; only a verified image switches the boot slot |
| POST | `/system/ota/image` | yes, bearer required | Raw app binary, `Authorization: Bearer <node token>` and `X-SHA256`; digest mismatch 422; success 202 then reboot |
| POST | `/_dev/qr/inject_jpeg` | no | Development decoder test: a real QR JPEG passes through the same decoder/provisioning path; no camera-optics validation |

The full-wipe operation is the explicit UART `factory-reset` command. Save a
private backup first. `nyx-status` masks credentials. `nyx-key`, `nyx-node`,
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
certificates. A failed download/hash leaves the active boot slot unchanged;
poll `ota.stage == "failed"` and `ota.last_error` after an accepted request.
