# Hardware acceptance, 2026-11-03

This record is updated during acceptance. An unchecked item remains open.
The acceptance target is the attached P4 v1.3 / 16 MiB board, OV5647, JD9365,
GT911 and ES8311. Private raw logs, credentials, flash backups and camera images
are kept outside the repository under the operator's dated output directory.

| Check | Evidence / result |
| --- | --- |
| Complete pre-migration backup | 16 MiB; SHA256 `b25d05f4e50b90eba16b969bc82a1e516518fa64a07decba5350e1db51f09005` |
| Host protocol/state tests | All 11 executables pass, including independently generated HMAC fixtures with hex-decoded keys, binary chunk validation and WebSocket fragmentation |
| Project boot | Serial identifies `nyxid-esp-panel`; display, touch, ES8311 and C6 initialize; missing SD card is logged and desktop still boots |
| LAN JPEG / LCD screenshot | Raw JPEG accepted; screenshot decoded and visually inspected |
| Nyx identity / auth | Runtime UART provisioning, WSS TLS certificate verification, node auth and `/status` request succeeded |
| Nyx JPEG / text / CJK | Actual proxy requests display Hello Lexa image and `Hello Lexa · 你好，世界`; 800×1230 app-screen JPEG visually inspected |
| Invalid image / oversize | HTTP 400 / 413; screenshot remains byte-identical to prior content |
| Text expiry | 2-second message appears; after expiry screenshot equals the original content |
| Invalid native navigation | Unknown target rejected |
| Real QR decoder input | A generated JPEG QR passes through the firmware decoder/provisioner and schedules reboot; camera optics are a separate physical check |
| OTA digest rejection | Wrong SHA256 returns 422; active slot unchanged |
| OTA A/B | Actual raw application updates switch `ota_0`→`ota_1`→`ota_0`; services restart, credentials persist and pending image becomes valid after 30 seconds |
| Camera repeated preview / snapshot | In progress; first run exposed a missing stream type in esp_video 2.1 G_FMT, now restored before S_FMT |
| WHIP H.264 receiver | In progress; local MediaMTX receiver is available |
| Clean build / CI | In progress |
| Link interruption / reconnect | In progress |

`tools/accept_hardware.py` performs real Nyx/LAN display checks and optional
three-cycle Camera/snapshot/back acceptance. It does not simulate hardware.
`tools/provision_agent.py` imports an encrypted Nyx node-agent configuration
without printing the credentials; a USB-UART open may pulse reset, so it waits
for the console before sending commands.

The binary receiver is bounded to 64 KiB JPEG pushes and advertises
`proxy_upload_v1` only with a signing secret. The actual ordinary Nyx HTTP proxy
used for the display acceptance above still forwards JSON/Base64. Machine
binary ingress uses a distinct negotiated path. Do not infer its end-to-end
acceptance from the host receiver tests or the ordinary HTTP proxy result.

Camera QR optics, physical gestures/audio perception and long-duration soak
are not established by a serial log or a generated JPEG injection.
