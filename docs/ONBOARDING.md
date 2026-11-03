# Native onboarding (v0.2.0)

The [NyxID audit](NYXID_FEATURE_AUDIT.md) records upstream and deployed behavior
before this integration. The deployed API was checked at commit
`2c0d66d66f25cb591682fb2bc778238248a33343`; latest upstream main at the time
was `ff302d5ecfe8d188cf0f167995e87bc08c722296`.

## Screen ownership

Nyx Panel displays full-screen pushed images and outlined text captions. Native Settings owns
WLAN, NyxID connection and device status. Camera handles previews and scanning.
Remote `home`, `settings`, `wifi`, `camera`, `scan`, `info`, `operator`,
`device_bind` and `back` remain supported. Scanning reports target `scan` and
returns to the Settings page that opened it. No status JSON or UART instructions
appear in the content app.

For an unbound board's USB bring-up, UART `screen <target>` opens the same native
pages, and `nyx-login start|status|cancel` controls the same login worker without
exposing account tokens or secret device codes. `nyx-status` reads the active
identity and does not activate a newly saved binding.

## Account device login

Open **Settings → NyxID → Sign in to NyxID**. A phone opens the trusted
`https://nyxid.ai/login/device` link and explicitly approves the displayed code.
The panel calls `/api/v1/auth/device/request`, waits the advertised interval,
polls `/auth/device/poll`, and accepts only `auth_kind=account_session`.
Pending, slow-down (+5 seconds), rate limiting, denial, expiry and already
delivered responses have separate handling. Codes expire rather than silently
starting another login. Retries respect the original monotonic expiry deadline.
Cancel/back stops further work; a late-delivered account token is revoked.

With the temporary account bearer, a worker creates `/nodes/register-token`,
sends WSS `register`, and validates `register_ok`. It creates/verifies a custom
`/keys` service with `auth_method=none`, endpoint `http://127.0.0.1:80`, and the
expected `node_id`. An existing same-account panel/service is reused; a
conflicting service on another node is never overwritten.

The reserved node ID and returned node credentials are staged in NVS namespace
`nyx_setup`, blob `pending_v1`. Interrupted setup resumes after another explicit
account authorization. A lost register response can be recovered using the saved
node ID and token rotation. Failed ownership checks stop setup. The existing
working identity remains until final success.

The completed identity is a single atomic NVS blob `nyx/binding_v1`: node auth
token, node ID, signing secret and service slug. Restart using the page's button
to activate it. Legacy credentials, UART setters and identity-preserving reset
also understand the new format. Account access/refresh tokens and secret device
codes are never persisted. Logout is attempted even on failure, then buffers
are cleared. Offline logout cannot be guaranteed; the server session retains
its normal expiry. NyxID logout also closes that user's unauthenticated MCP
fallback sessions; the approving browser session is separate.

These are NyxID first-party APIs. Separate `/devices/code` and bootstrap flows
deliver restricted client API keys. The panel has no cloud API consumer and
does not redeem those identities; they cannot authenticate a control-node WSS.

## Wi-Fi QR

Open **Settings → WLAN → Scan Wi-Fi QR** (also at Settings root). Standard
`WIFI:T:WPA;S:Network;P:password;;` codes support personal WPA/WPA2/WPA3 or
`T:nopass` for open networks. Escaped semicolons, colons, backslashes, Unicode
SSIDs and `H:true` are accepted. SSIDs are bounded to 32 bytes, personal
passwords to 8–63 bytes. WEP/enterprise/EAP are outside this onboarding UI.
Duplicate/unsupported fields and malformed escapes are rejected. Legacy
`nyxprov://wifi?ssid=...&psw=...` remains supported.

The camera shows the SSID without revealing the password. **Connect to this
network** confirms the change. A worker tries the shared station network for
up to 25 seconds, saving the primary configuration only after connection.
Cancellation/failure restores the prior station and saved configuration. The
UI stays responsive; success does not reboot. Existing media ownership still
serializes Camera and WHIP access.

Authenticated `/wifi/qr` exposes the same state and confirmation action. The
development JPEG-injection route uses the real decoder and now reports
`confirmation_required`; open `scan` first. It cannot bypass confirmation or
replace a candidate while confirmation/connection is in progress, and does not
prove camera optics. Legacy node/MQTT provisioning codes retain their restart
behavior. A new node import requires its signing secret; a same-node import may
reuse the current secret. Node credentials are saved together, and a different
node never inherits the previous node's service slug or signing secret. Bootstrap
client tokens are rejected rather than written as node credentials.

## Verification

Host checks cover login response/error semantics, wrong identity rejection,
Wi-Fi escaping, Unicode, hidden/open networks and input bounds. Firmware builds
verify real slot capacity. Live login grants, node registration, physical scans
and network transitions are tracked separately in [VERIFICATION.md](VERIFICATION.md).
