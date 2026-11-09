# Standalone NyxID voice

The panel signs in through NyxID Agent Key login and stores its own child
credential. In NyxID, publish Device voice from the intended Bot's details and
select only that Bot's Voice connection when assigning the panel's Key. Refresh
Bots in the panel settings and select that Bot. The provider key stays on NyxID.
Voice uses the account's saved provider selection and the Bot's existing grants.

The board discovers authorized connections with `GET /api/v1/keys` using that
same stored Agent Key. Only active `nyxbot-voice` catalog entries with an exact
`/api/v1/proxy/nyxbot-voice/{service_id}` endpoint path qualify. Each connection's
`/voice/agents` response must contain its one fixed Bot and the matching service
ID. Bot names are display labels, never routing or authorization inputs.

All Voice HTTP operations and the session WebSocket use
`{configured_nyxid_base}/api/v1/proxy/nyxbot-voice/{service_id}` with the existing
`/voice/options`, `/voice/sessions`, `/voice/sessions/{id}/control`, `/stream`
and `/announcements` tails. The board constructs this URL from its configured
NyxID origin and the verified ID; it never follows the inventory endpoint's
host or redirects with the credential. There is no native web Voice route
fallback or additional Voice token.

Saved selection version 2 binds the panel Key, Voice service ID, Bot UUID and
conversation ID. An existing version 1 selection is upgraded after refresh only
when the same Key, Bot and conversation identify exactly one authorized Voice
connection. A replaced service ID requires selecting the Bot again. Ambiguous
or mismatched discovery responses are rejected. The local Voice status and Bot
list expose the exact `service_id` for diagnostics.

Discovery's optional `description` supplies the selected Bot's introduction.
The panel caches it for the next boot without changing the saved voice binding.
Nyx Panel briefly shows the Bot name and introduction at startup, then hides the
notice after ten seconds. Servers that omit `description` still work and show
the name alone. A missing custom default background adds a setup hint to this
notice; the background is device-wide, independent of the selected Bot.

Agent Key voice and typed camera-tool discovery require
[NyxID #2169](https://github.com/ChronoAIProject/NyxID/pull/2169), the server-side
`assistant:voice-agent-key` feature enabled, and an explicitly published/granted
Bot Voice connection with the dedicated service proxy routes. The v0.6.15 audio
baseline was validated against local NyxID; the service transport migration
still requires matching backend and hardware validation. A firmware merge alone
does not deploy or enable the backend feature.

The firmware supports NyxID's `openai_live` WebRTC transport with Opus audio.
Tap the microphone icon in Nyx Panel to start a call and open the voice controls;
if a call is already connecting or active, the tap only opens its controls.
Tap the icon again to collapse them. The controls start collapsed at boot.
Green means call, and red means hang up; loading Bot metadata is shown in gray
and cannot be mistaken for a live call. A dot on the icon indicates
connecting, active microphone or local mute while the controls are hidden.
The expanded controls show the selected Bot, an animated **Connecting...**
state with preparation progress, then **Connected mm:ss** only once media,
provider readiness and NyxID control are all ready. Dialing offers **Cancel**;
an active call offers **Hang up**. Ending and failed calls have distinct states.
Local ringback plays during setup and stops before Bot playback, on connection,
or on cancellation/failure. It is not counted as provider audio or call duration.

`GET/POST /voice/greeting` configures an optional greeting on this panel only.
New panels leave it disabled. POST `{"enabled":true}` to enable it, or
`{"enabled":false}` to disable it. The panel stores only the switch, not a
script or language. After connection it asks NyxID to let the selected Bot greet
naturally using that call's configured identity, persona and language.
Updates persist atomically, apply to the next call and retain the selected Bot
and Key. The existing authenticated announcement queue carries a typed greeting
signal, not a user utterance or tool request. An explicit opening text in a
remote `/voice/open` takes priority; a failed or uncertain greeting is not replayed.
Servers advertise `greetings:true` in authorized Bot discovery. Older servers
can still make calls; the greeting is skipped with `greeting_state:unavailable`,
without turning it into a remote notification. Upgrade NyxID API/Voice workers
to enable generated greetings. v0.6.18's saved switch is retained and its script
is discarded on the next settings save.

Choose **Hold to talk** or **Auto conversation** in the dropdown before starting
a call. The choice persists for the current server profile. Automatic
mode sends the echo-cancelled microphone while the call is active; NyxID's
existing Live conversation handles pauses and responses. Its button toggles
local microphone mute. Hang up before changing modes. Playback remains audible
while Talk is held. A 16 kHz full-duplex
AEC processes the microphone against the board's synchronous I2S playback
reference before Opus encoding. It uses the voice-communication mode and the
vendor's default aggressive residual-echo suppression. A software playback
queue cannot provide the same capture/playback alignment. Continuous digital AGC
adds 15 dB compression gain with a limiter and a -3 dBFS target; it does not add
local voice activity detection or turn segmentation. Codec and AEC frame sizes
are adapted without dropping samples. MusicPlayer and SpecAnalyzer release the shared codec before voice
claims it. Ending a call releases the codec to idle wake listening when enabled.
The panel does not upload microphone audio while waiting for its wake word.

## Hi ESP wake-up and speech during connection

Local WakeNet9 **Hi ESP** detection is enabled by default. The voice panel has
a wake switch, and the daily service exposes `GET/POST /voice/wake`:

```json
{"enabled":true}
```

The switch persists in the current NyxID server profile. The listener pauses
for microphone mute, missing network/Bot configuration, foreground audio apps
and OTA. MusicPlayer and SpecAnalyzer can claim the codec without disabling
the saved wake preference. An active call keeps exclusive audio ownership.

After detection, the same microphone task keeps reading AEC/AGC-processed
16 kHz mono samples while NyxID connects. A 500 ms local lookback protects speech
at the detection boundary. Connection audio and subsequent live audio enter one
PCM FIFO; there is no codec reopen or second I2S reader at the transition.
After the provider, media and controls are ready, Opus packets carry the FIFO's
oldest samples first, with continuous 20 ms timestamps and one packet per 20 ms.
The existing Espressif Sonic processor preserves pitch while playing a backlog
at 1.5x speed, then returns to normal speed as the queue empties. This catches up
without flooding the receiver with future RTP packets. Its pending audio is
also reset on microphone mute. No local sentence detection or speech segmentation
is added. Wake-initiated calls use automatic conversation without changing the
saved input mode, and skip the automatic greeting and dialing tone so neither
can interrupt the user's initial request. Manual and remote calls retain the
existing greeting preference and dialing tone.

The PSRAM FIFO holds at most 60 seconds. Overflow fails the call explicitly;
it never silently overwrites earlier speech. Hangup and failure erase queued
samples. Microphone mute clears pending samples rather than sending them after
unmute. Disabling wake during a call affects later wake-ups, not the active call.
`GET /voice` reports `wake.enabled`, `listening`, `state`, `detections`,
`initiated_call`, `buffered_ms` and `transmitted_ms` for diagnosis.

The pinned Hi ESP model is compressed inside each OTA application image and
expanded into PSRAM once. Existing partitions, login and stored pictures stay
compatible; no separate model partition or model flashing step is needed.

Control WebSocket interruptions pause microphone input and reconnect within a
25-second grace period, retaining the existing media session. Invalid,
incomplete or unknown snapshots are ignored. Transport and codec failures
release local audio without sending a user hangup command; NyxID expires a
lost client's heartbeat while already accepted work keeps running. The local
Hang up action still explicitly ends the call.

## Local NyxID development

Production builds use the default HTTPS NyxID URLs. For local development:

```sh
NYX_DEV_HOST=192.168.1.100 # Replace with your reachable LAN server address.
idf.py \
  -DNYX_PANEL_AGENT_DEV_API_URL="http://${NYX_DEV_HOST}:3001" \
  -DNYX_PANEL_AGENT_DEV_LOGIN_URL="http://${NYX_DEV_HOST}:3000/login/agent-key" \
  build
```

These overrides accept a private LAN HTTP origin and verification page on the
same host. Agent credentials and voice selection use namespaces derived from
the URL pair. Production credentials are never sent to the development server.
Wi-Fi and background keep their existing configuration. By default the node
connection remains on production. Add `-DNYX_PANEL_NODE_DEV_MODE=ON` to also
onboard a separate node and board service against local NyxID. That mode uses
separate node and pending-setup namespaces and disables baked node credentials;
it never sends production node credentials to the local server. Use
`nyx-login node-setup`, approve its local device code, then restart after setup.
Board Key → Bot Voice and Bot → board service are independent grants.

A preserving factory reset clears configuration and custom artwork in place,
retaining Wi-Fi, all active/inactive identity profiles and pending setup. It does
not erase and reconstruct the NVS partition. Explicit full reset still erases it.

Clear all overrides when building for production:

```sh
idf.py -DNYX_PANEL_NODE_DEV_MODE=OFF \
  -DNYX_PANEL_AGENT_DEV_API_URL= -DNYX_PANEL_AGENT_DEV_LOGIN_URL= build
```

USB diagnostic commands carry no credentials:

```text
nyx-login start
nyx-login status
nyx-voice refresh
nyx-voice list
nyx-voice select <Bot UUID>
nyx-voice mode hold
nyx-voice mode automatic
nyx-voice start
nyx-voice status
nyx-voice hold
nyx-voice release
nyx-voice mute
nyx-voice unmute
nyx-voice stop
```

Only the panel may poll its login request. An administrator can approve its
displayed user code through the normal NyxID flow. Serial `hold` and `release`
simulate the panel's push-to-talk button for physical microphone checks.

Opus task budgets include Espressif's documented codec stack requirements:
48 KiB for the microphone encoder and 24 KiB for the speaker decoder. The first
encoded/decoded frame reports unused stack bytes for hardware validation.

The bundled peer component uses an ECDSA P-256 DTLS client certificate for
compatibility with modern WebRTC endpoints. Media still requires the negotiated
SDP fingerprint; this change does not relax HTTPS or credential validation.
The certificate is prepared in the background and reused for this boot.
After a server has returned saved `preferences` in `/voice/options`, later calls load
current provider choices and settings in one request. The first call and older
servers keep the established `/voice/agents` then `/voice/options` order. Only
the response capability is remembered in RAM. Neither path caches permission decisions;
NyxID checks the saved service/Bot binding again when creating the session.

Initial WebRTC offers normalize only SDP `a=setup` lines to `actpass` (RFC 5763).
The peer library's `passive` initial offer conflicts with endpoints that answer
`passive`. ICE secrets and fingerprints are preserved byte for byte; malformed
or oversized offers are rejected before contacting NyxID.

Provider readiness events may include the full session configuration; the parser
accepts bounded messages up to 96 KiB with a 128 KiB SCTP receive cache. It logs
only lengths and readiness flags. A lost control socket stops capture/playback
and reports an error; an intentional hangup remains idempotent.

The real provider startup event measured 4,369 bytes in local board validation;
the previous 4 KiB guard silently discarded it although RTP audio was already
flowing. Both encoder and decoder stack high-water marks are recorded in the
local validation artifact.

Unmuting the ES8311 restores the Settings output volume, matching MusicPlayer.
Voice does not override the user's selected volume or mute the speaker while the
Talk button is held. In Hold mode the microphone gate closes on button release.
Server mute, lost control authority and hangup close it in both modes.

Both input modes require the same active control authority, connected media and
provider readiness. Local and server microphone mute override both modes.
Automatic mode is enabled only during an explicitly started call; it does not
start listening at boot. The mode is sent as a per-call preference and does not
change the account's saved provider, billing source or input preference.

On-screen captions retain the latest user and Bot segments separately, using the
authenticated NyxID snapshot's `speaker` field. User speech is pale blue and Bot
speech light violet, stacked above the bottom gesture area with dark outlines
and no role prefixes. Unknown roles are ignored and long Chinese text is
truncated on UTF-8 boundaries. The shared bounded caption history retains each
pair together. A small Noto Sans SC fallback supplies common spoken fillers
missing from the main subset, including 嗯、呃、噢、欸、诶. These captions
are presentation only; NyxID's authoritative sideband still controls transcript
persistence and tool admission. AEC reduces physical loudspeaker echo; speaker
labels themselves are not acoustic speaker recognition.

Voice playback applies a default 4× digital gain with a soft limiter. The actual
post-gain samples feed the AEC reference. Settings still controls hardware output
volume. USB `nyx-voice gain <1-8>` adjusts and persists the voice gain for this
server profile, including during a call. Status reports pre/post peak sample
levels; raw PCM and provider event contents are never logged.

NyxID ends calls after its existing inactivity/time limits. A server-confirmed
closed call with a known normal end reason returns to Ready even if socket close
races the final snapshot; lost/revoked authorization remains an error.


## Remote calls and settings

See [API.md](API.md#daily-and-management-services-v0615) for the separately
permissioned daily and management services. `POST /voice/open` starts automatic
two-way voice with the selected Bot, using the panel's own login Key; it shows the
controls and optionally queues an opening notification. It does not change the
saved input mode. `/voice/say`, `/voice/microphone` and `/voice/stop` control that
call. Remote retries use persistent bounded request-ID deduplication. The
notification channel is distinct from user utterances and never authorizes a
Bot action. `/voice` and `/status.voice` expose call and resource progress.

Daily access can change saved input mode, hardware volume and digital gain;
mode changes require an idle call. Only management can change Bot selection.
Both services are ordinary UserServices on the same node;
voice access from the board Key remains a separate per-Bot grant in NyxID.


Software restart validation uses the configured C6 reset GPIO before Hosted SDIO
setup. The live Panel and USB recovery console initialize before the network
transport so Wi-Fi hardware setup cannot hold the picture page offscreen.
