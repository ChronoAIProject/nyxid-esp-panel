# Panel debugging and physical acceptance

This is the reproducible entry point for firmware, local/production NyxID,
acoustic voice tests, camera/display diagnosis and release verification.
Run commands from the repository root. Replace the example variables for the
**currently attached board**; a home/office move can change the board, USB port,
IP, Wi-Fi and server address independently.

## Identify the board and retain evidence

```sh
export PANEL_PORT='/dev/cu.YOUR_USB_SERIAL'
export PANEL_HOST='192.168.1.100'
export PANEL_SERVICE='nyx-panel-DEVICE'
export PANEL_CONVERSATION='nyxa-CONVERSATION'
export PANEL_DEBUG_DIR="$PWD/.local/debug"
mkdir -p "$PANEL_DEBUG_DIR"
chmod 700 "$PANEL_DEBUG_DIR"

python -m esptool --chip esp32p4 --port "$PANEL_PORT" chip_id
curl --fail "http://$PANEL_HOST/status" > "$PANEL_DEBUG_DIR/before-status.json"
nyxid service show "$PANEL_SERVICE" --output json
```

Check silicon revision/MAC against the intended board before writing flash.
`/status` identifies firmware, running app ELF SHA256, network, active OTA slot,
Nyx node, display/image, camera, heap and voice binding. Wait for
`net.connected=true`, `time_valid=true` and `nyx.state=up` before a signed-service
test. An IP response alone does not establish which USB board is attached.

Use a private ignored directory for UART logs, NVS backups, audio, screenshots
and histories. UART echoes typed commands: do not record credential-entry
commands. Do not commit `.env` files, device/owner/provider credentials, NVS
images, Wi-Fi QR codes or recordings of people. This guide and the tools contain
no machine-specific credentials. Historical acceptance scope is in
[VERIFICATION.md](VERIFICATION.md).

## Build, profiles and flash

Use ESP-IDF v5.5.4 and the checked-in component-manager pin:

```sh
. /path/to/esp-idf/export.sh
python -m pip install -r tools/requirements-build.txt
idf.py build
cmake -S tests/host -B build-host
cmake --build build-host -j2
ctest --test-dir build-host --output-on-failure
python -m unittest discover -s tests/release -v
git diff --check
```

The ordinary build targets the legacy revision-1 board. CI also builds/packages
the revision-3 profile using `sdkconfig.rev3_x.defaults`; do not flash that image
on revision 1. Compare `build/CMakeCache.txt`, `sdkconfig`, the IDF revision and
`dependencies.lock` when local and CI differ. Run `idf.py reconfigure` after a
component/source-list change. Component Manager older than the pinned version
can produce misleading missing-Kconfig failures from unselected candidates.

Local **voice only**, with the node still on production:

```sh
export PANEL_DEV_HOST='192.168.1.101'
idf.py -DNYX_PANEL_NODE_DEV_MODE=OFF \
  -DNYX_PANEL_AGENT_DEV_API_URL="http://$PANEL_DEV_HOST:3001" \
  -DNYX_PANEL_AGENT_DEV_LOGIN_URL="http://$PANEL_DEV_HOST:3000/login/agent-key" build
```

For a fully local test, change `NYX_PANEL_NODE_DEV_MODE` to `ON`. Voice Key,
selection, node identity and pending setup use separate server profiles. The
panel must perform its own local login and node setup; changing the URL does not
authorize the device or copy production credentials. Wi-Fi/default image stay
on the device. See [VOICE.md](VOICE.md#local-nyxid-development) and
[ONBOARDING.md](ONBOARDING.md).

Always clear **both URL overrides and node development mode** for production:

```sh
idf.py -DNYX_PANEL_NODE_DEV_MODE=OFF \
  -DNYX_PANEL_AGENT_DEV_API_URL= -DNYX_PANEL_AGENT_DEV_LOGIN_URL= build
```

Close UART monitors before flashing. For an ordinary update, preserve NVS and
the partition table. Prefer the authenticated OTA flow in [UPDATES.md](UPDATES.md).
For app-only USB debugging, first read `ota.running` from `/status` and use the
matching offset in this repository's current `partitions.csv`:

| Running slot | App-only address |
| --- | --- |
| `ota_0` | `0xe0000` |
| `ota_1` | `0x770000` |

```sh
# Example ONLY after confirming ota_0 on this board and this partition table.
python -m esptool --chip esp32p4 --port "$PANEL_PORT" --baud 1500000 \
  write_flash 0xe0000 build/nyxid-esp-panel.bin
```

Wait for reboot, then compare `/status.app_sha256` with the SHA256 of the built
ELF (`shasum -a 256 build/nyxid-esp-panel.elf`). This is **not** the binary SHA256
used by the OTA manifest. `esptool image_info` validates the image checksum.
Neither `erase_flash` nor a replacement NVS/storage image is needed for routine
debugging. Full vendor-layout migration and exact rollback are documented in
[README.md](../README.md#back-up-and-migrate-an-existing-board); use only this
exact device's private backup and `tools/extract_runtime_nvs.py`.

## UART and local audio isolation

```sh
python -m pip install pyserial
python tools/monitor_uart.py --port "$PANEL_PORT" \
  --output "$PANEL_DEBUG_DIR/uart.log"
```

The monitor sets DTR/RTS low before opening and appends raw bytes to a mode-600
file. Some USB adapters still reset on open. Ctrl+C closes it; opening a second
monitor or flashing concurrently is invalid. For interactive console work use
`idf.py -p "$PANEL_PORT" monitor` (Ctrl+] exits), without the recorder running.

Useful commands, entered on the console:

```text
nyx-status
nyx-login status
nyx-voice status
nyx-voice peer-test
nyx-voice refresh
nyx-voice list
nyx-voice select <authorized-Bot-UUID>
nyx-voice mode automatic
nyx-voice start
nyx-voice mute
nyx-voice unmute
nyx-voice stop
```

`peer-test` isolates local peer/SDP preparation from a provider call. To inspect
push-to-talk, select `mode hold`, then `start`, `hold`, `release`, `stop`; restore
the original saved input mode afterward. Do not issue login/select/config
commands merely to inspect a working profile. `factory-reset` is destructive
and is not a diagnostic step. See [VOICE.md](VOICE.md) for the complete console
and codec behavior.

## Start a local NyxID without a credential bridge

Use an existing NyxID checkout, normally under the developer's home directory;
record its commit and branch. Check for already-running services before starting
another instance. Use a separate Mongo database/profile for the test.

```sh
mongod --dbpath /private/path/panel-mongo --port 27127 --bind_ip 127.0.0.1 \
  --replSet nyx-panel-voice --wiredTigerCacheSizeGB 0.5
```

Initialize a new replica set once using that Mongo installation's normal
procedure; do not reinitialize an existing database. Configure NyxID using its
current `.env.example`, with the corresponding Mongo address, port, browser
origin and provider settings. Keep the local env file outside version control.
Then, from the NyxID checkout:

```sh
set -a
. /private/path/local-nyxid.env
set +a
export RUST_MIN_STACK=16777216
cargo run --manifest-path backend/Cargo.toml --bin nyxid-server
```

The debug backend used a 16 MiB worker stack in this investigation. Omitting the
existing stack setting caused a startup stack overflow during node registration;
that failed launch is not voice test evidence. Bound the Mongo cache instead of
leaving a temporary test server to consume memory indefinitely. Record PIDs and
stop **only the services this run started**, leaving unrelated services alone.

Grant both directions independently: panel Key -> the chosen Bot's exact Voice
service; chosen Bot's Key -> this panel's daily service. Management is a separate
grant. Provider credentials live on NyxID, never in panel firmware. The Mac only
plays test audio and observes owner-visible state/history; it does not forward
microphone audio or lend its Key to the panel. The same acoustic tools below can
use a separately logged-in local owner CLI profile via `--profile panel-local`.

## Verify service discovery before blaming audio

```sh
nyxid proxy discover --output json
nyxid mcp call nyx__list_connected_services --args '{"query":"tablet"}' --output json
nyxid mcp call nyx__search_tools --args '{"query":"tablet volume"}' --output json
nyxid proxy request "$PANEL_SERVICE" /config/audio --output json
nyxid proxy request "$PANEL_SERVICE" /openapi.json --output json
```

Check executable state, the exact connection ID/node, method/path/input schema,
and daily versus management access. The hosted typed contract and the device's
live OpenAPI are different observation points. Changing a local schema does not
deploy the hosted catalog. A 403 is an authorization or request-validation
failure; it is not evidence that audio packets were lost.

Custom connections currently have no separately configurable service description.
An older panel's description can therefore be null. Tool discovery includes the
connection name in each operation's description. If ordinary device terms find
nothing, give the **existing** connection an informative owner-selected label:

```sh
nyxid service update "$PANEL_SERVICE" \
  --label 'Home Nyx Panel — voice tablet / board / 平板 / 板子' --output json
```

Repeat discovery and the same ordinary spoken request afterward. Do not fix a
metadata ambiguity by requiring users to recite a UUID, adding a board-specific
Voice persona, granting extra services, or claiming the wrong device's state.
New panel registrations already include ordinary device terms; existing owner
labels remain untouched by firmware upgrades.

## Prepare and independently check acoustic fixtures

Use a nearby Mac speaker and the physical board microphone. Keep speaker
placement/output volume fixed and record them with the test. One run used a Mac
about half a metre away; this is a setup description, not a calibration standard.
Run one playback/test at a time and distinguish test speech from nearby people.

The macOS system Chinese TTS used early in this investigation had ambiguous
pronunciation. Its source itself was misrecognized, so Chinese exact-phrase
failures could not establish a firmware defect. Final acceptance used natural
English at the owner's request. Test source quality **before** acoustic capture.

`voice_fixtures.py` creates normal-speed English WAVs through the speech API and
independently transcribes every source. It rejects missing words, including
“Please repeat exactly”. The provider can occasionally speak only the quoted
target; that is a bad fixture, not proof that the panel dropped the prefix.

```sh
set -a
. /private/path/openai-key.env
set +a
python tools/voice_fixtures.py --output-dir "$PANEL_DEBUG_DIR/audio" \
  --wake-word /private/path/known-good-hi-esp.wav
```

Use a clearly spoken, separately validated Hi ESP recording. If converting a
recording, `afconvert -f WAVE -d LEI16@16000 -c 1 input.aiff output.wav` changes
sample format, **not playback speed**. Supply it with `--wake-word`; the tool
concatenates it directly with the initial directive without adding a pause.
Without a wake recording it still generates/checks follow-up fixtures, but does
not create `wake-en.wav`. Existing sources can be checked without regeneration:

```sh
python tools/voice_fixtures.py --check-only --output-dir "$PANEL_DEBUG_DIR/audio" \
  --wake-word /private/path/known-good-hi-esp.wav
```

This optional source-generation/check tool uses the operator's environment key
only for preparing test WAVs. It does not configure CookieBot or the board.
Model/voice are overridable; consult the current official
[speech guide](https://developers.openai.com/api/docs/guides/text-to-speech) when
changing them. Keep manifests, source transcriptions and hashes with the run.

## Physical voice acceptance

The CLI must be logged into an owner account that can read the selected
conversation and control the panel daily service. Get the actual conversation
from `nyxid nyxbot conversations`; never hardcode another board's conversation.
Wait until the panel is idle, unmuted, and listening for Hi ESP. The optional
LAN URL below is read-only status observation; actions still use NyxID.

```sh
python tools/accept_voice.py --scenario full \
  --service "$PANEL_SERVICE" --conversation "$PANEL_CONVERSATION" \
  --fixtures "$PANEL_DEBUG_DIR/audio" --status-url "http://$PANEL_HOST/status" \
  --output "$PANEL_DEBUG_DIR/full-voice.json"
python tools/accept_voice.py --scenario remote \
  --service "$PANEL_SERVICE" --conversation "$PANEL_CONVERSATION" \
  --fixtures "$PANEL_DEBUG_DIR/audio" --status-url "http://$PANEL_HOST/status" \
  --output "$PANEL_DEBUG_DIR/remote-voice.json"
python tools/accept_voice.py --scenario boundaries \
  --service "$PANEL_SERVICE" --conversation "$PANEL_CONVERSATION" \
  --fixtures "$PANEL_DEBUG_DIR/audio" --status-url "http://$PANEL_HOST/status" \
  --output "$PANEL_DEBUG_DIR/boundaries.json"
```

The full scenario checks the **complete initial directive and reply**, follows
up with counting, interrupts during actual playback, reads current speaker
volume through the Bot, checks exactly four user turns after quiet periods, and
plays speech while muted to check it is not replayed after unmute. Remote opens
automatic two-way voice, observes the existing greeting preference, then checks
the same generic tablet-volume request and one real user turn. Boundaries
temporarily disables wake, verifies no call, reenables it, cancels while still
connecting, verifies an empty queue/rearmed listener, and repeats hangup.
Each tool hangs up its own call and writes failure evidence as well as passes.
Inspect `cleanup_error`; do not call a failed cleanup a pass.

Without `--status-url`, signed polling is limited to about one request per
second. The firmware retains 128 replay nonces for 60 seconds: fast status
loops can exhaust the cache and produce 403, even while voice works. Do not
disable the replay guard or interpret a harness 403 as an audio failure.

For the simpler one-turn check, the existing `tools/accept_wake.py` accepts
`--service`, `--conversation`, `--audio`, `--output`, `--expected-input` and
`--expected-reply`. Require both directive and reply; a noun alone is insufficient.

Correlate the session ID, `voice.sealed/complete` transcripts, UART timestamps,
and actual `haCam.api: route method=GET path=/config/audio ... status=200` (or the
state-read operation the Bot actually used). A matching spoken number alone
does not prove a fresh service read. Test a manual icon tap separately: immediate
Connecting feedback/tone, Connected timer, automatic/hold input, hearing while
holding, and red Hang up. A remote open is not a physical touch test.

## Isolate faults and measure latency

| Observation | Next evidence |
| --- | --- |
| No wake detection | `wake.enabled/listening`, native audio/OTA ownership, fixture pronunciation/level, WakeNet model load logs |
| Wake detected, no first directive | Source transcription first; detection timestamp/lookback, FIFO depth, capture continuity, complete sent/decoded PCM |
| No active call | Board's own profile/grants, fresh discovery/options, certificate/ICE/DTLS/data-channel events, error and session ID |
| Bot asks for repetition | Compare source and received user speech before changing prompts or subtitles; captions only display server speech |
| Extra user turn while room is silent | Correlate actual speaker output, mic/reference capture and timestamps; do not assume echo from transcript words alone |
| Delay or extra speech after control reconnect | Check the same session/generation, live-session count, and FIFO depth during the gap; initial wake buffering must not restart after readiness |
| Wrong device or “volume unavailable” | Discoverable connection name, operation descriptions, exact granted service and actual read route |
| 403 during a test | Signed polling cadence/nonce cache and live Key scope; distinguish these from provider media errors |
| Device works but Web fails | Browser microphone permission, selected source, WebRTC stats/RTP and native Web route tests |
| Local Mongo consumes memory | Duplicate instances, dbpath, WiredTiger cache bound, own process cleanup |

Record wake detection, preparation start/end, certificate readiness, options,
offer/session creation, ICE/DTLS, provider start, first transmitted/received
packet, FIFO depth and first reply. Use one monotonic clock or a clearly aligned
trace. ESP log ticks and `esp_timer` sample timestamps have different boot
origins; subtracting them directly produced misleading timing in an experiment.
Compare like-for-like cold and warm calls. Report samples/medians, provider and
network variation, not an isolated server speedup inferred from a whole call.

The retained audio path is 16 kHz mono, 24 kbps Opus, 20 ms frames, one continuous
I2S/AEC/AGC producer and an ordered FIFO. Do not add a second reader, record the
same lookback twice, reopen capture at handover, locally segment sentences or
change playback speed to make a timing test pass. Sonic/time-compression and
experimental packet pacing were tried, failed acceptance, and were removed.

### Reconnection fault test

Use a separate local NyxID/Mongo profile and a development board with its own
restricted Key. Leave production devices and collections untouched. Start a
physical wake call, verify the complete initial directive/reply, wait for the
initial FIFO backlog to drain, and record its session ID and generation. Keep
the room quiet except for the documented test source.

To interrupt only the control socket while preserving provider/WebRTC audio,
temporarily replace this exact session's `control_socket_id` with a unique test
marker and set `control_socket_until` to eight seconds in the future in the
local `assistant_voice_sessions` collection. Match `_id`, the observed
`generation`, and `live_slot:true`; require exactly one matched document. The
old socket fails its lease refresh and reconnect retries are rejected until the
temporary lease expires. Do not clear the provider lease, create another session,
or disable authentication. Stop the owned call in `finally`, including failures.

Sample read-only LAN `/status` roughly twice per second and save the monotonic
time, `voice.active`, `busy`, session ID, `wake.buffered_ms`, and
`wake.transmitted_ms`. A control interruption should temporarily report
connecting while retaining `busy` and the same session. Verify that the local
owner still has exactly one `live_slot:true` session with an unchanged generation.
After the first readiness, the microphone FIFO should remain empty during the
gap and resume with fresh samples. It must not grow by the outage duration and
then drain after recovery. Repeat during short Bot speech and check for extra
user turns in the subsequent quiet period. Also speak a distinct validated
fixture wholly inside the gap: it must not appear after recovery. A separate
fixture after recovery must produce the expected reply. This tests stale replay;
absence of extra transcript rows alone does not establish acoustic cancellation.

The 2026-11-10 baseline reached FIFO depths of 7.88–8.46 seconds in three
eight-second control interruptions, then took another 19–25 seconds to drain. No duplicate session or
extra user turn was observed in that baseline; delayed PCM replay was established,
but the owner's reported self-echo was not reproduced by those three trials.
Keep private raw microphone/reference/AEC recordings if investigating remaining
acoustic failures. A test fixture spoken while capture is intentionally paused
is expected to be lost; preserving the initial wake request is a separate check.

For a long-call soak, include actual spoken follow-ups inside NyxID's reported
`idle_seconds` limit. Bot announcements alone do not count as user activity;
record the session's `end_reason` before calling a normal idle end a disconnect.
One physical request may span several caption segments. Wait for sealed output
and compare the complete text within the fixture's playback window. The short
fixture tool's row-count assertions are narrower checks, not a general mapping
from rows to spoken turns. Preserve any failed trace and inspect it; never change
product segmentation or filter transcripts to make those assertions pass.

For a genuinely unresolved capture/codec issue, use an **uncommitted diagnostic
build** that copies the exact PCM immediately before Opus encoding and the
independently decoded transmitted Opus to private WAVs. Compare their sample
order, duration and independent transcripts. Raw audio contains private speech;
do not expose it as a permanent LAN API. This investigation found one matching
Chinese PCM/decoded transcript, which did not prove a general packet-loss cause.

AEC diagnosis requires the actual microphone **and hardware speaker reference**,
with single/double-talk and quiet controls. Keep source placement constant. A
volume-80 control did not fix an old interruption fixture, so volume/AEC was not
established as its cause; no AEC parameters were changed. Do not filter extra
transcripts to conceal an acoustic problem.

For vendor-library transport errors, compare the exact managed component/version
and disassemble the relevant archive with the IDF toolchain's
`riscv32-esp-elf-objdump -d <archive>`. The bundled DTLS callback used the same
synthetic return for an empty-read timeout and socket error; interpreting it as
a parsed peer close reset handshakes. The checked adapter preserves WANT_READ
semantics while real mbedTLS alerts and normal deadlines remain authoritative.
Changing the original receive timeout to 5 ms caused failures and was reverted.

## Camera, display, persistence, OTA and Web regression

Use the existing acceptance tools rather than create a new streaming pipeline:

| Tool/reference | What it verifies / changes |
| --- | --- |
| `tools/accept_camera.py --help` | Native preview/photo/back, capture, JPEG dimensions and camera resource recovery |
| `tools/accept_hardware.py --help` | Display pushes, invalid images, native navigation, camera, onboarding cancel; changes displayed content |
| `tools/accept_media.py --help` | Camera/media arbitration and status; see tool options for active operations |
| `tools/accept_background.py --help` | Reboots, replaces/deletes persistent fallback and finally restores built-in art; use a test board or explicitly disposable default |
| `tools/font_decode_probe` and [DISPLAY.md](DISPLAY.md) | Real compressed-font decoder concurrency, serialized baseline, DMA-selected screen snapshots |
| [API.md](API.md) | Daily/admin operation boundary, validation payloads, persistent image, brightness/audio, camera return/display identity |
| [UPDATES.md](UPDATES.md) | OTA manifest/hash/profile, app slots, idle gating, boot confirmation, rollback and blocked version |

When diagnosing missing glyphs, compare the actual LCD screenshot with a photo
of the physical screen; an independently redrawn object tree can hide scanout
faults. The established fixes use serialized font decoding and DMA framebuffer
retirement. Keep the original Waveshare DSI timing. Camera preview uses the
vendor OV5647 profile; compare light/exposure before treating colour as a buffer
problem. On-demand capture and native live preview are different paths. The
sensor JPEG route is not the LCD screenshot route.

In a matching NyxID checkout, run the Voice tests and frontend voice unit/E2E
tests named by that checkout's package scripts. Use its CI configuration for
Mongo fixtures and worker-stack limits. For a real browser regression, allow
microphone access, select the intended source, complete a request and reply,
and retain both user/assistant transcripts plus inbound **and** outbound RTP
stats. Mocked frontend tests alone do not prove a provider WebRTC connection.
The 2026-11-09 server run passed all 178 Voice tests, workspace/all-target Clippy,
formatting, and a real Web Voice/RTP regression.

## Finish the run

Save source commits, build profile, app ELF/binary hashes, test-source hashes,
tool arguments, reports, actual route results and rejected hypotheses. Keep
failed runs; explain which assertion failed without recasting them as passes.
Inspect new UART panic/assert/watchdog/underrun messages and compare before/after
heap only in the same idle/native-page state.

Restore the intended local/production profile and verify selected Bot/service,
network, saved mode, volume/gain, greeting, wake/mic, default image and OTA state
against the original snapshot. Leave voice idle unless the operator wants a
call. Stop playback, UART and the backend/Mongo processes this run started.
Keep logs/audio private; commit methods and sanitized findings, then push the
existing PR and observe its current-head checks:

```sh
gh pr checks <PR> --watch --interval 20
python tools/package_release.py --build build --output artifacts/debug-package \
  --profile p4-rev1-16mb
```

A built package, green CI or a merged firmware PR does not publish an OTA release
or deploy NyxID. Do those only as separately requested operations. Physical
revision-1 acceptance must not be presented as revision-3 hardware validation.
