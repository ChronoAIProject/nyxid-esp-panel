# Multi-board isolation goal

Started 2026-11-16. This is an implementation and acceptance checklist, not a
claim that the complete isolation work has passed.

Support additional boards without changing the behavior of existing boards.
Share application and service logic; select hardware implementations through a
single board profile. Keep physical board wiring separate from chip revision,
server environment and wake model selection.

## Required boundaries

- One authoritative profile catalog supplies board identity, supported chip
  revisions, configuration defaults, hardware backend and release identity.
  Build tools, packaging, runtime diagnostics and CI consume that definition.
- Display initialization/timing, reset/backlight, touch transport, codec setup
  and capture-slot normalization belong to the selected hardware backend.
  Common UI and services use the existing display/audio interfaces.
- Each profile and local/production environment has its own sdkconfig and
  build directory. A cached configuration for a different profile must fail
  validation before producing firmware. Never infer wiring from chip revision.
- Runtime status and onboarding report the configured board truthfully.
  Capability descriptions distinguish fitted hardware from optional peripherals.
- Every produced image carries its profile identity. Packaging and both URL
  and raw-image OTA reject a different or missing board identity. Flash tooling
  checks the intended device and image profile and preserves persistent data.
- CI derives its supported-profile matrix from the catalog and exercises
  rejection of mismatched configurations, images and package labels.
- Adding a board has a documented path: define its profile and backend,
  integrate its capabilities, then pass build and physical acceptance gates.
  Do not claim support for arbitrary hardware without those steps.

## Acceptance

- [x] Centralize the existing external-screen rev1/rev3 and integrated LCD-X8 profiles.
- [x] Confine hardware-specific implementations to the board layer.
- [x] Enforce configuration/build separation and negative mismatch tests.
- [x] Report correct board identity and hardware capabilities.
- [x] Enforce image identity in packaging, flashing and every OTA entry point.
- [x] Build all supported profiles with CI-equivalent configuration locally.
- [x] Pass GitHub CI at the reviewable branch head.
- [x] Regress rendered pages, brightness, audio capture/playback and voice/wake
  ownership on the attached 10.1-inch and 8-inch development boards.
- [ ] Confirm physical scrolling/touch and intermittent optical display behavior
  after the isolation change. The earlier user observation predates this build.
- [x] Document verified evidence, outstanding limitations and how to add a board.

Preserve device credentials, Wi-Fi, persistent artwork and independent
local/production NVS profiles. Preserve one I2S → AEC → AGC → ordered FIFO
pipeline and natural audio speed. Do not touch the dining-room production panel.

## Existing evidence

On 2026-11-16 the user found both attached home boards responsive. UART
home/launcher/home comparisons exposed repeated Nyx Panel color invalidation
in the main-based LCD-X8 firmware: idle software drawing used 35–44%, while
the legacy c161 research image already avoided identical color updates.
Applying the same guard to the common UI reduced all four LCD-X8 drawing
samples to 0%, without watchdog, panic or DSI underrun. Twenty-two host suites
and thirteen release/tool tests passed. This does not establish the cause of
the separately reported office-board lag; its preview already had the guard.

Raw logs, firmware hashes and app-only flash/persistence verification are in
the local `output/panel-dual-compat-20261116` evidence directory. Keep device
logs and backups out of the repository.

## Isolation implementation and regression progress

The catalog now drives the three production profiles, separate profile/environment
build directories, BSP dependency/adapter selection, product audio backends,
generated runtime metadata, release packaging, preserving USB updates and CI.
LCD-X8 BSP edits apply to a build-local copy, including recovery from an older
managed-component copy patched in place. External-screen builds use the
unmodified BSP copy. The DPI/LVGL transport backport belongs to that BSP adapter.
Native applications receive its interface through `product_support`.

Twenty-five C host suites and twenty-three Python tests pass. These include each
profile's actual C image verifier, all profile-pair mismatches, stale wiring
flags, malformed/truncated image headers, revision/device checks, OTA metadata
and a real CMake rejection before SDK discovery when a build directory belongs
to another profile. All three production configurations built and packaged;
the two attached hardware profiles also built against the current local server.
This is local build evidence, not a claim of a completed GitHub CI run.

Both development boards passed app-only USB write and protected-region hash
verification. Wi-Fi, existing local/production namespaces and persistent artwork
were retained. The legacy USB bridge required lowering backup reads to 115200
baud after a read failure before any write; the flasher now separates backup
read speed from write speed. No production panel was accessed.

On LCD-X8, a full matching-profile raw OTA with its first 400 bytes deliberately
split over multiple receives switched app slots and passed the startup health
check. A request advertising a full multi-megabyte incompatible image was
rejected after its fragmented prefix with HTTP 422 and connection closure;
the running image and slot were unchanged. The canonical local images also
passed startup health verification. On the legacy board, URL OTA rejected the
LCD-X8 production image despite its correct digest, then installed the matching
local image into the other slot and passed the startup health gate. The retained
OTA UART traces show the ESP-Hosted peripheral reaching `Received INIT event`
and `Base transport is set-up` after the captured reboots. Those successful
traces also contain WebSocket timeouts to the historical `.57` backend. A
separate experimental receipt reports post-OTA connectivity loss and rollback,
but no retained failing transport trace has been correlated with that event.
Its cause remains unconfirmed; the successful traces do not resolve it or
establish a board-isolation regression.

Both boards completed an existing English acoustic fixture with the complete
user directive and an audible-response data path: microphone and output peaks,
sent/received packets and playback time were nonzero, and the transcript reply
was "purple elephant". Neither call produced another user turn during ten
seconds of subsequent quiet. Mute/unmute, hangup, wake-listener rearm and a
second connected call passed. One immediate legacy redial raced the backend's
asynchronous shutdown and was rejected; reconnect succeeded after that session
closed. This is a recorded lifecycle limit, not a fix in NyxID or a long-call
echo-cancellation qualification. The failed attempt's requested microphone mute
was explicitly restored and checked.

UART navigation through home, launcher and settings returned ESP_OK on both
boards. Screenshots had intact text; all four sampled idle software-drawing
loads per board were 0%. No watchdog, panic, DSI underrun or touch-bus error was
observed in these bounded runs. UART opening deliberately restarted each board;
these restarts are excluded from stability-failure conclusions. Remote
navigation does not exercise a physical finger, nor does a framebuffer JPEG
prove that the panel never flashes. The X8 native transition check then opened
SpecAnalyzer and MusicPlayer, verified that each acquired the shared codec
(`audio_busy`), switched back to Nyx Panel, and confirmed `wake=listening` plus
a muted voice connection without closing the native app first. MusicPlayer was
resumed after that switch, reacquired the codec, and released it again on the
second switch. Closing each app afterwards returned `ESP_OK`; no panic,
watchdog, I2S read error or cleanup error occurred. This check used the
experimental lifecycle image
`1614064f3fcc5aa4ea2db37c755dd567be775317dd94149eb1bcbeede758e1c8` (ELF
`554ab0f3028dede17a2b4d3399a388499dc95f8703baba57326165a9f8342cb7`). The
board was restored to the canonical X8 image after the check.

The legacy rev1 board was then tested with a matching local `.57` build
(`p4-rev1-16mb`, app SHA `38a0330fdb0cea585a658d3564d7a658ecbaecb293ed36151149dd358121b62e`,
ELF SHA `9f1f2953dc639a90617222aff587cd0e5a620a55563c9d293be57441833c0801`).
Its pre-existing local Bot/service configuration loaded with `configured=true`
and `wake=listening`; Settings/Home navigation, SpecAnalyzer ownership and
MusicPlayer pause/resume/release all passed. The run observed no watchdog,
panic, I2S read error or cleanup error. The exact historical c161 image was
restored afterwards with the protected NVS/storage hashes unchanged.

Sanitized hashes and bounded observations are recorded in
[the regression receipt](evidence/2026-11-16-board-isolation.json). Private raw
logs and failed attempts are retained alongside successful runs rather than
counted as passes. The office-board lag was not reproduced on the attached
home boards and is not claimed fixed.

The follow-up legacy-profile/native-audio receipt is
[evidence/2026-11-17-legacy-isolation.json](evidence/2026-11-17-legacy-isolation.json).

After its temporary main-based regression, the legacy board was restored to
the exact c161 research image, with app and ELF hashes verified and protected
NVS/storage hashes unchanged. Hey Nyx listening resumed. The historical image
still targets the previous local-server LAN address; successful voice regression
used the canonical isolation image at the current address. The LCD-X8 remains
on its canonical isolation build. The regression used Hi ESP and was not a
wake-model comparison. Runtime memory samples and UI navigation cannot
substitute for a physical touch/visual or microphone/playback check.

Extension and recovery contracts are documented in [BOARD_PROFILES.md](BOARD_PROFILES.md).
Current profiles cover ESP32-P4 and the existing 16 MiB layout; another chip or
partition layout needs a reviewed contract extension and physical acceptance.

## GPU delivery monitoring

Continue the existing hourly job, Issue #13 and the
`research-hye-nyx-voice-gpu` branch. Inspect useful deliveries and the original
development panel's stability state once per scheduled round. Do not create
another schedule or densely poll. Test useful compatible candidates through
the existing reversible development-board procedure, preserving the current
working fallback and recording the board/profile for each result.

No new training, duplicate acoustic baseline, TTS playback or recording is
authorized by the hourly check. Judge delivered candidates by practical
behavior; do not add a numerical-equivalence requirement.

Latest reviewed GPU head: `f84a255dc3a14a9cc9d305401fbc327687f3943f`.
The zf candidate and its 6000-step follow-up were rejected: corrected independent
Sol replay produced 0/24 events, the 3000-step candidate's fullwave recall was
0.0208, and the background exposure was below the preregistered FAR bound. The
Hi ESP QEMU check passed only as a transport control and is not a matched quality
comparison. The later Piper source-overlap audit also found no fresh non-overlap
source lock, so it produced no deployment artifact. Issue #13 has no approved
replacement; frozen v30 remains the only approved handoff. No research model was
installed. The existing hourly task remains enabled.

Kokoro v1.1 now has 144 development recordings and 72 isolated acceptance
recordings that passed the corrected 24 kHz native/16 kHz resampling audit. The
CUDA embedding cache and a linear frame-classifier diagnostic completed. The
latter reports 91.88% INT8 frame accuracy; it is not a deployable streaming
wakeword model. Frozen v30 replay on these development streams activated on
28/48 positive and 20/96 negative recordings. These recording activations use
the existing raw threshold and are diagnostic only, not a new operating point
or physical wake accuracy. No candidate was delivered for a board trial.
