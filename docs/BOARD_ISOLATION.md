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
- [ ] Pass GitHub CI at the reviewable branch head.
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
local image into the other slot and passed the startup health gate.

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
prove that the panel never flashes. Native MusicPlayer/SpecAnalyzer transitions
were not independently exercised after this change; their compile/dependency
checks and the voice/wake ownership test are separate evidence.

Sanitized hashes and bounded observations are recorded in
[the regression receipt](evidence/2026-11-16-board-isolation.json). Private raw
logs and failed attempts are retained alongside successful runs rather than
counted as passes. The office-board lag was not reproduced on the attached
home boards and is not claimed fixed.

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

Latest reviewed GPU head: `67288dab25f2209ebbaf0f71499d4860a665cc9b`.
Frame normalization and MFCC style augmentation were rejected before INT8
export (development targets 51/144 and 65/144, negative activations 19/720 and
39/720 respectively). Issue #13 also reports a rejected SpeechT5 coverage run:
97.22% target recall with 37.85% confusion activation. These are development
diagnostics, not physical-board acceptance. No useful replacement was delivered
or installed. The existing hourly task remains enabled.
