# Firmware and hardware verification

## v0.6.15 standalone voice and service scopes, 2026-11-05

The home panel runs v0.6.15 against local NyxID with
[NyxID #2169](https://github.com/ChronoAIProject/NyxID/pull/2169). Its own child
Agent Key is limited to the selected Bot's published Voice. That Bot separately
has the board's daily service and no management service. No Mac identity or
audio bridge is part of the device call. Production use requires those backend
changes, the default-off `assistant:voice-agent-key` feature enabled, and the
owner's explicit publication/grants; firmware publication alone does not enable
them.

Physical checks completed with that installed firmware:

- Remote voice: **33 checks passed**, covering open, opening notification and
  received audio, microphone mute, duplicate/conflicting IDs, hangup, saved-mode
  preservation and idempotent retries after hangup.
- Configuration and permission boundaries: **89 checks passed**, covering
  volume/gain, brightness, input mode, invalid/unknown/duplicate fields, daily
  denial of management operations, camera return/display identity, persistent
  fallback across reboot and restoration of the original settings. Requests
  were paced at 0.7 seconds; faster initial batches hit the existing replay
  guard. Its 128-entry cache was not disabled for the passing run.
- Real microphone input: one compound Chinese request captured, described and
  displayed one photo without asking for repetition. The JPEG supplied to the
  model and shown on the panel had matching byte counts and CRC. Reply playback
  was non-silent. A follow-up changed brightness; the original value was restored.
- Final read-only state: voice idle, built-in NyxBot fallback with no temporary
  image, brightness 80, volume 95, voice gain 4 and saved automatic input mode.
  Camera was idle, continuous publishing disabled and the test calls settled.

The associated backend audit retained `voice-remote-validation.json`,
`daily-config-validation.json`, `camera-display-voice-validation-round-1.json`
and `final-device-readonly-check.json` in its private local evidence directory.
Device identities, credentials and camera images are not repository fixtures.
These are functional checks, not a long-duration acoustic or display soak test.

Before publishing this firmware source, all **20 host test executables** and the
real-font serialized decode probe passed. A separate clean production profile
build with ESP-IDF **v5.5.4**, the default legacy P4/16 MiB configuration and all
development overrides cleared also passed. Its application was **6,375,248
bytes** (`0x614750`), leaving **506,032 bytes** in each `0x690000` app slot.
Binary SHA256:
`8e9f035c89173faa19df9763b1446438ac8d289c77ee3abee212410ce06eb01b`.
The compile commands contain no development voice origin override and the
binary contains neither the active local server address nor the documented
example address. This production artifact was built for publication validation;
it was not flashed over the board's working local-development configuration.

## v0.5.0 startup, persistent background and NyxID colors

Nyx Panel now opens at boot. A validated JPEG can be saved remotely in a single
NVS blob; temporary images and captions stay independent. The built-in fallback
is an 800×1280 composition of existing NyxID brand artwork, its N-mark, and the
default bot name NyxBot, embedded in the application. This is deterministic
branding/layout, not a new AI illustration. No image API/key fallback was used.
See `components/NyxPanel/assets/README.md`.

The NyxID stylesheet uses the vendor's 800×1280 geometry with official brand
colors. Its large object has static storage and is returned by reference to
avoid overflowing the startup stack. Background decoding runs on a temporary
12 KiB task on core 1, alongside hardware/UI initialization, outside the GUI
lock. The service task activates the actual Panel page as soon as UI setup
finishes, without a splash timer or a network prerequisite. Clock sync now
starts in the backend after network connection, independent of visiting Settings,
so a cold boot into Panel can verify signed Nyx requests.

The observed boot prepares the NyxBot image at 2.91 s, shows the Panel at 7.03 s,
obtains a Wi-Fi IP at 11.68 s and synchronizes time at 13.39 s. These are measured
timestamps for the home board, not a claim of instant hardware startup. Image
decoding completes ahead of the UI; no splash timer or network delay gates the
picture. The remaining startup time is the existing hardware/framework setup.
NVS owns its normal lock; flash serialization remains in the flash wrappers to
preserve the same NVS → flash lock order as native Settings.

- IDF v5.5.4 build and all 16 host executables pass. The v0.4.1 app was read back
  and its SHA256 matched the previously installed binary before app-only flash.
- Installed v0.5.0 binary: 5,939,776 bytes, at `ota_1`/`0x770000`.
  Binary SHA256:
  `35a0b4eac8e171092b609975f3bdcd5a5de2fb81b3b0fecbbd67d4900f8b78c5`.
  ELF SHA256, also reported by `/status`:
  `ee8ad8a4a458ffb61fb5593c997129862f97e42a1b0a9e6716ed7f3742078207`.
  The home board retains its WLAN and Nyx identity; only the app was flashed.
- `tools/accept_background.py` passed remote set/get/reset, invalid JPEG,
  oversized bytes/dimensions, wrong content type, LAN exclusion, concurrent
  image/Settings writes, TTL, clear, set/reset beneath a temporary image,
  persistence across reboot, and built-in fallback after reset and reboot.
- Two native page cycles captured Settings, WLAN, device info and NyxID at
  800×1280. Visual inspection confirms readable text, purple controls and dark
  surfaces. Single-worker font decoding and actual DMA framebuffer snapshots
  remain in place.
- Initial camera testing exposed preview frame-lock collisions: the original
  100/200 ms acquisition limit sometimes rejected a snapshot while native UI
  copied its 2 MiB frame. The capture now retries lock acquisition within the
  existing overall fresh-frame deadline. Concurrent requests still return busy;
  the system never starts continuous publishing.
- Final `accept_camera.py --count 3` passes: three remote photos take
  1.985–2.082 s; three concurrent requests produce one photo and two busy
  responses. Native LAN/Nyx snapshots and return to idle also pass. The test
  warms Brookesia's retained 2 MiB recents thumbnail before measuring camera
  heap recovery, with the original leak thresholds unchanged.
- Final native acceptance includes three Camera/photo/Back cycles. Combined
  tests finish with 15 successful captures and zero failed captures. A temporary
  Chinese/English/numeric outlined caption over the NyxBot fallback expires
  back to the byte-identical baseline screenshot. Final state is `home`,
  built-in NyxBot art, camera `idle`, continuous streaming `disabled`, and a
  synchronized clock. UART records no panic, assert, watchdog or DMA underrun
  during this run; this is a short functional regression, not a physical-screen
  soak test.

## v0.4.1 compressed-font concurrency repair

After v0.4.0, the home-board operator still reports missing/garbled letters.
The earlier scanout/JPEG fixes did not remove this separate font-decoder race.
The bundled Noto SC font is compressed, and the two configured LVGL 9.4
software render workers share its unprotected global RLE state. The real-source
host probe in `tools/font_decode_probe` uses separate glyph descriptors and
output buffers for each caller. In the recorded Latin/Chinese run, concurrent
calls corrupt 5,869 of 6,000 decodes; serialized calls corrupt zero. The count
is scheduling-dependent. The default serialized CTest passes. An earlier
ASCII-only probe reproduces the same fault independently.

The firmware now configures one software render worker and rejects a retained
multi-worker configuration at compile time. `/lcd/snapshot.jpg` copies the
actual DMA-selected framebuffer under the GUI lock instead of redrawing the
object tree, so this verification observes already rendered glyphs. Native
screenshots include system overlays and now consistently measure 800×1280.

- ESP-IDF v5.5.4 build and all 16 existing host executables pass, in addition to
  the real-font serialized probe. App-only flash at `ota_1`/`0x770000` verifies;
  the home WLAN, original node identity and slug are retained.
- Installed binary: `0x587e90` (5,799,568 bytes), with `0x108170` bytes spare.
  Binary SHA256:
  `4dbe8646d345b9fb5dc284de7771f6d7324a17fee68b631b22260ccac1ca70e0`.
  ELF SHA256, also reported by `/status`:
  `6ac3552cc1a12f6df0f4dd6d7a40e64ddbd6ff390cbfaae422b2d781e01031b0`.
- `panel-041/results.json`: ten actual static frame captures are byte-identical
  (23,433 bytes), both image orientations cover the canvas edges, and outlined
  Latin/Chinese text, alignment, color and long-text clipping are visually
  checked. The full-frame captures now show the native navigation affordance.
- `acceptance-041/results.json`: complete LAN-screenshot regression passes,
  including text expiry/restoration, invalid-image preservation, two native
  page cycles, scan return paths, Wi-Fi QR confirmation, real device-code
  request/cancel and three native Camera/photo/Back cycles. No account code is
  approved and the test Wi-Fi is never connected. Uptime remains continuous
  through 296 seconds, with three successful photos, zero capture errors and
  Camera idle at the end. One adapter-lock timeout appears in UART during the
  run; subsequent polling completes. No panic/assert/abort or DMA underrun is
  recorded. This is functional verification, not a preview frame-rate benchmark.
- `glyph-stability-041/results.json`: twenty actual Camera frames have identical
  title pixels while live preview redraws, and twenty Nyx Panel frames have
  identical mixed Chinese/English/numeric caption pixels. Final mode is idle,
  with no extra camera-photo requests and continuous uptime through 362 seconds.
  The operator checks the physical home-board screen and confirms that the
  letters are complete and currently normal. This is a short physical
  observation, not a long-duration soak result.

## v0.4.0 full-screen picture and outlined captions

The operator moved home and connected another board of the same model. USB
confirms ESP32-P4 rev v1.3, MAC ending `d2:ef:63`. Its own system/NVS/partition
metadata and running app were backed up privately before an app-only update;
the company board's NVS is not used. Its current partition table matches the
project, with the running `ota_1` app at `0x770000`.

Nyx Panel now hides the status bar, uses an edge-to-edge 800×1280 image with an
aspect-preserving center crop, and overlays the latest caption on transparent
labels. Eight offset bitmap labels provide a 2-pixel stroke for both Latin and
Chinese; `color` selects the fill and `bg` the outline. Temporary captions can
expire back to earlier retained messages. Native navigation stays available.
Both JPEG encoding paths also hand output cache lines to DMA before encoding.

- ESP-IDF v5.5.4 build and all 16 host executables pass.
- Binary size: `0x587fe0` (5,799,904 bytes), leaving `0x108020` bytes per slot.
  Binary SHA256:
  `c02792305e15a780f0a78ff85e12e58b49afb0553ddd8b9702dcf40303b854f3`.
  ELF SHA256:
  `6cab88f36c046bb6b4f7d11015ce4e7df3bd52ad2274b30bfdf7e425e64dc6b6`.
- Final app-only write verifies successfully; `/status` reports the exact ELF
  hash, `ota_1` valid, the original home WLAN/slug, and disabled streaming.
- `panel-040-final/results.json`: ten authenticated 800×1280 static LCD JPEGs
  are byte-identical and decode normally (22,107 bytes each). Portrait pixels
  reach all four corners. Landscape cover reaches the extreme canvas edges;
  upward scale rounding removes the one-pixel seam seen in the first build.
  A round source marker measures 533×535 pixels after enlargement. Chinese and
  English outlines are visually checked on bright/dark halves, with left/right
  alignment, colored outlines and long text clipped inside the safe area.
- `acceptance-040-final/results.json` passes the complete LAN-screenshot run:
  text expiry/restoration, rejected JPEG preservation, two native page cycles,
  scan return paths, injected Wi-Fi QR confirmation/replacement rejection,
  real device-code request/cancel, and three native Camera/photo/Back cycles.
  Uptime stays continuous through 188 seconds; Nyx is up and Camera is idle.
  Wi-Fi QR is cancelled before connecting and the account code is never
  approved. This home-network pass does not resolve the company-network JPEG
  timeouts recorded below. One WLAN page transition emits an adapter lock
  timeout during screen polling; the subsequent poll succeeds and all checks
  finish. Missing SD-card errors remain expected on this board.
- `camera-040-final/results.json` passes using the unchanged 15-second Nyx
  deadline: eight fresh 800×1280 cold photos with the LCD unchanged, concurrent
  status calls, three concurrent photos (one success/two busy), and native
  LAN/Nyx photos followed by Back to idle. Cold calls take 2.035–2.464 seconds
  remotely. The combined run counts 14 successful captures and zero capture
  errors, with idle internal heap unchanged across the cold captures.
- `expiry-040-final/results.json` confirms image expiry preserves the caption,
  native Back exits to the launcher, and reopening restores the same caption
  pixels. Image/text-only operations leave Camera idle without increasing its
  capture count. The board finishes on the full-screen “Hello Lexa” picture
  with Chinese outlined captions. UART records no panic/assert/abort or DMA
  underrun during the final run.

These screenshots establish logical rendering and the tested DMA JPEG handoff.
The company board's earlier physical-text feedback remains the only operator
observation of scanout after the display fix; this home board's intermittent
physical artifacts still need observation over time. The injected Wi-Fi QR
check does not verify camera optics, and SD video playback remains untested.

## v0.3.1 system Camera profile and display synchronization

Reference comparison uses the phone-style `p4-brookesia` checkout, including
its Camera, main display setup, resolved Waveshare BSP and JD9365 driver.
Panel initialization and DSI timing match that reference. The Camera capture
profile is now 800×1280 RGB565, two buffers, SCCB 400 kHz and the native
50 FPS OV5647 sensor mode. This describes the configured sensor mode, not a
measured 50 FPS GUI rendering rate. QR reads an aspect-preserving 200×320 sample.
See [DISPLAY.md](DISPLAY.md) for the source comparison and DMA ownership fix.

- ESP-IDF v5.5.4 build and all 16 host executables pass. The new submission-gate
  tests cover stale DMA selection, repeated completions, alternating buffers
  and completion before arming. The actual DMA callback is exercised on board;
  host substitutes do not reproduce the DSI hardware.
- App-only flash verified on the company replacement board. Its installed binary is
  `0x588100` (5,800,192 bytes), with `0x107f00` bytes spare in each app slot.
  Binary SHA256:
  `b46c8837c97884e72b2852742735753ffca679c3bd60365e93411ea759d92648`.
  Installed ELF SHA256:
  `97ab3f85f53098890c411389ff56ed8e23a7b920a012c852f81d42a4975eab13`.
  USB confirms the existing node identity and company WLAN/IP are retained.
- `camera-031-final-ready/results.json` passes with the unchanged 15-second
  per-Nyx-request deadline: eight fresh 800×1280 remote JPEGs, no LCD navigation,
  concurrent status requests, then three concurrent photos (one success, two
  busy), native LAN/Nyx photos, and Back to idle. Eight cold captures take
  2.516–6.451 seconds remotely, including 1.394–1.414 seconds on board; concurrent
  status takes 0.445–2.079 seconds. All 11 successful photos are counted, with
  no capture errors, reset or retained large capture allocation.
- Initial cold photos were visibly underexposed. Native preview settles brighter
  in the same ceiling scene; a one-second warmup now produces cold photos with
  approximately the same brightness. Artifacts retain that comparison. This is
  exposure-startup evidence, not color calibration or general image-quality QA.
- An earlier build's native Nyx snapshot exceeded the 15-second CLI deadline,
  although UART records a 200 response about 269 ms after dispatch. A first
  final-image attempt began before SNTP completed (42 seconds after boot) and
  received `invalid_signature_or_replay` on the schema request. The passing run
  starts after time synchronization. A separate initial runner started before
  network readiness and timed out. These failures are retained privately;
  the later pass does not establish their general removal.
- Two broad LAN-screenshot navigation runs stop at their 30-second JPEG
  request deadline (`acceptance-031`, `acceptance-031-retry`). UART records
  successful image generation and HTTP sends; the board stays alive. The retry
  completes both full Settings/WLAN/info/NyxID/home navigation cycles before
  timing out on a scan-page screenshot. LAN status probes afterward take
  0.871–1.749 seconds; four pings have no loss but take 170–743 ms. The cause of
  this intermittent transport behavior remains unresolved. The runner now has
  `--snapshots-via-nyx` to exercise JPEGs through authenticated Nyx explicitly;
  that result must be distinguished from the unsuccessful LAN runs.

Logical LCD screenshots cannot prove physical scanout quality. The user's
`IMG_4182.jpg` shows damaged Camera title pixels; the synchronization defect
is fixed, and the operator reports that the updated physical screen's text is
currently normal. This short observation does not establish a complete cure
for intermittent artifacts; a longer soak remains outstanding. SD video
playback and physical camera QR optics remain unverified.

## v0.3.0 on-demand Camera

The operator requested retirement of continuous video publishing. The final
firmware is installed on the replacement board using app-only flash, with a
verified write. Its existing Nyx identity and company WLAN remain active.
A private post-binding NVS backup was taken before flashing (860,160 bytes,
SHA256 `ee1ed741d1c5f10c2860646f5243c0c0f9d1e7171d30018de16344026e4520a9`).
The first flash attempt was interrupted by a prematurely opened serial monitor;
subsequent complete writes verified successfully.

- ESP-IDF v5.5.4 build passes; final binary is `0x587bd0` (5,798,864 bytes),
  leaving 1,082,416 bytes in each app slot. Binary SHA256:
  `55e3183fe265dfa7708ec3c17ce9f82454fdc5d74f88659ffa4aa068c3a32583`.
  Installed ELF SHA256 from `/status`:
  `38437c25e5f96ba9a3b164ec6fbb0a64312f9f8feaa6c5714a414bdc9ca7cfa8`.
- All 15 host executables pass. The new tests run the actual camera-owner source
  with a pthread-based mutex adapter and fake camera driver. They cover idle
  release, preserving native preview, failed-start/encode/cleanup, rejecting LAN
  activation, and concurrent capture versus opening native preview. Hardware
  driver faults/power measurements are outside that substitute's coverage.
- `camera-030-final/results.json`: eight authenticated, fresh 320×240 JPEGs decode
  successfully. Each returns to idle without changing the native screen, with
  continuous uptime. End-to-end calls take 0.768–1.864 s; simultaneous Nyx status
  calls take 0.400–1.140 s. Board capture time is 315–333 ms, including
  startup and teardown. These are short-run measurements on the company WLAN.
- Three simultaneous camera requests produce one valid JPEG and two 503 busy
  responses. Native Camera supports both LAN and Nyx snapshots without closing
  preview; Back returns to idle. The final capture count is 11, with zero failed
  driver captures. Busy rejections do not increment the driver failure counter.
- After warmup, eight idle samples have unchanged internal free heap (238,119
  bytes while response workers settle), with total free heap varying by tens of
  bytes. After native preview closes, free heap is 21,412,780 bytes and internal
  free heap is 241,551 bytes. UART confirms CSI/buffers/JPEG encoder released;
  no continuous publisher is linked or started. This establishes bounded release
  in this run, not a long-duration leak or power/CPU benchmark.
- New `/camera`, `/camera/snapshot.jpg` and `/openapi.json` routes are absent on
  unauthenticated LAN. LAN `/snapshot.jpg` returns 503 in idle. Stream start and
  stream config writes return 410; the compiled-out publisher cannot read or
  activate legacy auto-start settings. Reboot starts idle.
- Full final-firmware regression passed (`acceptance-030/results.json`): pushed
  text/JPEG/expiry and invalid images, two native navigation cycles, scan return
  paths, Wi-Fi QR confirmation/replacement rejection, real login request/cancel,
  and three Camera/snapshot/Back cycles followed by Settings. Uptime remained
  continuous through 226 seconds; the final media mode is idle and Nyx is up.
  Device status was visually checked: the Camera label reads “Remote photos on
  request” and fits the native page.
- Wi-Fi QR JPEG injection followed by explicit confirmation actually reconnects
  and saves the current company WLAN (`wifi-030-results.json`). An unreachable
  hotspot attempt times out and restores the previous WLAN and Nyx connection
  without reboot (`wifi-030-rollback.json`; UART records about 30 seconds from
  disconnect to original-IP recovery). The initiating Nyx request timed out
  during the intentional disconnection; later state confirms rollback.
  Closing the scan page during another unreachable-hotspot attempt also cancels
  the worker, restores the company WLAN/Nyx, and releases Camera to idle
  (`wifi-030-cancel.json`). This tests network behavior using the real decoder,
  not physical camera optics.
- Screenshot-heavy navigation logs include transient LVGL lock acquisition
  timeouts. A separate immediate WLAN→scan request returned 500 while WLAN was
  scanning; the page remained operable afterward. General navigation reliability
  under native Wi-Fi scanning remains a limitation separate from the passed
  Camera call/ownership checks.
- OpenAPI fetched through Nyx is valid JSON, version 3.0.3, with the two Camera
  operations. It is an authenticated schema endpoint; dedicated MCP discovery
  has not been configured with a server-reachable schema URL.

## v0.2.2 native onboarding

The independent upstream/deployed NyxID audit is in
[NYXID_FEATURE_AUDIT.md](NYXID_FEATURE_AUDIT.md); implemented behavior is in
[ONBOARDING.md](ONBOARDING.md).

- All 14 host test executables pass. New coverage checks real account-login
  response/error/backoff semantics and rejection of client identities; standard
  Wi-Fi QR escaping/Unicode/open/hidden/bounds; legacy identity migration,
  failed binding writes, delayed activation and identity-preserving reset.
  The NVS API substitute does not establish physical power-loss durability.
- ESP-IDF v5.5.4 firmware build passes. Application size is `0x673d10`
  (6,765,840 bytes), within each `0x690000` slot with 115,440 bytes spare.
  SHA256: `acdbdbf748b9bfcc595a126b304aa01e3e786394345ca0505c508bfaef4462b5`.
- `git diff --check` and Python compilation/help checks pass. The hardware tool
  adds `--onboarding` for scan/back and a real login request/cancellation, plus
  optional `--wifi-qr-image` for decoder/confirmation checks without connecting.
  Preparing that tool is not live acceptance evidence.
- The original-board OTA attempt stopped at its initial status preflight;
  no OTA upload or reboot occurred. The operator moved to the company network
  and attached a different, same-model board. Public NyxID/CLI access recovered.
- New target: ESP32-P4 v1.3, 16 MiB, MAC `80:f1:b2:d2:ee:ad`. Its full flash was
  backed up privately under `20261103-d2eead`; SHA256:
  `9c0a43240e0b9690f82493a9b6eeaa55f385f6cb4de4443bbb2401381fb7b3e6`.
  Its own NVS was extracted from `0x9000`, size `0x6000`, padded to the new
  `0xd2000` partition. Namespaces are `storage`, `wifi-set`, `stream`; it has
  no existing Nyx identity. No original-board credentials were imported.

- v0.2.0 was flashed with write verification and booted using the new board's
  saved company WLAN (`10.0.1.235`). A real account device-login QR was displayed
  and decoded from its actual LCD JPEG to the trusted NyxID authorization URL.
  Pending cancellation cleared the code. Three Camera/snapshot/back cycles
  passed, but then revisiting Settings caused a load-access fault in
  `PanelPage::refresh`. This is a failed navigation acceptance, not a full pass.
- Root cause: Brookesia paused an active app when resuming an existing app, but
  did not pause it when starting a new app. Settings retained a subpage timer;
  its later root rebuild deleted the timer's label. v0.2.1 pauses the old app
  on both transitions, closes Settings before rebuilding, and binds the panel
  timer to its root object's deletion. Camera acceptance now revisits Settings
  to exercise that exact transition. v0.2.1 passed three Camera → launcher →
  NyxID → Settings cycles with continuous uptime (75–112 seconds), then all
  three Settings/WLAN/NyxID → scan → originating-page paths (130–168 seconds).
  Actual LCD and preview JPEGs were saved for each step.
- The operator approved a real code on v0.2.1. Polling delivered an account
  session, but authenticated HTTP setup failed: ESP-IDF's default 512-byte send
  buffer could not fit the Authorization header. No new node appeared. v0.2.2
  uses a 10 KiB send buffer for bounded account JWTs (maximum 8 KiB), covering
  both registration/service requests and logout. It was flashed with verified
  writes, reconnected, and displayed a fresh real code for approval. The first
  failed grant's RAM tokens were erased; its logout attempts also encountered
  the header failure, so that attempt does not prove session revocation.
- v0.2.2 LAN checks confirm `/nyx/login` and `/wifi/qr` return 404 without Nyx
  authentication. Source `895134f` passed host and real firmware CI in run
  `41606374672`, including the lifecycle/header fixes.

The operator then approved the v0.2.2 code on their phone. Account login,
node registration, service creation, atomic binding save and account logout all
completed. The page showed the saved connection before restart; active memory
remained unbound until reboot. After reboot, authenticated WSS and service
`nyx-panel-d2eead-cf4ad508` were online using the replacement board's own node
`cf4ad508-aef1-4b3c-9af7-7360c22a2f7b`. The custom service was independently
checked for its node binding, active status and `auth_method:none`. The successful
grant left no new device-code account session. This does not revoke the earlier
failed grant described above.

Full v0.2.2 acceptance passed after the old publisher was stopped
(`acceptance-022-idle/results.json`): image/text/expiry and rejected images, two
native page cycles, scan/back to all three origins, real Wi-Fi QR decoding with
SSID confirmation/password hiding/replacement rejection/cancel, real login
request/cancel, and three Camera/snapshot/Back cycles followed by NyxID/Settings.
Uptime stayed continuous. The first run with the old publisher running timed out
reading an LCD JPEG; that failed run remains in `acceptance-022.log`.

At v0.2.2 acceptance, optical scanning and Wi-Fi network transitions remained
unverified. The v0.3.0 checks above establish reconnect and failed-attempt
rollback; optical scanning and switching to a different available AP remain open.

## v0.1.14 original-board evidence (historical)

The original acceptance target is an ESP32-P4 v1.3 / 16 MiB board, 32 MiB
PSRAM, OV5647, JD9365, GT911 and ES8311. Raw logs, registration credentials,
flash backups and camera images remain private outside the repository, under
the operator's dated `nyxid-esp-panel/20261103` output directory. Artifact names
below identify those records without publishing the device's surroundings or
credentials.

The migration is installed as v0.1.14. Build, display/navigation/Camera, reset
and actual A/B OTA checks pass. Media transitions also complete, but strict
control responsiveness and intermittent C6/Wi-Fi stability remain unresolved.

## Build and recovery

- Private repository: `ChronoAIProject/nyxid-esp-panel`.
- ESP-IDF v5.5.4, legacy P4 revision profile, 16 MiB flash; dependency paths are
  portable and the managed-component patches fail closed.
- All 11 host test executables pass, including independent HMAC fixtures with
  hex-decoded signing keys, binary chunk validation and WebSocket fragmentation.
- A separate clean defaults build passed (`build-clean`, v0.1.2); GitHub Actions
  also builds the actual firmware, not just the host tests. Runs `41536695098`,
  `41540218209`, `41541223809` and `41542434054` succeeded. OTA/Flash-fix
  source `0ca728c` passed both jobs in run `41547615184`. Final v0.1.14 source
  `4fdd57d` passed both jobs in run `41551346822`.
- v0.1.14 builds locally: application `0x66bbf0` bytes within each
  `0x690000` OTA slot (148496 bytes spare). Binary SHA256:
  `f4611883bbc063f548055c5dfe2471bab3199e34f5eb19bae3322ccb83e2f235`.
- Complete pre-migration backup: 16 MiB, SHA256
  `b25d05f4e50b90eba16b969bc82a1e516518fa64a07decba5350e1db51f09005`.
- Before reset acceptance, current same-board NVS was separately saved at its
  migrated offset `0x9000`, length `0xd2000`; SHA256
  `36948c8615fc4aa25751bc217b43de1baa56c9a8889b41994ad22e676f57b1d9`.
  This contains the new identity; the original vendor NVS must not replace it.

## Measured behavior

| Check | Evidence / result |
| --- | --- |
| Project boot | Serial identifies `nyxid-esp-panel`; display, touch, ES8311 and C6 initialize. Missing SD card is logged and the desktop still boots. |
| Nyx registration/auth | Runtime UART provisioning, WSS certificate verification, authenticated node registration, heartbeats and actual signed proxy requests succeed. |
| Text/JPEG/CJK | Actual Nyx calls display the Hello Lexa image and `Hello Lexa · 你好，世界`; the 800×1230 app-screen JPEG was decoded and visually inspected. |
| Invalid JPEG / oversize | HTTP 400 / 413; exact screenshot remains unchanged. |
| Expiry / clear / navigation | Clear removes prior content; final v0.1.14 observes a 10-second message during the Nyx request, then returns to the exact baseline screenshot. Unsupported native targets are rejected. The concurrent observation avoids mistaking response latency for an expiry failure (`acceptance-final-v014/results.json`). |
| Three Camera cycles | Final v0.1.14 completes three actual open/snapshot/Back cycles with decodable 320×240 camera JPEGs; uptime remains continuous (`acceptance-final-v014/results.json`). |
| Native Settings/WLAN | Pages render with actual connected/scanned networks. The GUI stack is internal to allow native NVS writes. Repeated navigation found paused Settings retained its WLAN subpage; v0.1.14 closes subpages on pause, rebuilds the root on resume and explicitly resolves remote `settings`. Two full cycles of Settings→WLAN→Settings→WLAN→info→provisioning→home now pass on v0.1.14 without a reset (`acceptance-final-v014/results.json`). |
| QR provisioning | An actual generated QR JPEG passes through the firmware QR decoder and provisioning path, schedules reboot and persists configuration. This checks decoding, not camera optics. |
| H.264/WHIP | Final v0.1.14 completes a 142.6-second sequence with a 30-second LAN client deadline: MediaMTX/ffmpeg decode initial, resumed and final 640×480 H.264 frames; Camera suspends the publisher and Back establishes a new receiver session. Eight actual Nyx status calls succeed; uptime is continuous (`media-diagnostic-v014/results.json`). Earlier v0.1.5 completed a strict 126.8-second / 17-request run (`media-acceptance-v015/results.json`). |
| Publishing control latency | Final v0.1.14 diagnostic Nyx latency is 0.639–19.495 seconds. The 10-second LAN-deadline run failed; this is functional media/transition evidence with known control latency, not a strict stability pass. Earlier v0.1.5 measured 0.408–7.357 seconds. Some RTP loss remains; the scene is very dark. |
| Stream geometry/timing | Sensor 800×800 is center-cropped to 640×480 at (80,160), avoiding incomplete PPA scaling borders. Encoded frames are paced with backpressure and elapsed-time PTS; UART reports about 150 frames/10 seconds. |
| Network interruption | While publishing, deliberate station disconnect recovers Wi-Fi, authenticated Nyx and a new WHIP receiver session in 11.13 seconds. An actual Nyx request and newly decoded receiver frame pass (`stream-recovery-v015.json`). |
| Configuration validation | Quoted/CJK device label and backslash location roundtrip correctly; FPS 256 returns 400 without persisting. Disabled MQTT accepts clearing its old test fields (`config-acceptance-v014.json`). |
| Settings persistence | On v0.1.13, temporary label/location and brightness 63 survive reboot (`persistence-v013-retry.json`). |
| Identity-preserving reset | v0.1.13 authenticated reset restores configuration defaults (including brightness 80), retains Wi-Fi and the Nyx identity, and reconnects successfully (`persistence-v013-retry.json`). See the intermittent C6 startup limitation below. |
| Full P4 NVS wipe / restore | Explicit UART wipe boots the desktop/LAN without Nyx identity, with `credentials_unset`; screenshot inspected. Saved NVS is restored with esptool hash verification; all six config getters equal the pre-test baseline and authenticated Nyx works (`persistence-v013-retry.json`, v0.1.13; also `persistence-final.json`, v0.1.9). C6 station settings survive this P4-only wipe. |
| Raw OTA | v0.1.13 binary moves `ota_1`→`ota_0` and validates after 30 seconds. Fourteen actual Nyx status requests during upload all succeed in 0.415–8.758 seconds; the reboot has its expected connection gap (`raw-ota-concurrent-v013.json`, `raw-ota-final-v013-status.json`). The v0.1.14 UI fix is then installed through raw OTA, switching `ota_0`→`ota_1` and reaching valid/Nyx-up (`ota-0.1.14-status.json`). |
| Raw digest mismatch | Wrong SHA256 returns 422 and leaves the active boot slot unchanged. |
| URL OTA | Actual background update switches `ota_0`→`ota_1`, boots v0.1.13 and validates after 30 seconds. Wrong SHA fails without reboot/slot switch; concurrent update returns 409. During download/write, 20 actual Nyx status calls succeed in 0.489–1.235 seconds (`url-ota-v0113.json`). |

The current baseline has the QA WHIP target cleared, auto-start false and MQTT
disabled with its test credentials removed. Final native
display/navigation/Camera regression passes on v0.1.14.
The complete media/Camera/resume sequence passes with a 30-second LAN client
deadline; the stricter 10-second deadline and intermittent transport failures
remain known limitations. The final actual LCD JPEG was decoded and visually
inspected (`final-display.jpg`).

## Repeating acceptance

`tools/accept_hardware.py --camera --native-pages` performs actual Nyx/LAN
text/JPEG/rejection/expiry checks, two Settings/WLAN/info/provisioning navigation
cycles and three Camera/snapshot/Back cycles. It checks uptime to catch resets.
It does not simulate hardware. `tools/accept_media.py` uses a real receiver,
requires a new receiver source after Camera release, decodes H.264 and measures
Nyx responsiveness. See README and each tool's `--help` for arguments.

`tools/provision_agent.py` imports encrypted node-agent credentials without
printing them. Opening this board's USB-UART pulses reset, so do not open a
second monitor or reset the board during OTA/pending startup validation.

## Limits of the evidence

One v0.1.13 identity-preserving-reset boot failed before GUI/services, with
C6 SDIO `failed to get CIS data` / `Failed to read data`, followed by repeated
slave resets (`reset-v013-private.log`). Restoring the verified same-board NVS
and restarting recovered operation. A complete repeat of reboot, both resets
and backup restoration passed on the identical binary
(`persistence-v013-retry.json`). The cause of that intermittent transport
startup failure remains unresolved; the repeat does not establish its removal.
This is distinct from the Flash watchdog failure fixed and retested during OTA.

The ordinary Nyx HTTP proxy used above still forwards JSON/Base64. The separate
machine-ingress `proxy_upload_v1` receiver is bounded to 64 KiB JPEG pushes and
has host protocol/signature tests; its machine-ingress end-to-end path has not
been exercised against the deployed server. LAN JPEG and raw OTA carry bytes
directly. Do not infer binary machine-ingress acceptance from ordinary proxy
success.

The first v0.1.13 media run decoded H.264 and completed 12 Nyx status
calls, but its Camera snapshot request exceeded the 10-second client timeout
(`media-acceptance-v013/results.json`). It did not pass the full sequence;
the serial-backed retry (`media-acceptance-v013-retry/results.json`) later
stopped on one Nyx HTTP 504 after six successful status requests. The missing
request did not appear in the board log, and the board remained online with
continuous uptime; this does not identify whether the loss was in Wi-Fi,
the Internet path or the server. Neither attempt passed the full sequence.
The v0.1.14 strict run likewise
decoded H.264 and completed successful Nyx calls before a LAN status request
exceeded its 10-second client deadline (`media-acceptance-v014/results.json`).

Later LCD reads also timed out despite serial logging a successful response
send. A short LAN ping check had 20% loss and 987–1990 ms responses, while
the gateway had no loss and 5–11 ms responses. A deliberate station reconnect
restored Wi-Fi/Nyx without reboot; the next eight board pings had no loss and
7–180 ms responses. These short samples
(`network-diagnostics-v013.json`) show a transient link problem; they do not
establish its root cause or long-term removal. Native
WLAN logs on v0.1.14 show strong signal (RSSI −26 to −29 dBm), so these
measurements alone do not identify the cause as weak signal. The final
30-second-deadline diagnostic completes the media/preview/resume sequence
without a reboot, including all eight Nyx requests, but its worst Nyx request
takes 19.495 seconds. That result does not cancel the earlier strict failures.

The camera scene is extremely dark/covered. Physical camera QR scanning, image
quality, high-motion streaming, physical gestures/audio perception and a
multi-hour soak are not verified. H.264 has some packet loss; the short run does
not establish zero loss or production media stability. Ethernet, a secondary
Wi-Fi AP and optional real device-code endpoints were not physically exercised.
The native video player requires SD media, which was absent during acceptance.
