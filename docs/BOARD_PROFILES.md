# Board profiles

`boards/catalog.json` is the shared definition used by CMake, the build wrapper,
release packaging, image validation, runtime metadata and CI's build matrix.
A profile identifies both board wiring and supported silicon revisions. The
server environment and wake model are separate choices.
Current catalog validation and image headers target ESP32-P4 and the existing
16 MiB partition contract. Supporting another chip family or partition layout
requires extending those contracts and their tests; a catalog entry alone is
not sufficient.

## Build boundaries

Use ESP-IDF 5.5.4 and its exported Python environment:

```sh
python3 tools/build_profile.py --profile p4-rev1-16mb
python3 tools/build_profile.py --profile p4-rev3-16mb
python3 tools/build_profile.py --profile p4-rev3-lcd-x8-16mb
```

Each build uses `build/profiles/<profile>/production/sdkconfig`. A local build
uses a separate directory and CMake's existing environment-specific NVS profile:

```sh
python3 tools/build_profile.py --profile p4-rev3-lcd-x8-16mb \
  --environment local --api-url http://192.168.1.20:3001
```

The local login page defaults to port 3000 on the same host. Existing URL
validation still requires a private IPv4 LAN origin. No credentials are baked
into either build. Stable release packaging rejects development builds.

For a custom diagnostic build use `idf.py -B <unique-directory>
-D PANEL_BOARD_PROFILE=<profile> -D SDKCONFIG=<unique-sdkconfig> build`.
Do not use `SDKCONFIG_DEFAULTS` to select a board: the catalog supplies those
defaults. CMake validates the resolved configuration and permanently associates
each build directory with its profile. It rejects changing the profile in that
directory and detects incompatible saved sdkconfig values. Shared application
settings can still be changed with menuconfig; hardware changes must match the
selected profile. Bare `idf.py build` defaults to the legacy rev1 profile with
its configuration inside `build`, rather than reusing the project-root sdkconfig.

## Hardware implementations

The catalog selects a BSP dependency, adapter and product backend. Applications
receive the BSP interface through `product_support`; they do not select a
vendor component themselves. Current boards share
the pinned Waveshare platform component. `boards/platform/bsp.cmake` copies its
display/touch/audio BSP into the individual build directory and applies only
the selected board patch there. An already patched managed component from an
older build is normalized in the copy. Building one profile does not change
the managed BSP used as input to the other profile.

`boards/platform/product_backend.c` implements the external-screen brightness
readback and ES8311 capture/format setup. `boards/lcd_x8/product_backend.c`
implements PWM brightness readback and ES7210 TDM normalization/format setup.
Only the selected source is compiled. Shared `product_support` retains audio
ownership, playback and the existing signed microphone/reference interface.
The UI, Nyx services and AEC/FIFO pipeline do not inspect board model flags.

Generated `panel_board_config.h` supplies identity, display dimensions, the
stylesheet and camera geometry. Status reports the configured board/profile
and codecs. Camera support is marked optional; this does not assert that a
sensor is physically attached. Neither status nor a chip revision is hardware
wiring autodetection.

## Images, OTA and preserving USB updates

Images contain a fixed 112-byte board descriptor directly after IDF's standard
application descriptor, at image offset 288. It binds the profile and board ID.
Packaging also verifies the actual chip/minimum/maximum revision image header,
resolved Kconfig, application project/version and slot size. URL OTA and raw
image OTA check the same embedded identity; a release filename or a correct
SHA256 alone does not establish board compatibility. Raw OTA waits for the
complete board header before erasing/writing the inactive app slot.
An early rejection closes the upload session instead of draining an untrusted
body or allowing its remainder to be parsed as another request.

An image without the new descriptor is rejected by the new OTA implementation.
Current older installations can receive their first descriptor-bearing update
through their existing exact-profile release channel or USB. Downgrading to a
descriptor-less historical image requires explicit USB recovery. Partition
offsets, server namespaces and the release manifest schema are unchanged.

Use esptool 5 or later for a normal preserving USB update:

```sh
python3 tools/flash_profile.py --port /dev/cu.YOUR_PORT \
  --profile p4-rev3-lcd-x8-16mb --expected-mac aa:bb:cc:dd:ee:ff \
  --image build/profiles/p4-rev3-lcd-x8-16mb/production/nyxid-esp-panel.bin \
  --backup-dir .local/flash-backups/unique-run
```

The operator must identify the board wiring and intended P4 MAC first. The
tool checks the incoming image, connected MAC/revision, existing partition
map, valid OTA-selection CRC and installed image profile when available. It
backs up the current app and prefix/NVS, writes only the selected app partition,
and verifies the app plus unchanged prefix/storage hashes. It does not install
factory images or change partition offsets. A pending-verification OTA is
rejected until the boot health check completes. For an existing exact-device
app backup, `--rollback-image` verifies it against the current flash and avoids
reading the full slot. All backups remain private and outside version control.

Before the first migration, old installed images have no board descriptor;
the operator's profile/MAC selection is the hardware identification boundary.
Thereafter an incompatible installed profile is rejected as well. If a write
is interrupted, the private app backup and recorded offset provide the USB
recovery source; the tool does not silently change OTA metadata or device keys.
Backups use 115200 baud by default to accommodate the older USB serial bridge;
writes switch to 460800. `--read-baud` and `--baud` override those rates.

## Add a board

1. Add a physical board entry and its revision profile(s) to the catalog. Keep
   existing profile IDs stable; OTA uses exact identity. Add explicit hardware
   defaults, including disabling incompatible selections from shared defaults.
2. Supply its BSP adapter and product backend. A board with a different vendor
   BSP selects its dependency with `bsp_component` and adapter with `bsp_adapter`
   (declare the managed dependency in `main/idf_component.yml`); do not add model tests
   to shared UI or service handlers. Preserve the product audio contract and
   the one-reader capture/ownership rules.
3. Define its display geometry/stylesheet, codecs/capture layout and optional
   camera geometry truthfully. Extend the capability interface before claiming
   support for hardware absent from the existing contracts.
4. Run host and release tests, then `tools/build_profile.py` and release packaging.
   CI discovers the new profile from the catalog. Include old profiles in the
   regression run and exercise mismatched-image/configuration rejection.
5. Back up and test an identified development board: display/touch/brightness,
   native app transitions, microphone/reference routing, playback, audio
   ownership and camera if fitted. Verify persistence and recovery/OTA. Record
   source, image/ELF hashes, board/profile and observations. Compilation alone
   does not establish physical hardware support.

Existing acceptance progress and limitations are tracked in
[BOARD_ISOLATION.md](BOARD_ISOLATION.md).
