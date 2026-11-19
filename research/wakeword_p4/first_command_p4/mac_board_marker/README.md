# Fixed Hye Nyx playback on the Mac board branch

The patch applies to `research/c161-parallel-runtime-20261118` at `0d8e1935702a561b535a8c1a0921fd8034943d2c`. That revision's single C161 idle loop opened on the model alone. The research switch feeds the same microphone frames to frozen C161 and the timed fixed-tone receiver. It feeds the synchronous electrical reference to the playback guards. The external target uses the distributed original-level af_bella WAV. C161 uses its original 60,952 bytes, threshold 0.91796875 and five smoothing steps. The product call uses its existing AGC/AEC and FIFO.

## Build and replay

In the board checkout, use its documented ESP-IDF v5.5.4 environment and build requirements. Set `research` to the GPU research checkout. Choose the profile recorded for the device. The previous Mac stability report identifies `p4-rev1-16mb` for the external-screen board and `p4-rev3-lcd-x8-16mb` for LCD-X8. The separate external-screen rev3 profile is `p4-rev3-16mb`.

```sh
handoff="$research/research/wakeword_p4/first_command_p4/mac_board_marker"
git apply --check "$handoff/c161-fixed-marker.patch"
git apply "$handoff/c161-fixed-marker.patch"
python3 "$handoff/check_replay.py" "$PWD" > replay.json
cmp replay.json "$research/research/wakeword_p4/results/4090-c161-mac-board-marker-replay.json"

profile=p4-rev1-16mb
build="$PWD/build/fixed-marker/$profile"
idf.py -B "$build" -D PANEL_BOARD_PROFILE="$profile" -D SDKCONFIG="$build/sdkconfig" \
    -D NYX_PANEL_WAKE_ENGINE=hey-nyx -D NYX_PANEL_RESEARCH_FIXED_MARKER=ON reconfigure
cmake --build "$build" --parallel 2
```

The marker switch defaults to OFF. Enabling it selects single C161 and disables update checks in this research build. Each listening epoch allocates 13,304 bytes of marker state in PSRAM. A wake releases that allocation before call startup. Firmware initialization primes C161 in full 480-sample blocks, rounding 81,440 requested silent samples to 81,600 processed samples.

## Clocks to collect over UART

- `panel.wake.marker begin` gives the listening epoch. The physical reference uses the second slot of normalized microphone/reference PCM. LCD-X8 takes these from slots 0 and 1 of four-slot input. This build has no synchronized TX event. It logs `tx=unavailable`.
- `decision` gives the one-past sample of each allowed route (`c161_end`, `cue_end`, zero for absent), the delivered read end and decision time in microseconds since boot. `start_call` reports local call setup. `min_internal_boot` is the allocator's minimum free internal 8-bit heap since boot at that observation. Guard decisions can follow raw detector hits.
- `first_pcm` gives the zero-based start of the first captured FIFO frame accepted by Opus, its length, and the time immediately before encoding. Its sample clock counts completed AGC output from the same epoch and includes preroll. Algorithmic delay from mic input to processed output must be measured from PCM.
- `peer_send` records the first successful return from the local peer send API and, if different, the first successful call carrying captured audio. `captured=0` means a synthetic silence frame with `start=-1`. `pts` is in milliseconds, `accepted_us` is on the boot clock. Use a peer/network capture for first arrival and decoded speech.

## Digital controls

The replay compiles the two C feeders from the patched board source and uses the actual X8 slot normalizer. It injects the original simulator UART C161 return at the exact end of each matching 480-sample block and verifies every PCM window supplied. For 16 captures from the same timed/middle firmware, supplying the archived TX reproduces both allowed routes and first 256-sample read end exactly. The report also retains eight older-firmware traces for comparison. The board feeder uses physical R only. With zero R, missing TX lets some self-play model hits through.

For geometry 0, replace only the zero R with the distributed panel signal at its original clock. Panel-only C161 hits sample 48,480 with either input. The restored reference suppresses its call. Simultaneous external cue and panel playback still open at sample 48,896, before command sample 49,600. The 24 original-level dry af_bella targets also open before the commands with model hits disabled. Erasing their address produces zero openings. All these numbers come from digital replay. The two boards supply the actual microphone/reference alignment, first captured frame, packet arrival, decoded first sentence, heap and runtime observations.

## Cross-build and P4 simulator evidence

Run `check_builds.py BUILD ...` in the IDF environment to inspect all three linked apps. It checks the selected board backend, feature flags, linked marker entry points, type size in the target ELF, both OTA partitions and the original model bytes inside the final flashable app. The image hashes identify the particular builds. The board project includes compilation timestamps.

The digital P4 fixture recompiles the board's C161 runtime and links its INT8 weights, marker and read functions with the existing P4 simulator support libraries. It starts a fresh detector for each distinct input, primes with 81,600 silent samples and reads interleaved M/R in blocks of 256. QEMU uses its test memory region and its floating-point context workaround. The fixture runs detection through the full clip to retain both routes. `O` records only the first read end that would request a call. The app-level network, codec and loudspeaker are exercised on the physical panel.

For an existing `middle-uart` build, run the following commands from the research checkout. The fixture build uses the original Ninja compile and link flags and writes a separate ELF and flash image. To prepare the simulator support build, use the existing `first_command_p4/mac_listener/build_panel_marker_order.sh middle-uart` research workflow.

```sh
fixture=research/wakeword_p4/first_command_p4/mac_board_marker
sim=.local/wakeword/c161-panel-middle-uart/build
qemu=.local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32
python3 "$fixture/build_guest.py" "$patched_board" "$sim" "$work"
python3 "$fixture/run_guest.py" "$work/board.bin" "$sim" "$qemu" "$work/run" > "$work/results.json"
```
