# Fixed Hye Nyx playback on the Mac board branch

The patch applies to `research/c161-parallel-runtime-20261118` at `0d8e1935702a561b535a8c1a0921fd8034943d2c`. That revision's single C161 idle loop opened on the model alone. The research switch feeds the same microphone frames to frozen C161 and the timed fixed-tone receiver. It feeds the synchronous electrical reference to the playback guards. The external target uses the distributed original-level af_bella WAV. C161 uses its original 60,952 bytes, threshold 0.91796875 and five smoothing steps. The product call uses its existing AGC/AEC and FIFO.

## Build and replay

The existing research release provides both linked application images. Use the image matching the development board profile. The download record gives the byte count, app SHA256, linked ELF SHA256 and embedded C161 offset for each original build. [External display rev1 app](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-fixed-marker-p4-rev1-16mb-8bbf5b6b.bin) · [LCD-X8 rev3 app](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-fixed-marker-p4-rev3-lcd-x8-16mb-e6bde7d0.bin) · [Verified download record](../../results/4090-c161-mac-board-marker-app-downloads.json)

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

### External-screen rev1 with the existing development call profile

Mac built the external-screen diagnostic app from the last identity-matched rev1 configuration. Its app SHA256 starts with `dd8af8a0`, its descriptor ELF SHA256 starts with `f022d9c8`, and it uses service fingerprint `6f3d3f8368` for voice, agent and node in development. Mac reports 6,623,392 bytes, 257,888 bytes free per OTA slot, one C161 model and 13,304 bytes of marker state. These figures describe the linked app. The published rev1 Release app uses the public service environment. [Mac build receipt and full identities](../../results/mac-c161-rev1-dev-marker-build.json) · [Mac report](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7027599185)

To reproduce the last observed development service configuration, activate ESP-IDF v5.5.4 in the patched board checkout and set `historical_app` to the archived rev1 app with SHA256 starting with `97c394aa` and descriptor ELF starting with `974260b7`. Choose a new local build directory. The command builds the app and writes a shareable verification report.

```sh
build="$PWD/build/fixed-marker-rev1-dev"
python3 "$handoff/build_rev1_dev.py" "$PWD" "$build" --old-app "$historical_app" > rev1-dev.json
```

The tool matches the entire historical app and its descriptor against the Mac record. It recovers the API and login URLs from the two compiled service namespaces and verifies the recorded fingerprint. The node WebSocket, verification URL and both node namespaces identify the node environment. For historical reproduction it also recognizes the earlier muted-call app (`d4055558…`, descriptor `b427b14a…`, service fingerprint `a30fb6b4f6`). To use the actual retained build cache as Mac did, pass `--old-cache "$historical_build/CMakeCache.txt"`. The report includes that file's SHA256 and the effective service fingerprint.

The verifier compares the inherited service settings with the new cache, the compiler definitions and the linked app. It also checks the board revision, the patched sources, single C161 runtime, model bytes, descriptor ELF identity and space in both OTA slots. The JSON contains the ten-character service fingerprint and node mode. The build directory and app contain the local endpoints. To check a finished build again, repeat the command with `--verify-only`. [Example host verification with development node](../../results/4090-c161-rev1-example-dev-profile.json) · [Example cache inheritance with public node](../../results/4090-c161-rev1-example-public-node.json)

### Trial for externally played generated calls

The playback-only option opens a call when the receiver recognizes the timed marker in the generated Hye Nyx WAV. The original C161 still processes the microphone stream. Its speech detections alone can open the primary diagnostic app. Ordinary unmarked “Hey Nyx” and “Hey Nicks” each opened that receiver on 5 of 12 connected synthetic phrases, and opened the playback-only receiver on 0 of 12. All 12 marked calls opened before the command with either receiver. The ordinary six-cups sentence opened the primary receiver on 6 of 36 placements and the playback-only receiver on 0 of 36. [Original-level playback WAV](../../results/hye-nyx-af-bella-marker-rms-v1.wav) · [Per-input decisions and protocol](../../results/4090-c161-fixed-playback-only.json) · [65 P4 simulator UARTs](../../results/4090-c161-fixed-playback-only.zip)

For a separate rev1 playback-only build, start in a board checkout with the base fixed-marker patch applied. The second patch changes the call decision and adds `policy=fixed-playback-only` to the UART listening-epoch message. The verifier checks the selected source and linked policy tag. Use the retained Mac development cache to reproduce the service configuration.

```sh
git apply "$handoff/fixed-playback-only.patch"
python3 "$handoff/build_rev1_dev.py" "$PWD" "$PWD/build/fixed-playback-only-rev1" \
    --old-cache "$historical_build/CMakeCache.txt" --fixed-playback-only > playback-only-rev1.json
```

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

## First commands with the panel speaker active

The original-level af_bella study reuses 24 rear-facing development room mixes with music and panel speech. The digital reference control places the panel's own WAV at its known playout time on R. The fixed marker opened all 24 clips 31–46 ms before their commands, with zero openings on the 24 address-erased controls. Both the board feeder and the listener capture fixture returned the same allowed routes and first read ends. [Board feeder and paired controls](../../results/4090-c161-bella-board-reference.json)

The table counts complete command suffixes after the entire two-second preroll and P4 Opus stream were decoded. All four rows use the same microphone and wake clocks, with eight examples at each of three external levels.

| Audio presented to Opus, with the synthetic panel reference present | Qwen | Whisper turbo |
|---|---:|---:|
| Board policy. AEC receives R from the first read after wake, followed by AGC | 12/24 | 9/24 |
| AEC receives R throughout listening and the call, followed by AGC | 10/24 | 5/24 |
| Reference activity selects continuous AEC/AGC or delayed microphone | 13/24 | 6/24 |
| Microphone delayed by 90 ms | 14/24 | 11/24 |

With a zero reference, the board-policy scores were 9/24 and 5/24. The earlier 14/24 and 9/24 scores came from delayed microphone PCM. Restoring R can move the wake time, so comparisons between the two reference conditions include the corresponding change in preroll. Each lane uses a fresh encoder. The Mac measures electrical R, connection fill packets, packet arrival and first-sentence quality on the physical panel. [Zero-reference comparison](../../results/4090-c161-bella-product-uplink.json) · [Restored-reference clocks and transcriptions](../../results/4090-c161-bella-reference-uplink.json)

The retained capture archive contains 48 UART logs. Its SHA256 is `25b231cad37dd780e395467d5283974698f4e8a9c33a421fd8f5032fbcc4fe21`. The separate 48 codec logs contain 96 encoded lanes. One codec boot recorded two idle watchdog warnings during its second lane and completed all packets. [Capture UART archive](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-bella-reference-uplink-uart.zip) · [Opus UART archive](../../results/4090-c161-bella-reference-uplink.zip)

## Matched Hey Willow playback

Use the same relative filename from both dry playback packs, with the same player, gain, position and panel playback. The two fixed af_bella addresses have matched active RMS. The command begins at the same sample in each pair and the PCM from that sample onward is identical. Keep each board's recorded M/R and the original trial clock. The official `wn9_heywillow_tts` DEFAULT reference consumes the raw microphone channel M. Its native return and the Nyx board decision can be scored independently. [Hye Nyx pack](../../results/c161-bella-command-playback.zip) · [Hey Willow pack](../../results/c161-bella-willow-command-playback.zip)

On the 24 digital rear-room inputs with music and panel playback, the Nyx marker requested the microphone before all 24 commands and Willow before one. Each detector had zero openings on the 24 address-erased controls. Each call kept its own request clock, two-second history and full P4 Opus stream. Qwen and Whisper turbo recovered 12 and 9 complete Nyx commands and 1 and 0 Willow commands. The isolated Willow positive opened 282 ms before the end of the address. [Native returns and first-command scores](../../results/4090-c161-bella-room-willow.json)

With the Nyx request clocks fixed, muting only the external music raised the complete-command counts from 12 to 21 for Qwen and from 9 to 18 for turbo. Muting only the panel yielded 13 and 8. With both sources silent the counts were 22 and 21. Keeping both sources and excluding the first two decoded seconds from ASR yielded 11 and 8. Mac can alternate music on and off with the panel playing and the speaker positions fixed. On `first/040.wav` (“turn down the hall light”) and `second/001.wav` (“please turn on the porch lights”), both readers recovered the full command only after muting music in that pair. [Paired source mutes](../../results/4090-c161-bella-product-interference.json) · [Receiver history comparison](../../results/4090-c161-bella-product-history.json)

## First captured packet during setup

The sender fixture compiles the wake branch of the patched board's `mic_task` and its FIFO into the existing P4 codec simulation. It takes the archived processed PCM at the actual open of each of the 24 rear-room Nyx cases. It runs an immediate-ready control and a simulated 500 ms until connection plus another 1,000 ms until provider readiness. The logical clock starts the microphone task at open. Real boards perform session setup before launching it. The fixture accepts each local send and checks the PCM supplied to every encoder call.

```sh
p4=research/wakeword_p4/first_command_p4
work=.local/wakeword/c161-bella-sender
python3 "$p4/mac_board_marker/build_sender.py" .local/wakeword/c161-mac-board-marker/src \
    .local/wakeword/c161-first-command-v5/build "$work/build" \
    --idf-python .local/wakeword/idf-tools/python_env/idf5.5_py3.11_env/bin/python
source .local/wakeword/c161-onset-names/env
TOKENIZERS_PARALLELISM=false taskset -c 2,4 nice -n 10 /usr/bin/python3 "$p4/mac_listener/check_bella_sender.py" score
```

All 7,396 immediate-ready captured packets matched the archived bytes. In the wait condition the board branch encoded 25 unsent silence frames, sent 50 silence frames, then sent the same 7,396 captured PCM frames across the 24 calls. Qwen and Whisper turbo recovered 12/24 and 9/24 full commands in the immediate condition, and 11/24 and 6/24 with the wait. For physical runs correlate both board `peer_send` markers with the first packet at the receiver and its first decoded command. [Paired transcripts and packet counts](../../results/4090-c161-bella-sender.json) · [Original packet UARTs](../../results/4090-c161-bella-sender.zip)

The three weak-level playback triplets include speech overlapping the first 0.40–0.62 s of the command. For a physical waveform comparison, use the clean command from the original playback pack as the reference. Map each triplet to the pack's relative file and pass the listed 16 kHz command boundary to the reader. It accepts 16- or 24-bit PCM WAVs and projects the selected receiver channel onto a 16 kHz clock. The 24-bit triplet itself can also serve as a reference when locating the combined speaker output. [Weak playback triplets](../../results/c161-bella-daily-missed-playback.zip) · [Clean commands](../../results/c161-bella-command-playback.zip) · [Capture reader](../mac_listener/measure_command_capture.py)

| Triplet | Clean playback file | Command start in samples | Command |
|---|---|---:|---|
| 077 | first/025.wav | 44800 | turn down the hall light |
| 079 | first/035.wav | 48400 | could you check the time for me |
| 090 | second/007.wav | 46000 | please turn on the porch lights |

For the speaker-gain comparison on these three cases, the external playback pack supplies the original mix, twice the amplitude on the intended speaker, and twice the amplitude on both the intended speaker and the interruption. Each mix includes Nyx, matched Willow and address-erased versions. Start the external file, timed panel file and music bed at sample zero. The 27 external files all use one-quarter amplitude. Keep the physical gain fixed within each comparison. The music target is −39.97 dBFS RMS over the whole music-only microphone recording for each scene. The room pack also contains the sample-exact simulated microphone and electrical reference as stereo PCM16. Its manifest maps every file to the external track and provides the panel onset and microphone hashes. [External speaker files](../../results/c161-bella-weak-speaker-gain-playback.zip) · [Room audio and digital inputs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-bella-weak-speaker-gain-room-audio.zip?sha256=a9bbe3137b0eefc6279dac0e77991e4b8e49f82c8ad9ac9d5f5189fdc3afe153)

Across the 32 weak-level interruptions, doubling the intended speaker moved timely Nyx opens from 29 to 32 and immediate full-command recognition from 5 to 10 (Qwen) and 3 to 10 (Whisper turbo). With both external voices doubled there were 31 timely opens and one address-erased false open. That false input combined an unaddressed command, “Hey Nick, can you help me?” and music. Muting any of the three eliminated the open. The music in the room pack and the false-wake microphone example is an excerpt of the Advent Chamber Orchestra's *Selections from the 2005-2006 Season*, MUSAN track `music-fma-wa-0004`, rescaled and mixed under CC BY-SA 3.0 US. [Word and timing results](../../results/4090-c161-bella-weak-speaker-gain.json) · [Source-mute replay](../../results/4090-c161-bella-gain-false-wake.json) · [Album](http://freemusicarchive.org/music/Advent_Chamber_Orchestra/Selections_from_the_2005-2006_Season/) · [Music license](https://creativecommons.org/licenses/by-sa/3.0/us/)

For example, with the clean Nyx pack extracted and the Opus receive stream exported as a WAV, run the following from the research checkout.

```sh
python3 research/wakeword_p4/first_command_p4/mac_listener/measure_command_capture.py \
    first/025.wav uplink-decoded.wav --command-at 44800
```

In P4 replay, retaining the panel reference and the enhanced full-mixture open, both recognizers missed all three full commands with music playing, even after removing the competing voice. With music off and the competing voice present, Qwen recovered 2/3 with immediate readiness and 3/3 with delayed readiness. Whisper turbo recovered 0/3 in both cases. Both sources muted gave 2/3 and 3/3 for the immediately ready readers, and 2/3 and 1/3 for the delayed readers. These counts use exact normalized command suffixes on the three selected development cases. [Source mutes, waveform matches and transcripts](../../results/4090-c161-bella-weak-interference.json)

For a fourth playback on the same weak trials, the code-boosted WAV adds 5.68 dB to the three known code components. The rest of the address and command keeps its original gain. In the 96 synthetic interruptions this cue opened before all commands. Qwen and Whisper turbo recovered 38 and 21 full commands with immediate readiness and 35 and 17 with the simulated wait. The original-level cue scored 34 and 23, then 42 and 16. Neither reader recovered a complete command on the three additional weak openings. Both readers transcribed the isolated candidate as “Hey, Nick.” The original-level WAV remains the primary board playback, and the code-boosted files provide a paired weak-level comparison. Each file in the supplement shares the half-gain PCM24 representation and the complete post-address waveform of its corresponding triplet. [Candidate WAV](../../results/hye-nyx-af-bella-marker-only-v1.wav) · [Three matched playback files](../../results/c161-bella-marker-only-weak-playback.zip) · [Per-call results](../../results/4090-c161-bella-marker-only.json) · [Isolated-address transcripts](../../results/4090-c161-bella-marker-level-readings.json)

The P4 simulation also played each isolated address through the panel impulse response from four rooms. Each version opened in all four when the reference channel was silent, and in zero with the synchronous reference. In the bedroom path, three microphone samples saturated on each boosted version. The original-level bedroom microphone reproduced the archived PCM. [Panel self-play records](../../results/4090-c161-bella-marker-only-self.json) · [Capture UARTs from the 96 external calls](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-bella-marker-only-capture.zip?sha256=948057040dcb9ca59d85247d4b8f86c496afeb999d17cdc22d495123c9e418f4)

## Bella final consonant and first-command comparison

The isolated Bella voice from the source synthesis gave “Hey, Nick's.” in Qwen and “Hey Nix.” in Whisper turbo. With the original three markers the readings were “Hey, Nick.” and “Hey Nicks, it's”. Twelve dry microphone replays tested the original speech stem and its marked versions with individual changes to the generated /s/ or /k/ interval. Single C161 opened the unmarked address. On both marked versions, all three speech edits preserved the exact marker opening. [Dry readings and detector clocks](../../results/4090-c161-bella-final-s.json)

A paired WAV advances the voice by 200 ms into leading digital silence and tapers it at the end of the generated /s/ interval. The three markers keep their original timing and level. Whisper turbo read “Hey, Nicks.” and Qwen read “Hey, Nick.” All 96 daily mixtures had the same first-open sample as their original counterparts, including the same three missed calls. The 93 timely calls opened 30–46 ms before the command. [Paired WAV](../../results/hye-nyx-af-bella-separated-v1.wav) · [Matched room and dry replays](../../results/4090-c161-bella-speech-clock.json)

For the first rotation over all 24 playback files, with eight per external level, Qwen and turbo recovered 9 and 6 complete commands with immediate service readiness on either cue. With the simulated connection wait they recovered 12 and 6 on the original cue and 10 and 7 on the revised cue. Keep the original-level cue for primary board playback. [Full stream readings](../../results/4090-c161-bella-speech-clock-uplink.json) · [72 original UART files](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-bella-speech-clock-uplink.zip?sha256=ad43bfebf3f325d6b935179f2897fc7f5c4c59b674c382dc9f5986dafa097056)

## Desktop alerts and the six-cups false wake

The freedesktop 0.8 theme supplied 27 unique source files (35 event names, 35.23 seconds). With each of these external sounds, the original-level Bella Nyx and matched Bella Willow addresses both opened before the command on all 27 trials. Neither listener opened on an alert alone or an alert with the command alone. One ordinary speech mixture opened the C161 model route. Official Willow had zero openings on those 27 mixtures. The plan identifies the released sound package, source files, speech inputs, clocks and shared command. [Matched input plan](../../results/4090-c161-desktop-alerts-plan.json) · [Per-detector events and transcripts](../../results/4090-c161-desktop-alerts.json)

“We need six cups of water” opened the model 248 ms before the service-login alert began, measured at the board's first allowed read. The isolated negative WAV preserves the complete microphone input up to the alert start. For a digital injection, after the normal C161 warmup, play the complete file starting at sample zero of the read stream. Expect the model at sample 37,440 and the first open at 37,632. Of six speech offsets spaced 5 ms apart, the 10, 15 and 20 ms offsets opened. Adding 52 complete 30 ms processing periods of silence gave the same pattern. Two other renditions of the sentence each had zero openings in twelve trials. For a physical playback record the actual microphone/reference and listening epoch for comparison. [PCM16 negative WAV](../../results/c161-six-cups-false-wake.wav) · [36 paired clock checks](../../results/4090-c161-six-cups-clock.json)

With each detector's own opening and the complete decoded send stream, Qwen recovered “turn down the hall light” on all 27 alert trials for either address. Whisper turbo recovered it on 26 for each. Both immediate and delayed service readiness gave these counts. On the rear-left channel announcement, turbo transcribed the command onset as “return down” with either address. The 56 simulated captures and sends completed with zero watchdog warnings. Ten standalone Opus exports logged 13 warnings in total and completed their packet checks. [Individual first-command records](../../results/4090-c161-desktop-first-command.json) · [168 original UART files](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-desktop-first-command.zip?sha256=673b61aaf7de84f6a4d8343f7ec2b31902748e622a16542085b9d66fdbd1004a)

For case 072, use one external speaker for both voices, a second speaker for the music, and the panel's own playback and reference. Start the tracks at sample zero and hold the gains fixed across the eight source-mute combinations. The full recording window is 106,832 samples at 16 kHz. At the microphone the solo full-window RMS targets are −34.12 dBFS for the mixed voices, −40.39 dBFS for the music and −45.10 dBFS for the panel. The manifest also provides targets for the individual voice tracks and maps each combination to its exact stereo digital input. [Case 072 playback pack](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-bella-072-room-playback.zip?sha256=e1d8685bd0d90f8f969320093dfbd7ab833e1bb6f03c161194e86ecd811435e0) · [Calibration and file hashes](../../results/4090-c161-bella-072-room-playback.json)

## Official Hey Willow on the external-screen rev1 panel

The public reference app runs the official ESP-SR 2.6.1 `wn9_heywillow_tts` on the raw 16 kHz microphone. It uses `DET_MODE_95` and the model's default threshold. The address detector consumes 81,600 leading zeros continuously with the first live read. A return of 1 opens a call. Both the Nyx and Willow builds keep the same I2S reader, AEC, AGC, preroll, FIFO and Opus sender. The Willow build releases the detector at call startup and recreates it for the next listening epoch. [Image and model hashes](../../results/4090-hey-willow-p4-rev1-reference.json) · [Public reference archive](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hey-willow-p4-rev1-public-reference-v1.zip)

For the physical first-command comparison with Mac's C161 `dd8af8a0…`, build Willow with the same local service settings. Start in the research checkout under ESP-IDF v5.5.4. Set `primary_app` to the absolute path of the retained C161 `.bin`. The helper checks the known app and ELF identity, reads its service profile locally, and verifies the resulting image against the same fingerprint `6f3d3f8368`. A retained C161 build cache also works with `--old-cache /path/to/CMakeCache.txt`. Each OTA slot has 20,640 bytes of headroom in the public build. The tool reports the actual headroom for the local build.

```sh
research=$(pwd -P)
fixture="$research/research/wakeword_p4/first_command_p4/mac_board_marker"
trial=$(mktemp -d)
board="$trial/board"
git worktree add --detach "$board" 0d8e1935702a561b535a8c1a0921fd8034943d2c
git -C "$board" apply --index "$fixture/c161-fixed-marker.patch"
git -C "$board" apply --index "$fixture/hey-willow-reference.patch"
python3 "$fixture/build_willow_rev1.py" "$board" "$trial/build" \
    --primary-app "$primary_app" > "$trial/willow-identity.json"
```

Use the 10.1-inch rev1 panel for both images. Play the original Bella Nyx and matched Willow at the same speaker position and level. Include the address-erased versions of 077, 079 and 090 and hold the panel playback and music fixed. Record both runtime image identities, UART listening epoch, detection sample and read end, call startup result, first PCM, first accepted peer packet, minimum internal heap, resets, physical M/R and the first sentence at the receiver. `peer_send` timestamps local acceptance. The receiver recording supplies the audio actually delivered. [Paired files and levels](../../results/c161-bella-weak-speaker-gain-playback.zip) · [Same-room playback tracks](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-bella-weak-speaker-gain-room-audio.zip)

In the 12 existing Mac synthetic fixtures the official classic ESP32 reference returned a wake on 4/4 Willow and 0/8 other addresses. Replaying all 3,468 archived returns through the host-compiled panel code reproduced every model boundary. Four channel-verification returns of −1 passed through without requesting a call. Both P4 QEMU attempts selected the official WakeNet9 interface and stopped during model creation at `dl_esp32p4_sr_memcpy` with illegal instruction `0x03a1801f`. The archive includes both UARTs. Physical P4 recognition, heap during calls, and delivered first sentences require board measurements. [Per-clip evidence](../../results/4090-hey-willow-p4-rev1-reference.json)

Start with quiet case 077. Its three files have the identical command starting at sample 44,800. With zero playback reference and the frozen digital inputs, Nyx opened at 43,520 through the fixed marker and Willow at 39,936 on its own address. Both rejected the other address and the address-erased command. Keep the same speaker gain while switching files. The physical request clocks can be compared against these digital references after aligning the microphone recording. [Three quiet playback WAVs](../../results/c161-willow-077-quiet-pair.zip) · [Quiet clocks](../../results/4090-hey-willow-p4-rev1-quiet.json)

For the weak room triplets, the Nyx receiver opened on 0/3 original addresses, 3/3 with the intended speaker doubled, and 2/3 with both external voices doubled. Official Willow gave zero wakes on its nine matched weak addresses. On case 072 the Nyx model opened when unaddressed speech, interruption and music coincided, and Willow gave zero wakes in all eight source combinations. These are selected digital examples. Twenty-five Willow return sequences were reused by exact microphone hash and ten were newly measured in classic ESP32 QEMU. [35 paired clocks](../../results/4090-hey-willow-p4-rev1-room.json) · [Original UART and return sequences](../../results/4090-hey-willow-p4-rev1-room.zip)
