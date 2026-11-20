# Half an hour of indoor background followed by generated calls

All three receivers stayed closed throughout the six full DEMAND recordings, one channel each from a living room, kitchen, washing area, meeting, cafeteria and restaurant. Each then opened before the command on its own address in all six recordings and stayed closed on all six reciprocal addresses. Only the fixed marker triggered on the two Nyx receivers. Both Nyx listeners ran in P4 QEMU. Official Hey Willow DEFAULT ran in classic ESP32 QEMU with every native return replayed through the panel feeder. The command was “turn down the hall light.” [Registered protocol](fixed_rearm_demand_protocol.md) · [Per-input events](../results/4090-fixed-rearm-demand.json) · [Raw receiver evidence](../results/4090-fixed-rearm-demand.zip)

The six complete background segments total 28,800,384 samples or 30 minutes and 24 ms. Receiver repeats and the tail overlays count once. Each simulation used zero electrical reference and monitored continuously through both calls. A physical panel enters the call state after waking. The real listening intervals, call audio and minimum heap must come from the board. The issued Nyx playback was louder than the Willow address. The original test preserves both playback levels. An initial long-input transport attempt exceeded the default QEMU loader size before any PCM was scored. The archived failure records accompany the successful runs with 16 MiB of loader memory and a guest-verified input CRC for every recording.

The Mac playback ZIP contains six complete, single-channel WAVs, a readme and a manifest with original archive checksums and address/command boundaries. All eight members and the ZIP were anonymously downloaded and verified. The archive SHA256 is `cf12d3b051a322106e53fd082af1ae1ec8029c3986c8d7a0fdbae40af72311ee`. The audio license follows the authors’ CC BY-SA 3.0 Unported statement in the record description. [Playback files](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-willow-six-indoor-recordings-v1.zip) · [Download verification](../results/4090-fixed-rearm-demand-mac-download.json) · [DEMAND by Thiemann, Ito and Vincent](https://doi.org/10.5281/zenodo.1227121)

## Equal cue levels on the same six noise windows

The two addresses were measured over the same 1.45 seconds before the same spoken command. The original Nyx address had 8.64 dB more RMS level in this window. Both were scaled to the original Willow level for one comparison, and both were scaled to the level of the local noise for the second. All 24 addresses achieved their target within 0.01 dB, with zero clipped samples. Each short file starts a fresh listener against the known background tail. The short files contribute zero further background hours. [Level-matching protocol](fixed_rearm_demand_levels_protocol.md) · [Per-file levels, events and first-open clocks](../results/4090-fixed-rearm-demand-levels.json) · [150 evidence files](../results/4090-fixed-rearm-demand-levels.zip)

| Timely response to the receiver's own address | Standalone Nyx | C161 plus marker | Official Hey Willow |
| --- | ---: | ---: | ---: |
| Both at the original Willow level | 6/6 | 6/6 | 6/6 |
| Both at 0 dB cue-to-background RMS | 6/6 | 6/6 | 3/6 |
| Erased address, openings | 0/6 | 0/6 | 0/6 |

At 0 dB Willow opened in the living room, kitchen and washing recordings. It stayed closed in the meeting, cafeteria and restaurant recordings. All Nyx openings were marker decisions, 880 ms before the command. Willow successful openings were 304 ms before the command. Neither receiver opened on the other address. The six comparisons use one Bella cue and one Willow cue in digitally mixed background. Physical speaker and board microphone responses remain to be measured.

The short-play ZIP contains 30 WAVs and a manifest with level and sample boundaries. Its anonymous download matched all 30 registered WAV hashes and the manifest. SHA256 is `867657cdcd6a04c4afafffde000c1edffe15b619a27491d33af46498036190b5`. [Download matched short clips](https://raw.githubusercontent.com/ChronoAIProject/nyxid-esp-panel/a61d078d/research/wakeword_p4/results/hye-nyx-willow-six-indoor-equal-levels.zip)

## First command at the receiving peer

The issued 0 dB trials attenuate the address and preserve the command at its original level. A second pair for each room applies the same player gain to the dry address and command. Both presentations have byte-identical microphones through the command boundary and thus the same timely opening. For these new suffixes we used that causal prefix to determine eligibility. The trial covers one command and one generated address per receiver, with the original recorded background. The Nyx source required 8.64 dB more attenuation than the Willow source to give the two addresses equal RMS. Applying each source's gain to the command produces different speech levels between receivers. [Procedure](fixed_rearm_demand_peer_protocol.md) · [Waveform matches, PCM identities and full ASR transcripts](../results/4090-fixed-rearm-demand-peer.json)

Each cell below counts the exact words “turn down the hall light” at the end of the full decoded transcript, out of all six scenes including closed calls. Qwen and Whisper turbo ran on the 4090. “Waited” uses the 1,500 ms service-ready schedule. The 30 timely calls passed the simulated capture, Opus and sender checks. All six erased files stayed closed.

| Playback | Timely | Qwen immediate | Qwen waited | Whisper immediate | Whisper waited |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nyx, 0 dB address, original command | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Nyx, 0 dB address, same gain on command | 6/6 | 5/6 | 5/6 | 2/6 | 4/6 |
| Willow, 0 dB address, original command | 3/6 | 3/6 | 3/6 | 3/6 | 3/6 |
| Willow, 0 dB address, same gain on command | 3/6 | 3/6 | 3/6 | 3/6 | 3/6 |

With both addresses at the original Willow level and the unattenuated command, the two readers recovered the exact suffix in 6/6 scenes for both receivers and both schedules. In the restaurant weak-command pair, the Nyx address opened 880 ms before the command. Reducing the command with the player took its active speech-to-background ratio from +32.13 to +0.68 dB. Both readers returned “turn down the hallway” on the raw microphone and the pre-Opus FIFO. On the immediate peer send Qwen returned “turn down the volume” and Whisper returned “turn down the hallway.” Qwen recovered the command at the raw mic, FIFO and both peer sends in the other five rooms. Whisper's overall counts at the mic and FIFO were 4/6 and 5/6. Across both schedules all three waveform windows matched in all six original-command Nyx sends. For the weak commands the three-window test matched in one immediate and two waited sends. Other window results were inconclusive. [Microphone and FIFO diagnostics](../results/4090-fixed-rearm-demand-peer-stages.json)

For a rev1 physical comparison, play the previously issued `PRESTO-zero-snr-nyx.wav` and the new `PRESTO-zero-snr-nyx-whole.wav` at the same volume and position. Record the board microphone, actual listening state, trigger sample and decoded peer sentence. The two external WAVs differ only from sample 44,800 onwards. The new playback ZIP and its 13 members were verified on anonymous download. SHA256 is `1e2d29fe138bba5f25b7534ab437e1ad2ecdbe9cde3114205672906b9670102e`. The 90 native UARTs were separately downloaded and verified. [Whole-speaker controls](https://raw.githubusercontent.com/ChronoAIProject/nyxid-esp-panel/8722c830/research/wakeword_p4/results/hye-nyx-willow-indoor-whole-speaker-v1.zip) · [Original command controls](https://raw.githubusercontent.com/ChronoAIProject/nyxid-esp-panel/a61d078d/research/wakeword_p4/results/hye-nyx-willow-six-indoor-equal-levels.zip) · [90 UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-fixed-rearm-demand-peer.zip)

## Same command waveform for both receivers

At 0 dB address-to-background RMS, both receivers were given the exact same command and background samples from sample 44,800 onwards in each room, at each of two levels. The levels are 8.64 dB apart. The equal pre-command prefixes for each receiver allow the original timely-open decisions to be reused. Of twelve new combinations, nine had a timely open and went through capture, Opus and both sender schedules. Both ASR readers ran on the RTX 4090. Each count below includes all six rooms. [Registered crossover](fixed_rearm_demand_common_command.md) · [Transcripts, native input identities and denominators](../results/4090-demand-common-command.json)

| Common command level | Receiver | Timely | Qwen immediate / waited | Whisper turbo immediate / waited |
| --- | --- | ---: | ---: | ---: |
| Lower | Standalone Nyx | 6/6 | 5/6 · 5/6 | 2/6 · 4/6 |
| Lower | Official Willow | 3/6 | 3/6 · 3/6 | 3/6 · 3/6 |
| Higher | Standalone Nyx | 6/6 | 6/6 · 6/6 | 6/6 · 6/6 |
| Higher | Official Willow | 3/6 | 3/6 · 3/6 | 3/6 · 3/6 |

Among the three rooms where both opened, at the lower level Qwen recovered the complete command 3/3 for both receivers on each schedule. Whisper turbo recovered 2/3 for immediate Nyx and 3/3 for immediate Willow. Both reached 3/3 on the waited schedule. At the higher level both readers recovered 3/3 for both receivers on each schedule. The restaurant Nyx microphone has the same prefix and opens at the same read in both conditions. Increasing only the command level took active command SNR from +0.68 to +9.32 dB and restored “turn down the hall light” on all four reader/schedule combinations. These observations use one sentence and the same six previously measured backgrounds.

The archive contains the twelve new WAVs, 27 native UART logs and one manifest. The anonymous download and all 40 members matched their SHA256 records. Archive SHA256 is `207087871f5e14fc6889fc2ca6984cc35f6775c2d80ad3e9b0ad2f8267d5f235`. [Playback and raw evidence](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-demand-common-command.zip)

## Recorded speech and music without an address

The standalone rearm receiver, C161 plus marker, and official Hey Willow each completed 22 source streams with zero idle wake decisions. The fixtures and their positive and erased controls match the indoor 0 dB trial. Every eligible 30–300 s file from the two specified development manifests was played from start to finish. We added one five-minute excerpt around an earlier false trigger. Zero electrical reference and the original microphone level were used throughout. [Registered selection](fixed_rearm_speech_music.md) · [Source and runtime hashes](../results/4090-fixed-rearm-speech-music-plan.json) · [Per-file events](../results/4090-fixed-rearm-speech-music.json)

| Recorded input | Files | Original audio | Standalone Nyx | C161 plus marker | Willow DEFAULT |
| --- | ---: | ---: | ---: | ---: | ---: |
| VoxConverse speech | 14 | 36 min 29.824 s | 0 | 0 | 0 |
| MUSAN western art music | 7 | 22 min 15.771 s | 0 | 0 | 0 |
| Earnings-22 conference call | 1 excerpt | 5 min | 0 | 0 | 0 |

The conference call includes the sentence about major tenants “across the portfolio.” An earlier wide-window marker fired at 540.0405 s of that file. All three current receivers stayed closed throughout seconds 390–690. The three receiver passes cover the same 61,209,526 samples. All 63 min 45.595 s were previously used in development and add zero independent background exposure. The archive contains 44 P4 UARTs and the official reference's 22 UARTs, 22 reports and 22 raw-return files. [110 raw evidence files](../results/4090-fixed-rearm-speech-music.zip) · [Earlier event and word timestamps](../results/4090-c161-final-marker-background-event.json)

## Alarms and bells

All three frozen listeners stayed closed on each of the 40 clock alarms and 40 church bells in ESC-50. Each category contains 36 Freesound sources. The 80 clips total 400 seconds and each starts with a freshly primed listener. This measures responses to selected short sounds. Long-running household false opens per hour still require continuous recordings. The plan fixes the source revision, the 80 input hashes and both native runtimes before replay. The P4 Nyx images and official Willow reference used the same mono 16 kHz PCM16 at original gain and zero electrical reference. The archive contains all 400 native logs, reports and return streams. [Input plan](../results/4090-esc50-bells-plan.json) · [Per-file outcomes](../results/4090-esc50-bells.json) · [Receiver evidence](../results/4090-esc50-bells.zip) · [ESC-50 and source credits](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)

## Mac C161 baseline and the Nicole follow-up

Mac reported 26 automatic playbacks on the rev1 external-screen board with its installed C161 image `20beaf25…`. Five of twelve connected synthetic calls opened, and the peer received the complete “please repeat exactly: purple elephant” on all five. At player gain 0.65, Marin, Echo and Sage opened on 2/2, 0/2 and 0/2 trials. At 0.85 each opened on 1/2. Eight ordinary commands or conversational clips, three name probes and one deliberate “Hey Mix” probe stayed closed. Two intervening wake-counter increments had no paired source. During approximately eight minutes the board had no observed reset. Sampled internal free heap went from 209,095 B to a sampled low of 138,551 B and returned to 209,095 B. Sampling had a gap. Physical command lead, sustained stability and same-board fixed-sound/Willow results remain open. [Mac board receipt, playback hashes and service-side observations](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7041028477)

The published original Melo EN-BR “Hey Nicole, are we still meeting outside the cinema?” matches the development WAV byte for byte on anonymous download. In a new isolated digital replay, C161 plus marker produced one model event at sample 12,960 (810 ms) and delivered it at read end 13,056 (816 ms). The fixed-sound receiver stayed closed on the same PCM. A physical replay can check this name at the board's actual input level. [Original WAV](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/melo-br-hey-nicole-cinema-original-095.wav) · [Both receiver clocks and WAV hashes](../results/4090-melo-br-nicole-original-p4.json) · [Both P4 logs](../results/4090-melo-br-nicole-original-p4.zip)
