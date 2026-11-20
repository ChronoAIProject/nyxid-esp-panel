# Recovering real external calls after panel self-play

On the same 400 ms two-marker WAV, the combined receiver variant eliminated the two panel self-opens in the P4 simulation and kept the first-open clocks of all 96 external historical interruptions and all 15 weak player-rate cases. All 96 erased inputs stayed closed. Silent-reference playback still opened on all four panel paths. Frozen C161 supplied zero accepted events in these tests. The marker state occupied 13,304 bytes in each guest epoch. Simulator heap readings are recorded separately from minimum internal heap on a physical board. [215 primary receiver trials and original clocks](../results/4090-bella-reference-barge-historical.json) · [Native UARTs](../results/4090-bella-reference-barge-historical.zip) · [400 ms WAV](../results/hye-nyx-af-bella-early-marker-experiment.wav)

The original complete reference matcher recognized the first panel code at sample 47,512. The veto ended at 51,001. After a match this detector skipped 8,039 subsequent 4 kHz ticks. In rooms 1 and 3 the raw microphone matched the late code at 54,268 and 54,020. The firmware requested a call at the end of the corresponding 256-sample reads. An 80 ms recovery on the electrical reference registered the second code at 53,908 and held the veto through sample 57,396. Both late microphone matches then stayed suppressed. The microphones, reference signals, C161 model bytes and detection thresholds were identical in the comparisons. [Reference-only P4 results](../results/4090-bella-reference-rearm-historical.json) · [Reference and microphone clocks](../results/4090-bella-reference-rearm-collisions.json)

The second change lets the microphone begin reacquiring after 80 ms when its raw hit is immediately vetoed by confirmed playback. For each of four room paths we played the 400 ms panel cue and the external original +5 dB cue through the same room transfer and gain, with only the panel in the electrical reference. External-alone inputs produced a raw match in all 20 comparisons. At +400 and +800 ms all eight recovered raw match samples equaled the external-alone clock exactly. The old receiver opened on late panel echoes in rooms 1 and 3 before the subsequent external marker arrived. [Collision design](bella_reference_rearm_protocol.md) · [Microphone recovery procedure](bella_reference_barge_protocol.md) · [20 paired native observations](../results/4090-bella-reference-barge-collisions.json) · [Original and reference-only UARTs](../results/4090-bella-reference-rearm-collisions.zip) · [Combined receiver UARTs](../results/4090-bella-reference-barge-collisions.zip)

| External source start relative to panel | Old receiver opens / 4 | Reference change only | Both changes | Source of combined receiver's first open |
| --- | ---: | ---: | ---: | --- |
| −800 ms | 4 | 4 | 4 | External marker |
| −400 ms | 0 | 0 | 0 | Veto active |
| Simultaneous | 2 | 0 | 0 | Veto active |
| +400 ms | 2 | 0 | 4 | External marker |
| +800 ms | 2 | 0 | 4 | External marker |

At −400 ms and at simultaneous starts the external template lands near one of the panel sequences and the receiver stays closed. Both sources use one monaural path in this construction. The two suppressed overlaps are measurable limitations for a user speaking during a panel wake demonstration. The eight recoveries at positive offsets concern wake requests. These collision inputs end with the calls and contain no spoken command.

## First command after an accepted wake

On the RTX 4090, a 16 kHz WHAMR SepFormer processed the decoded Opus streams from 15 weak-room presentations, eight quiet two-sentence presentations and one unaddressed opening, each under two service schedules. Both Qwen3-ASR and Whisper turbo scored 0/15 complete weak commands on each schedule, including when either separated track could supply the answer. On the eight quiet immediate sends the original Qwen and Whisper results were 6/8 and 8/8. Selecting one separated track by the preregistered energy ratio yielded 1/8 and 2/8. For waited sends the same comparison was 6/8 and 8/8 before separation and 2/8 and 1/8 after selection. Both readers still recovered the full unaddressed command in case 072 on both schedules. The board images use the original uplink. [Procedure and conversion identity](bella_sepformer_commands_protocol.md) · [CUDA operators and transcripts from both tracks](../results/4090-bella-sepformer-commands.json)

Across the same 15 weak presentations, the first-command results changed before Opus encoding. Every stage used the measured P4 FIFO start and stop for that presentation. The player views used the published quarter-scale amplitude. The five speeds in each scene repeat one target and one interfering sentence. [Matched clocks and scoring procedure](bella_early_command_stages_protocol.md) · [Per-stage words and PCM hashes](../results/4090-bella-early-command-stages.json)

| Signal at the measured capture window | Qwen complete / 15 | Whisper turbo complete / 15 |
| --- | ---: | ---: |
| Intended external speaker alone | 15 | 15 |
| External player including competing speech | 15 | 11 |
| Combined room microphone | 8 | 2 |
| AEC/AGC FIFO before Opus | 0 | 0 |
| Immediate receive decode | 0 | 0 |

In room 077 both readers already missed the complete command on the raw microphone at all five speeds. In 090 Qwen read the complete microphone command on all five, then missed all five in the FIFO. In 079 Qwen lost three microphone successes at the FIFO and Whisper lost two. Whisper also recovered one command on the raw microphone that it had missed on the player mix. These stage comparisons locate changes in recognizer output. The microphone-to-FIFO step includes the capture processing and its latency.

## Board handoff

Both public-service applications link against ESP-IDF v5.5.4. The rev1 app has 257,360 bytes free per OTA slot, and the integrated LCD-X8 app has 230,560. Each ELF reports 13,304 bytes of marker state. The downloaded application descriptors match their build ELFs. Mac can reproduce the rev1 receiver using its own development-service cache for first-command tests. [Board files and archive identity](../results/4090-bella-reference-rearm-app-download.json) · [Mac build instructions](../first_command_p4/mac_board_marker/README.md#two-marker-400-ms-playback-trial)

## ESP echo suppression level and the isolated playback

The ESP VOIP_HIGH_PERF capture produced byte-identical AEC/AGC FIFOs with the normal and aggressive nonlinear suppression settings on all 15 weak commands. The three nominal-clock panel-only controls also matched sample for sample. The raw post-open echo levels were −53.96, −47.16 and −54.28 dBFS. After the AEC/AGC chain both settings yielded −44.55, −41.47 and −44.80 dBFS respectively. The 15 identical speech FIFOs reused their verified Opus and sender outcomes. All 21 new capture UARTs were anonymously downloaded and checked against the individual hashes. [Registered settings](bella_early_aec_level_protocol.md) · [Per-input hashes and levels](../results/4090-bella-early-aec-level.json) · [21 capture UARTs on the research Release](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-early-aec-level.zip)

For the 1.45-second delivered cue played alone, unprompted Qwen returned “Hey, Nick.” and Whisper turbo returned “Hey Nix, it's”. On the original +5 dB control they returned “Hey, Nick.” and “Hey Nicks, it's”. These are direct recognizer readings of the PCM with no command attached. [Both original transcripts and waveform hashes](../results/4090-bella-release-cue-transcripts.json)

## Command delayed past the competing speaker

Delaying only the target command by 640 ms in the external source preserved all 15 first-open samples. The three dry interruptions had extended 398–618 ms into the original command. Both readers still recovered zero complete commands under immediate and waited service readiness at the five tested player speeds. The speech level, music, panel and room path were fixed. These results support keeping the original paired playback for the physical comparison. [Procedure](bella_delayed_command_protocol.md) · [Per-case clocks, PCM identities and actual receiver words](../results/4090-bella-delayed-command.json)

The first serial export recorded 53 task-watchdog warnings in the simulated Opus encoder with host affinity limited to one logical CPU. Replaying all 15 original FIFOs with affinity to two logical CPUs produced zero warnings. All 5,483 packets in each encoder lane matched the first exports byte for byte. The original capture and sender logs had zero watchdog warnings. Both evidence archives were downloaded anonymously and every member matched its hash. [61 original evidence files](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-delayed-command-uart.zip) · [Packet comparison](../results/4090-bella-delayed-command-codec-repeat.json) · [15 repeat exports](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-delayed-command-codec-repeat.zip)
