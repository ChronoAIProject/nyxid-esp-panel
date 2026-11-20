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
