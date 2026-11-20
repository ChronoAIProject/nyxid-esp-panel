# Fitting the early marker within PCM16 headroom

The new 1.45-second WAV moves the three early marker segments together by 12 samples (0.75 ms earlier). At the new position the RTX 4090 search could use the full nominal amplitude of the early segments, up from a common factor of 0.7404. It used the original polarity. The speech carrier, later triplet and sample count match the control as specified by the source-only fit. The candidate WAV SHA256 is `a62dfff9d2e1beaf9dbb63f54ac88f33665082884ecf67f64150606a69f1a9c2`. [Candidate WAV](../results/hye-nyx-af-bella-headroom-experiment.wav) · [Registered source fit](bella_marker_headroom_protocol.md) · [66 evaluated settings and 47 native replays](../results/4090-bella-marker-headroom.json)

| Timely opens with opposite-speaker speech | Delivered Nyx | Candidate Nyx | Official Hey Willow |
| --- | ---: | ---: | ---: |
| 30° and 300°, 12 presentations | 10 | 11 | 7 |
| 120° and 240°, 12 presentations after candidate freeze | 10 | 10 | 6 |

The candidate recovered rear-speaker case 077 at 30°. Case 079 still missed in that direction. In the two additional orientations both Nyx files opened on the same ten presentations, with neither opening on front-speaker 077 or 079 at 240°. For rear-speaker 077, 079 and 090 at 120°, the candidate opened 416, 400 and 416 ms earlier respectively. Among the other seven successful trials, one moved forward 16 ms and six kept the previous read. All 12 erased and all reciprocal address controls stayed closed on both receivers in this second group. [Initial room results](bella_surrey_orientations_results.md) · [Additional-angle procedure](bella_headroom_new_angles_protocol.md) · [48 additional paired native comparisons](../results/4090-bella-headroom-new-angles.json) · [Native evidence](../results/4090-bella-headroom-new-angles.zip)

Both Nyx files reached 12/12 in the first quiet group and 15/15 in the five-speed weak group. With synchronous electrical reference the panel self-opened 0/4, and with a silent reference it opened 4/4. Across all 96 legacy interruptions the candidate and control opened before the command on 96/96. Twelve candidate opens moved forward 400 ms, two moved forward 16 ms, 81 had the same clock and one moved back 400 ms. The erased waveforms and receiver were byte-identical, so all 96 original negative traces were reused with verified input and log hashes. [47 primary UARTs](../results/4090-bella-marker-headroom.zip) · [96 historical positives and 96 matched erased controls](../results/4090-bella-headroom-history.json) · [Historical UARTs](../results/4090-bella-headroom-history.zip)

The Mac A/B pack contains both standalone WAVs, the three target commands for each address variant, the isolated interruptions, erased controls and a manifest. Use the same recovery-patched C161 app when swapping the two Nyx WAVs. All 21 ZIP members matched an anonymous download, and the candidate source tracks reproduced all 12 additional-angle digital microphones. Archive SHA256 `7f54eac1de62c2135152f04e2d64e635b3c259ce996bbd1901d5cfecc42cdfd0`. [Mac A/B playback pack](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-headroom-mac-ab-v1.zip) · [Member hashes](../results/4090-bella-headroom-mac-download.json) · [P4 firmware handoff](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7032582923)

## Received commands with competing speech

Each cell below uses all 12 presentations, including missed wakes. The decoder and two readers process the whole actual send at the first opening of each address. The immediate service was ready at the open. The delayed service became ready at 1,500 ms.

| Orientations | Address | Qwen / Whisper turbo immediate | Qwen / Whisper turbo delayed |
| --- | --- | --- | --- |
| 30°, 300° | Delivered Nyx | 10 / 9 | 10 / 9 |
| 30°, 300° | Candidate Nyx | 10 / 11 | 11 / 8 |
| 30°, 300° | Official Willow | 7 / 6 | 7 / 5 |
| 120°, 240° | Delivered Nyx | 10 / 8 | 10 / 9 |
| 120°, 240° | Candidate Nyx | 10 / 9 | 9 / 7 |
| 120°, 240° | Official Willow | 6 / 3 | 6 / 4 |

For the recovered 30° rear case 077, both readers returned the complete “turn down the hall light” suffix on the immediate send. Under the delayed schedule Qwen retained the complete suffix and Whisper transcribed “I'm turning down the hall light.” The four orientation groups reuse three commands and one listening room. A higher wake count or earlier opening did not consistently improve the received command. Retain the delivered WAV as the board comparison baseline, with the candidate as a paired test on the same firmware. The additional 37 successful calls produced 111 zero-watchdog capture, Opus and sender logs, all anonymously downloaded and verified. [Per-case receive words and clocks](../results/4090-bella-headroom-commands.json) · [111 UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-headroom-commands-uart.zip)

## When both loudspeakers play the identical address

Both WAVs gave the same open counts across the four simulated panel paths at each external source offset. At −800, −400, 0, +400 and +800 ms they gave 4/4, 2/4, 0/4, 4/4 and 4/4 openings. For each WAV only four first matches equaled the external-alone sample exactly, two at −800 ms and two at +800 ms. The other ten open cases had a mixture-specific clock. A positive offset alone cannot identify the triggering source when both speakers play the double marker. The earlier eight exact recoveries at positive offsets used a different external waveform. [Source controls and protocol](bella_headroom_same_cue_collision_protocol.md) · [40 paired clocks and reference traces](../results/4090-bella-headroom-same-cue-collisions.json) · [40 native UARTs](../results/4090-bella-headroom-same-cue-collisions.zip)
