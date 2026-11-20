# External player level and the command received after wake

At Surrey 60° and 210°, lowering both stereo channels by 6 dB kept the three-group Hye Nyx timely on all 12 presentations. Lowering the target alone put its command and address 6 dB lower relative to the opposing speaker. All cases use three reused commands, two directions and a silent panel. The native receivers processed every microphone. Across the 96 lowered-level inputs there were zero clipped samples, zero reciprocal opens and zero erased-control opens. All Nyx opens used the fixed marker. [Procedure](bella_third_triplet_player_level_protocol.md) · [Original native results](../results/4090-bella-third-triplet-new-angles.json) · [96 reduced-level inputs and clocks](../results/4090-bella-third-triplet-player-level.json) · [384 evidence files](../results/4090-bella-third-triplet-player-level.zip)

| Target / opposing player gain | Two groups | Three groups | Official Hey Willow |
| --- | ---: | ---: | ---: |
| 0 / 0 dB | 10/12 | 12/12 | 7/12 |
| −6 / −6 dB | 10/12 | 12/12 | 6/12 |
| −6 / 0 dB | 8/12 | 11/12 | 1/12 |

Each simulated call used the complete captured FIFO, its own first-open clock, native Opus and the sender's immediate and delayed readiness schedules. The following counts require the complete target command at the end of the transcript. The denominator includes every intended address, including misses. Values are Qwen / Whisper turbo.

| Gain and address | Immediate, per 12 | Delayed, per 12 |
| --- | ---: | ---: |
| Original, two groups | 10 / 9 | 10 / 10 |
| Original, three groups | 12 / 11 | 12 / 10 |
| Original, Willow | 7 / 4 | 7 / 5 |
| Both −6 dB, two groups | 10 / 9 | 10 / 10 |
| Both −6 dB, three groups | 12 / 8 | 12 / 9 |
| Both −6 dB, Willow | 6 / 5 | 6 / 6 |
| Target −6 dB, two groups | 7 / 6 | 8 / 6 |
| Target −6 dB, three groups | 10 / 6 | 10 / 6 |
| Target −6 dB, Willow | 1 / 1 | 1 / 1 |

At target −6 dB the candidate missed 210° front 079. In 210° front 077 it opened, but aligning only the first 400 ms of the cue estimated the command 6,440 samples late. Even with the 100 ms guard Qwen lost “Turn” from “Turn down the hall light.” The complete original send retained the command in both schedules. A simulator capture boot for that same input required a retry. The archive includes the failed UART. The 29 original-level and 48 reduced-level completed calls passed all three native stages without watchdog warnings in the accepted logs. Both anonymous release downloads matched the archives and all 87 and 145 members. [Original commands](../results/4090-bella-third-triplet-player-level-original-commands.json) · [Reduced-level commands](../results/4090-bella-third-triplet-player-level-reduced-commands.json) · [Original UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-third-triplet-player-level-original-commands.zip) · [Reduced-level UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-third-triplet-player-level-reduced-commands.zip)

The 24 stereo WAVs in the Mac level pack already include the two gain choices. Keep the playback device and master at the original setting, route left to the target speaker and right to the opposing speaker, and compare the same numbered case across directories. After room replay the exports matched all 96 measured digital microphones by hash. The 6,572,533-byte archive and all 25 members passed anonymous download verification. SHA256 is `f73eff56040a8acf44f922b549b0eb8e36b784def268e6d6d34dfd864c389401`. [Mac level pack](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-three-triplet-player-level-v1.zip) · [Member hashes](../results/4090-bella-third-triplet-player-level-mac.json)
