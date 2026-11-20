# A 1.45-second generated Hye Nyx for Mac playback

The selected Bella WAV puts the multirate triplet first, followed by two nominal-rate triplets. It fits the original 23,200 PCM16 samples. The 4090 source fit accepted the full added amplitude at its intended sample position. Audio and protocol were frozen in `ee991300` before receiver scoring. All opens in the 59-case comparison came from the fixed marker in the C161 reference-rearm receiver. [Source WAV](../results/hye-nyx-bella-multirate-before-nominal.wav) · [Protocol](bella_third_triplet_protocol.md) · [Fit](../results/4090-bella-third-triplet-fit.json)

| Digital playback | Delivered two groups | Three groups, nominal first | Three groups, multirate first |
| --- | ---: | ---: | ---: |
| Original-level Surrey, opposing speech, four orientations | 20/24 | 24/24 | 24/24 |
| Surrey, quiet, first two orientations | 12/12 | 12/12 | 12/12 |
| Three weak scenes at nominal, ±100 ppm and ±1% | 15/15 | 14/15 | 15/15 |
| Panel playback with synchronized reference, opens | 0/4 | 0/4 | 0/4 |
| Panel playback with silent reference, opens | 4/4 | 4/4 | 4/4 |

At original level, 077 and 079 recovered both at 30° with a rear target and at 240° with a front target. All four opened 469–871 ms before the command. The matched official Hey Willow inputs opened on 13 of the 24 interfering-speech presentations. These are three reused commands. The head orientation and target side vary within one measured room. Both earlier two-group nominal-first trials recovered the same four misses, with nine and fourteen timely opens on the speed matrix. [First three-group order](../results/4090-bella-nominal-first-nominal-before-A.json) · [Selected order](../results/4090-bella-nominal-first-multirate-before-nominal.json) · [First Willow placements](../results/4090-bella-surrey-orientations.json) · [Additional Willow placements](../results/4090-bella-headroom-new-angles.json)

All 96 historical interruptions opened before the command and 92 opened earlier than the delivered two-group cue. The corresponding 96 erased inputs were byte-identical to their previous tests and their verified logs gave zero opens. Historical scene 081 had an interrupt watchdog reset in a simulator boot. Its retained retry passed on the same input and firmware. Different reruns of one scene measure simulator behavior under the host scheduler. [192 paired records](../results/4090-bella-third-triplet-multirate-before-nominal-history.json) · [Native evidence including the failed boot](../results/4090-bella-third-triplet-multirate-before-nominal-history.zip)

Both versions stayed closed in four of four scenes where the panel and the external speaker started the identical address together. With the three-group file starting externally 800 ms earlier, all four first events matched the external-only clock. When the external file started 400 or 800 ms later, all eight first events appeared only in the mixture. [Offset-by-offset clocks and source attribution](../results/4090-bella-third-triplet-collisions.json)

The Mac pack has two standalone mono WAVs, 12 stereo programs and a manifest. The original-level stereo programs reproduce all 24 candidate microphone hashes used above after applying the measured room responses and matched noise. Play the numbered files from a single stereo device, route left to the target speaker and right to the opposing speaker, and keep the same output level and placement across the four files. The anonymous release download matched the 2,094,854-byte archive and all 15 member hashes. Archive SHA256 is `fbe565b7af14b64e97b872a6bde58a52ff3d78600d84d2deb0cb0b68131ae790`. [Mac playback pack](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-three-triplet-mac-v1.zip) · [Manifest and SHA256s](../results/4090-bella-third-triplet-mac.json)

## What reached the simulated peer

Each of the 24 original-level interfering-speech presentations used its own open sample, captured pre-roll, Opus packets and first send. Both GPU readers evaluated the complete decoded stream. The table counts exact target-command suffixes and keeps all presentations in the denominator. Some passing transcripts include the opposing speaker before the target command.

| Address and receiver | Timely opens | Immediate Qwen | Immediate Whisper | Delayed Qwen | Delayed Whisper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Delivered two groups | 20/24 | 20/24 | 17/24 | 20/24 | 18/24 |
| Three groups, multirate first | 24/24 | 24/24 | 21/24 | 24/24 | 17/24 |
| Official Hey Willow | 13/24 | 13/24 | 9/24 | 13/24 | 9/24 |

For the four recovered opens, Qwen read the complete command 4/4 immediately and 4/4 with delayed readiness. Whisper read 2/4 immediately and 0/4 with the delay. For 30° rear 077, its immediate transcript ended in “turn down the whole line” for the target “turn down the hall light.” On the isolated generated wake sound, Qwen returned “Hey, Nick.” and Whisper returned “Hey Nix, it's”. [Three-group transcripts and receiver clocks](../results/4090-bella-third-triplet-commands.json) · [First baseline group](../results/4090-bella-surrey-orientations-commands.json) · [Second baseline group](../results/4090-bella-headroom-commands.json)

Five of the 24 calls opened at sample 31,488, before two seconds of input existed. The legacy sender test harness aborted on this valid partial pre-roll. The capture held samples from zero. We updated the harness input check to accept the actual available history, then rebuilt the sender fixture in a separate directory. The wake detector and the board microphone task use the delivered sources. On a full-history control the two fixture versions sent identical packets and clocks in both schedules (286 and 336 packets). The 24 candidate capture, codec and sender logs passed without watchdog warnings. The anonymous archive matched all 72 logs and the additional full-history control log. [Sender fixture source](../first_command_p4/mac_board_marker/sender.c) · [73 UART logs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-third-triplet-commands.zip)

## Two orientations selected after freezing the playback pack

The original-level pack was replayed at Surrey 60° and 210°, with the target on each side. The selected three-group WAV opened before the command in 12/12 trials, the delivered two-group baseline in 10/12 and official Hey Willow in 7/12 on its paired address. Both additional recoveries were case 077, at 60° rear and 210° front. The candidate opened 465–880 ms before command onset. Both receivers stayed closed on the 12 erased controls and on the other address. None of the 48 simulated microphones clipped. All 12 candidate opens used the fixed marker. The two orientations reuse the same room and three commands as the earlier 24 trials. [Registered plan](bella_third_triplet_new_angles_protocol.md) · [Per-file clocks](../results/4090-bella-third-triplet-new-angles.json) · [192 native evidence files](../results/4090-bella-third-triplet-new-angles.zip)
