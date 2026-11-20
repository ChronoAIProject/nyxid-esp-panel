# Recovering the 210° front weak-target 079 command

The 1.45-second Bella candidate raises the first two fixed-marker groups by 3 dB. Its 23,200 samples peak at +32,014 and −31,416 in signed PCM16. The file was frozen in commit `0945a2c2` before receiver scoring. The only change to playback is in 7,737 samples of the generated address. The receiver uses the frozen C161 model and the reference-rearm fixed-marker path. [Predeclared comparison](bella_early_groups_3db_protocol.md) · [Source identities and PCM hash](../results/4090-bella-early-groups-3db-fit.json)

At Surrey 210° front, the target plays 6 dB below the issued level, the opposing speaker plays at its original level and command 079 starts at microphone sample 49,009. In the middle triplet of the previously issued cue the second marker's most negative correlation was −0.275, short of the required magnitude of 0.30. Raising the first two groups gave −0.333 at sample 39,408. This paired with the first marker and completed a chain at sample 41,328. The native P4 simulation opened at 41,472, 471 ms before the command.

| 079 address | With the other speaker | Other speaker muted |
| --- | --- | --- |
| Issued three groups | Closed | Marker open at 35,072 |
| First two groups +3 dB | Marker open at 41,472 | Marker open at 35,072 |
| Issued cue's nine markers alone | Closed | Marker open at 35,072 |
| Bella speech alone | Closed | C161 model open at 47,616 |
| Erased address | Closed | Closed |

All ten tests retain the target command, microphone noise and source timing. The 4090 correlation analysis predicted the exact native marker sample or absence in all ten. The speech-only model open is one synthetic voice in one presentation. [Per-group correlations, clocks and hashes](../results/4090-bella-early-groups-3db-diagnosis.json)

After observing the isolated 079 wake, the command check was extended to all twelve original-timing weak-target cases to see whether moving the open changes the first sentence elsewhere. Use the same two readers, both readiness schedules and complete received audio as in the previous player-level comparison. Count the exact target at the end of each transcript and keep the original twelve presentations in the denominator. This extension was made before running either reader on the new cue.

## Interruption timing and first sentence

Across 60 presentations of the same three commands at Surrey 60° and 210°, both directions, the issued cue opened on time in 58 and the new cue in 60. The two gains were 210° front 077 with the opposing speaker advanced 160 ms and 210° front 079 at the original timing. The 60 erased controls stayed closed. Each of the 12 original-timing microphones and all original receiver clocks matched the prior measurement for the issued cue. All 120 addressed presentations used the fixed-marker path when they opened. Across the separate earlier 59 presentations, both cues opened on all 55 external or silent-reference trials, and both stayed closed on the four synchronous self-play trials. [Five timing shifts](../results/4090-bella-early-groups-3db-shifts.json) · [Paired counts against the issued cue](../results/4090-bella-early-groups-3db-comparison.json)

| Target alone −6 dB at original timing, per 12 | Issued three groups | First two groups +3 dB | Official Hey Willow |
| --- | ---: | ---: | ---: |
| Timely wake | 11 | 12 | 1 |
| Full command as final words, immediate Qwen / Whisper | 10 / 6 | 10 / 7 | 1 / 1 |
| Full command as final words, delayed Qwen / Whisper | 10 / 6 | 10 / 7 | 1 / 1 |

On the recovered 079, both readers on both schedules returned “Nick, please turn on the light. Can you check the time for me?” The target starts with “Could you”. The exact-command score therefore stayed zero on this presentation. In the other paired calls, Whisper recovered 210° front 090 immediately and 60° front 079 on delayed readiness. Qwen had the same exact-command outcomes as before. The Willow column reuses the official receiver on the original zero-offset Willow inputs. [New complete sends and transcripts](../results/4090-bella-early-groups-3db-commands.json) · [Prior matched reads](bella_third_triplet_player_level_results.md)

## Mac playback

The 079 playback archive contains four synchronized stereo programs with the target gain already lowered 6 dB, both standalone Bella cues, and a manifest. All four programs reproduced their tested microphone hashes across four geometries. Anonymous download verification matched the 1,211,840-byte archive and all seven members. SHA256 is `430daf90fd5faac6a998abb753422790e1f9f8bc1004be59bfc1dc2aa6b761bd`. [Mac 079 comparison](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-079-early-groups-3db-v1.zip) · [Member hashes and sources](../results/4090-bella-early-groups-3db-mac.json)

## Historical interruptions and overlapping identical cues

The new cue opened on time on 96/96 previously consumed interruptions and the 96 matched erased controls stayed closed. The controls reused the earlier native UARTs after checking the full serialized input. The issued three-group cue has the same totals. [Historical events and input identities](../results/4090-bella-early-groups-3db-history.json)

When the panel and an external speaker emitted the same cue simultaneously, the receiver stayed closed in all four geometries for both versions. At a +400 ms external delay, three geometries still opened and one previously open case stayed closed. In geometry 3, the identical reference veto windows covered both raw candidate clocks, at 47,620 and 54,020. The issued cue also had a raw clock at 60,420 after the last veto ended at 57,397, and opened through the marker path. Its allowed clock differed from both solo clocks. This is a regression in opening on overlapping identical sounds, with ambiguous source attribution for the earlier open. The other four delay settings retained the same counts. [Paired collision clocks](../results/4090-bella-early-groups-3db-collisions.json)
