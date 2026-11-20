# Two missed Hye Nyx calls at 240°

At the published player levels, both A (delivered cue) and B (headroom cue) missed front-speaker commands 077 and 079 at 240°. The calibrated target-player step of +6 dB recovered both calls on the same C161 firmware. The 12 presentations use the same three commands, two speaker directions and two orientations in one measured Surrey room. [Source-muting protocol](bella_240_masking_protocol.md) · [Target-player protocol](bella_6db_player_protocol.md)

| Timely openings across the 120° and 240° presentations | Published levels | Target player +6 dB |
| --- | ---: | ---: |
| A | 10/12 | 12/12 |
| B | 10/12 | 12/12 |
| Official Hey Willow | 6/12 | 9/12 |
| Address erased, Nyx / Willow | 0/12 each | 0/12 each |

The target-player gain applies to the address and the entire command, with the opposite speaker at its published level. All 48 original PCM hashes matched before the level change. At +6 dB every Nyx opening on its own address came through the marker route. The two receivers stayed closed on the reciprocal addresses. Willow also opened once 1,005 samples (62.8 ms) after command onset, for 10 total and nine timely opens. The largest absolute player sample was 0.499 full scale. In the simulated microphone, four versions of the same 240° back 079 command each clipped one sample after command onset. The address intervals had zero clipped samples. [Original clocks](../results/4090-bella-headroom-new-angles.json) · [New input plan](../results/4090-bella-6db-player-plan.json) · [New clocks](../results/4090-bella-6db-player.json) · [Native evidence](../results/4090-bella-6db-player.zip)

## Which part of the cue missed

The RTX 4090 analysis checked the three normalized fixed-pattern correlations in each of the two triplets. For front 077 and 079 at the original level, all individual peaks had absolute correlation between 0.305 and 0.486. In each triplet the first two patterns failed to produce a same-polarity pair 160 ms apart within the firmware tolerance of 1.5 ms. Even with the interruption muted, the early triplet on this path had no such pair. On the later triplet, muting the interruption or attenuating it by 6 or 12 dB restored the pair and the complete chain for both calls and both WAVs. Front 090 already completed the later chain at the original level. In the reverse direction all three commands opened at every interruption setting. The interruption-only controls opened 0/6. The first predicted marker sample agreed with the native receiver on all 54 inputs, including those with no marker event. These trials localize the failure to source balance and marker timing on this acoustic path. [Per-pattern signed extrema, compatible pair times, source levels and native clocks](../results/4090-bella-240-masking.json) · [54 UARTs](../results/4090-bella-240-masking.zip)

## First command received at 240° from the front

Each cell gives Qwen and Whisper turbo counts for the complete target-command suffix among all three presentations. A missing open counts as a miss. The actual opening starts the send path, including Willow's late opening on 090. Readers saw the whole decoded stream and sometimes included words from the interruption before the target command.

| Address | Original immediate | +6 dB immediate | Original delayed | +6 dB delayed |
| --- | ---: | ---: | ---: | ---: |
| A | 1 / 1 | 3 / 2 | 1 / 1 | 3 / 2 |
| B | 1 / 1 | 3 / 3 | 1 / 1 | 3 / 3 |
| Official Willow | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |

A and B opened on the same read in all three front cases at +6 dB. Whisper changed the requested wording on A's 079 on both schedules. It transcribed the complete target suffix on B. Both reported reader counts apply to these three already studied commands. All 21 send-path UARTs across seven openings had zero watchdog alerts. [Transcripts and clocks](../results/4090-bella-6db-front-commands.json) · [21 UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-6db-front-commands.zip)

## Playing the pair on the board

The pack contains 24 stereo 16 kHz IEEE float64 WAVs and a manifest. Left carries the target and right carries the synchronized interruption. Route them to opposing loudspeakers using one stereo output. With the same master gain and panel silent, play `077/early-nyx-baseline.wav` then `077/early-nyx-plus6.wav`. Repeat with `headroom`, `willow` or `absent` in the same position. Each plus6 file already includes the left-channel gain. The 079 and 090 directories use the same naming. Both PyAV and libsndfile decoded the exported inputs. Rendering the files through both measured orientations reproduced all 96 native microphone hashes. The anonymous Release download matched all 25 member hashes. ZIP SHA256 is `d349ac1157307ed47851e5f4e287e8c69d634862cb817ab21b4365f260fdf287`. [Stereo playback pack](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-stereo-player-6db-v1.zip) · [Member hashes](../results/4090-bella-6db-stereo-mac.json)

Keep A as the physical-board baseline and B as the paired candidate. On the board, record the actual M/R channels and first open relative to the command, minimum internal heap, and the service's first received words with the board variant, app/ELF identity and service configuration. Physical speaker level, channel routing, device decoding and P4 behavior still require the Mac setup. [Board firmware and configuration](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7032582923)
