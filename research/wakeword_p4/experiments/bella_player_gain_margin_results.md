# Playback level and rate of the generated Hye Nyx cue

At +5 dB the C161-plus-marker receiver opened before the command in all three selected weak rooms at all three near-nominal player speeds. The lead ranged from 31.75 to 43.31 ms. At +3 and +4 dB, room 079 still missed when the external player was 100 ppm fast. At ±1% every Nyx attempt missed across all four levels shown. The fixed marker produced every Nyx opening in the new sweep. The decibel values are relative to the original digital cue, with player volume and room transfer held fixed. [Frozen procedure](bella_player_gain_margin_protocol.md) · [Per-input native clocks](../results/4090-bella-player-gain-margin.json)

| External player speed | +2 dB | +3 dB | +4 dB | +5 dB |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 3/3 | 3/3 | 3/3 | 3/3 |
| −100 ppm | 2/3 | 3/3 | 3/3 | 3/3 |
| +100 ppm | 1/3 | 2/3 | 2/3 | 3/3 |
| −1% | 0/3 | 0/3 | 0/3 | 0/3 |
| +1% | 0/3 | 0/3 | 0/3 | 0/3 |

Official Hey Willow missed on all 45 paired Willow playbacks at +3 to +5 dB. Both receivers stayed closed on all 45 other-address trials and all 15 shared erased controls. None of the 90 new microphone inputs clipped. All 18 nominal microphone/reference pairs and both receivers' event clocks were checked against the original gain sweep. Each speed changes the whole external player, including the competing talker. The table counts three reused rooms and samples five discrete speeds. [Frozen input hashes](../results/4090-bella-player-gain-margin-plan.json) · [360 native evidence files](../results/4090-bella-player-gain-margin.zip) · [+2 dB and erased controls](bella_external_player_rate_results.md)

## Standalone +5 dB file and nominal-speed replay

The isolated synthetic WAV is 1.45 seconds, 16 kHz mono PCM16, 46,444 bytes, with a sample peak of 27,012. Its SHA256 is `75952a6b0a0779a37b8718847b36989ed3c22ac0bb3a5e9802d7de4316c50997`. Its PCM equals the isolated +5 dB address contribution in all three selected external source tracks. [Download +5 dB WAV](../results/hye-nyx-af-bella-plus5db.wav) · [+2 dB WAV](../results/hye-nyx-af-bella-plus2db.wav)

Both the marker-only guest and the primary C161-plus-marker P4 guest opened before the command in all 96 historical interruptions. Every first-open sample equaled the corresponding +2 dB marker-only sample. The marker supplied all 96 primary opens. Both guests stayed closed on all 96 paired erased inputs. These 96 controls belong to this specific playback set. On four panel-only room paths, each guest had four opens with silent playback reference and zero with synchronous reference. The external microphones had zero clipped samples. Panel geometry 0 clipped two samples, shared by both reference conditions. [400 native replay logs](../results/4090-bella-gain-candidate-playback.zip) · [Per-path decisions and hashes](../results/4090-bella-gain-candidate-playback.json)

## Correctly declared 44.1 and 48 kHz output

After SoX high-quality conversion from 16 kHz through a 24-bit 44.1 or 48 kHz player and back, every native event sample on all 54 microphones equaled its direct 16 kHz baseline. The converted microphone bytes differed in every case. Each rate retained 4/4 quiet Nyx and 4/4 quiet Willow openings. In the three weak rooms Nyx retained 3/3 at both +2 and +5 dB, and Willow remained at 0/3 at both levels. All erased and cross-address inputs stayed closed, with zero ADC clipping. Both impulse peaks returned at the original sample. These digital conversions did not exercise a Mac DAC or a speaker. [Procedure](bella_player_src_protocol.md) · [Source and converted PCM hashes](../results/4090-bella-player-src-plan.json) · [54 native event pairs](../results/4090-bella-player-src.json) · [216 evidence files](../results/4090-bella-player-src.zip)

## Where the ±1% weak-room misses arise

The three-part correlation calculation on the 4090 reproduced all 60 raw marker decisions and their sample clocks against the compiled portable matcher. All 31 positive reports reached the guest's reference gate output 480 samples (30 ms) after the raw match. The original diagnostic windows used the gated timestamp, which excluded the true peaks. The corrected analysis uses the raw nominal +5 dB match as its anchor.

At ±1%, correcting only the expected spacing for the known speed recovered 0/24 missed presentations. Even permitting any same-polarity spacing inside the three local ±25 ms windows recovered only 2/24, both in slow case 079 at +4 and +5 dB. At +5 dB, five of six misses had a peak absolute correlation below 0.30 on the first or middle code. Stronger playback and spacing correction therefore leave limited individual-pattern correlation on these inputs. [Diagnostic procedure and clock correction](bella_player_marker_timing_protocol.md) · [Correlations, event clocks and firmware coefficients](../results/4090-bella-player-marker-timing.json)

## GPU fit of the isolated marker codes

A 4090 fit changed 3,930 samples in the three code windows while preserving the vocal samples outside them and the original code energies. In the weak rooms, the experiment recovered five of the six ±1% misses. It lost six of the nine timely openings at nominal and ±100 ppm. The original +5 dB file stays the hardware handoff candidate. [Fitting protocol](bella_multirate_stems_protocol.md) · [GPU weights, before and after similarities](../results/4090-bella-multirate-stems.json) · [15 measured clocks](../results/4090-bella-multirate-stems-wake.json) · [Native UARTs](../results/4090-bella-multirate-stems-wake.zip)

| Player speed | +5 dB | Three fitted codes |
| --- | ---: | ---: |
| Nominal | 3/3 | 2/3 |
| −100 ppm | 3/3 | 1/3 |
| +100 ppm | 3/3 | 0/3 |
| −1% | 0/3 | 3/3 |
| +1% | 0/3 | 2/3 |

## Received first commands with the +5 dB handoff

Across the 96 historical interruptions at nominal speed, Qwen3-ASR and Whisper turbo read fewer complete commands with +5 dB despite the identical open samples. Under immediate readiness Qwen gained 3 and lost 9 previously complete cases, and Whisper gained 1 and lost 7. Under delayed readiness the corresponding changes were 8 gained and 9 lost, and 3 gained and 4 lost. The address-erased cases were evaluated for openings in the wake replay. [Predeclared received-command procedure](bella_gain_candidate_commands_protocol.md) · [108 addressed cases, both reads and clocks](../results/4090-bella-gain-candidate-commands.json)

| Address level | Timely / 96 | Immediate, Qwen / Whisper | Delayed, Qwen / Whisper |
| --- | ---: | ---: | ---: |
| +2 dB | 96 | 38 / 23 | 37 / 17 |
| +5 dB | 96 | 32 / 17 | 36 / 16 |

For the three weak rooms, neither reader recovered a complete first command under either readiness condition at either 100 ppm offset, including the six +5 dB opens. The 105 successful sends across the historical and clock-offset inputs have 315 capture, Opus and sender UARTs. The anonymous Release download matched all members and the ZIP SHA256 `4e89bb2cc5502635237c69474afa97e98d6f290b4d32df8183f6d28413d4d9e8`. [Download sender evidence](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-gain-candidate-commands-uart.zip)
