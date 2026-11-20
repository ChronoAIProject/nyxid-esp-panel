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
