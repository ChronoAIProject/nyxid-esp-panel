# External player rate on the frozen +2 dB call

At the nominal speed, the primary C161-plus-marker listener opened 32, 33 and 43 ms before the command in three previously selected weak rooms. At a 100 ppm slower source speed it retained cases 077 and 090. At 100 ppm faster it retained only 077. The competing speech shares this external player. Music, panel audio and the electrical reference use the receiver clock. A changed result can reflect changes in relative phase or arrival time as well as pitch. The three inputs were previously used to select +2 dB. [Registered procedure](bella_external_player_rate_protocol.md) · [108 paired inputs and clocks](../results/4090-bella-external-player-rates.json)

| Source playback speed | Timely Nyx in weak rooms, out of 3 | Quiet Nyx lead | Quiet Willow lead |
| --- | ---: | ---: | ---: |
| Nominal | 3 | 80 ms | 304 ms |
| −100 ppm | 2 | 80 ms | 304 ms |
| +100 ppm | 1 | 80 ms | 304 ms |
| −1% | 0 | 76 ms | 332 ms |
| +1% | 0 | 68 ms | 276 ms |
| −5% | 0 | Miss | 355 ms |
| +5% | 0 | Miss | 171 ms |
| 44.1 / 48 | 0 | Miss | 360 ms |
| 48 / 44.1 | 0 | Miss | 404 ms late |

Official Willow missed all three weak inputs at every rate. Each listener stayed closed on all 36 other-address inputs and all 36 erased controls. All Nyx requests came from the fixed marker. No input clipped. The 12 nominal Nyx and Willow decisions matched the earlier experiments on byte-identical microphone and reference waveforms. The tests contain four source scenes and nine discrete speeds. The two extreme ratios deliberately change pitch and duration. [108 microphone hashes](../results/4090-bella-external-player-rates-plan.json) · [432 native evidence files](../results/4090-bella-external-player-rates.zip)
