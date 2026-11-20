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
