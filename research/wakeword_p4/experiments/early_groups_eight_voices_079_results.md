# Voice transfer on weak 079

At Surrey 210° front with the target 6 dB lower, all eight issued three-group voices missed 079. With the early-group gain limited by each voice's single highest peak, four opened. River, Eric, Fable and Bella all reached the same marker at 41,328 and opened at 41,472. The achieved increases were 3.00, 2.92, 3.00 and 3.00 dB respectively. Lily, Echo, Isabella and Lewis had only 0.31, 0.27, 1.39 and 0.25 dB available under this rule and stayed closed. The single erased control stayed closed. All eight share the same command, interruption, room and noise. [Paired clocks and per-voice gains](../results/4090-early-groups-eight-voices-079.json)

## Local gain test

Apply the same requested +3 dB to the same per-voice increment using a time-varying gain, without changing the speech carrier or final group. Derive the maximum gain separately for each source sample, dilate each required reduction by 32 samples on each side, then smooth with a 65-sample rectangular average. This keeps the applied gain below the available PCM16 headroom at every sample. Voices that already accepted the full boost must remain byte-identical. Compare all eight on the same 079 test and against the earlier 24 Surrey and aula presentations of command 077. Compare all opens, losses and saturated samples. Freeze the source files before measuring.
