# Standalone rev1 receiver for the eight generated Hye Nyx sounds

The fixed-sound receiver now uses the same 80 ms playback recovery patches as the C161 receiver. In the P4 digital fixture all 100 microphone/reference pairs produced the same accepted marker clocks as the C161-plus-marker build. The two everyday challenges and the plain synthetic Bella voice stayed closed on the standalone receiver. The C161 model opened on all three in the comparison. This build supports playback of the marked sound. [Registered procedure](eight_local_fixed_rearm_protocol.md) · [Comparison with both released listeners](eight_local_fixed_only_results.md) · [100 measurements](../results/4090-eight-local-fixed-rearm.json) · [100 UARTs](../results/4090-eight-local-fixed-rearm.zip)

| Selected input | Released standalone | Standalone with recovery |
| --- | ---: | ---: |
| 079, eight new voices through PCM48 and AAC, timely | 16/16 | 16/16 |
| 077, eight new voices in three room/level conditions, timely | 24/24 | 24/24 |
| Panel plays new Bella with electrical reference, false opens | 2/4 | 0/4 |
| Same panel microphone with zero reference, opens | 4/4 | 4/4 |
| “Six cups” and unaddressed speech/talk/music, false opens | 0/2 | 0/2 |
| Plain Bella, quiet 079 carrier, opens | 0/1 | 0/1 |

Both receivers opened 471 ms before the command on the new 079 voices. The four synchronized self-play cases now stayed closed through the end of playback. In collisions with an identical external cue, exact simultaneous playback stayed closed in all four geometries. At +400 ms the new receiver opened in three geometries and at +800 ms it opened in four. Each of those seven first marker clocks differed from both isolated-source clocks. With the external cue leading by 800 ms, all four openings matched the external-alone marker clock. These five offsets reuse the same four room transfers. Together with the selected daily controls they add zero independent background hours.

The second listening cycle in all six designated runs used 9,492 bytes of external heap and zero additional internal heap bytes. All six returned to their starting heaps on release. The matched C161 receiver used 13,308 external and 94,136 internal bytes. The rev1 hardware trial covers the physical reference, I2S timing, minimum heap under a real call, peer audio and repeated use. [Mac board and development-profile instructions](../first_command_p4/mac_board_marker/README.md#external-screen-rev1-with-only-the-generated-wake-sound)

The public-service rev1 app is 6,429,184 bytes with 452,096 bytes free in each OTA slot. The app contains zero copies of the frozen model. The anonymously downloaded ZIP and its two members matched the local archive. Its SHA256 is `ab767fd6f0e749ac4acfe2cbccb16dc790083706fdfed66176576b83c14656a4`. [Download rev1 app](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-fixed-sound-rearm-p4-rev1-public-v1.zip) · [Manifest and member hashes](../results/4090-eight-local-fixed-rearm-rev1-download.json)
