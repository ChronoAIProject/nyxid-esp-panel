# Weak-target 079 after 48 kHz PCM and AAC playback

All eight generated voices triggered on time through the fixed marker in both 48 kHz player paths. In the Surrey 210° front simulation the target was 6 dB quieter and the opposing speaker was at original level. Each of the 16 new-voice microphone streams produced the same marker clock as its direct 16 kHz counterpart, at sample 41,328. The receiver opened at 41,472, 471 ms before the command. The code and all converted microphones were frozen in `2adeb7be` before scoring. [Procedure](eight_local_player_079_protocol.md) · [36 native comparisons](../results/4090-eight-local-player-079.json) · [144 firmware evidence files](../results/4090-eight-local-player-079.zip)

| Native receiver on its target | Direct 16 kHz | 48 kHz PCM24 | 48 kHz AAC-LC |
| --- | ---: | ---: | ---: |
| C161 plus fixed marker, issued local gain | 8/8 | 8/8 | 8/8 |
| Same receiver, prior three-group voices | 0/8 | 0/8 | 0/8 |
| Official Hey Willow, single issued Willow address | 0/1 | 0/1 | 0/1 |

All 18 direct microphone hashes reproduced the original measurements. The two converted erased controls stayed closed on both receivers. The official Willow runner stayed closed on every converted Nyx address, and the Nyx receiver stayed closed on the two Willow inputs. The P4 receiver logged zero C161 model events across all 36 inputs. The GPU room render produced zero clipped samples. Both conversion paths preserved the two-channel impulse clock, with separate receiver runs on each converted microphone. All cases reuse the same simulated room, one spoken command and one interruption. The previous first-sentence evidence uses the direct player. [First-sentence results](eight_local_peer_results.md)

## Mac playback files

The comparison ZIP contains Bella old, Bella new, erased and Hey Willow in each of two formats. The PCM WAVs are 48 kHz stereo signed PCM24. The M4A files contain the 128 kbit/s stereo AAC-LC bytes used in the experiment. Each of the eight exported programs reproduced its registered player stream by hash. Keep the player and speakers at the same settings and play complete files. For a peer recording, the command-window reader accepts the PCM48 source directly, with `--command-at 145200`. Its reported match clocks are in 16 kHz analysis samples. In the original 16 kHz pack the same boundary is at 48,400. Eight CLI comparisons of the four PCM48 sources with the corresponding 48 kHz and 16 kHz recordings recovered all three command windows. [48 kHz playback ZIP](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-079-48k-player-v1.zip) · [Member hashes and sources](../results/4090-eight-local-player-079-mac.json)

The 2,655,776-byte ZIP and all nine members were reverified on anonymous download. Archive SHA256 is `b36881d8c2ae49f0d56ab621589d3758799d7126391014d86d121ac8d2682c4b`.
