# First sentence received after the eight generated wake sounds

All 32 native calls opened on the fixed marker. They use the same two commands, with only the generated wake voice changed within each condition. Each call passed native capture, Opus and sender checks. At the listening peer the complete 079 sends mixed the competing request to turn on the light with the target request for the time. Locating the end of the known cue from the microphone samples available at open and keeping a 100 ms guard gave Qwen the target sentence alone in 8/8 immediate and 8/8 delayed sends. The earlier Bella report measured the full send. The cropped version of that same call reads “Could you check the time for me?” in Qwen on both schedules. Whisper still reads “Can” on Bella. [Procedure](eight_local_peer_protocol.md) · [Exact transcripts, clocks and PCM identities](../results/4090-eight-local-peer.json)

Every entry is out of eight and lists Qwen / Whisper turbo. The full columns require the exact command at the end of the transcript. The cropped columns require the exact command as the entire transcript.

| Playback | Full, immediate | Cropped, immediate | Full, delayed | Cropped, delayed |
| --- | ---: | ---: | ---: | ---: |
| Surrey 210° weak target 079 | 2 / 1 | 8 / 0 | 3 / 2 | 8 / 1 |
| Surrey 240° original 077 | 8 / 4 | 8 / 1 | 8 / 3 | 8 / 0 |
| Surrey 240° target +6 dB 077 | 8 / 2 | 8 / 2 | 8 / 3 | 8 / 4 |
| Aachen aula original 077 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Across the 32 openings the estimate differed from the annotated end by −2 to +259 samples (−0.125 to +16.19 ms). For all eight Aachen aula recordings the signal passed the wake detector while both readers missed the command on both schedules. The crop runs offline on decoded peer audio using microphone and peer clocks. The board and the normal sender still send the full preroll. [Native capture, codec and sender UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-eight-local-peer.zip)

The Mac command-window tool now accepts the issued IEEE float stereo WAVs directly and preserves raw channel bytes for hashes. It recovered the same command start at sample 48,400 (within four samples) in all four 079 playback files when compared to themselves. The existing PCM16/24 capture checks also passed. [Mac replay command](../first_command_p4/mac_board_marker/README.md#eight-voice-external-playback-and-weak-target-079)

## Aachen source controls

Both GPU readers transcribed the dry Bella 077 command as “Turn down the hall light.” In the full measured room, Qwen recovered the entire command when the opposing speaker was silent, and Whisper returned “Turn down the whole light.” On the raw mixture Qwen returned “The whole night.” and Whisper returned “and a whole life.” The raw mixture matched the previous native microphone SHA256. Keeping the first 50 ms after each direct arrival and removing later reflections gave the exact command on both readers even with the opposing speaker present. Target-to-interference energy over the active command was +13.30 dB with the full paths and +13.47 dB with the shortened paths. In the full paths 41.7% of target impulse-response energy and 9.6% of opposing-response energy arrived after the cutoff. This source control uses measured room filters available in simulation. [Method](aula_077_source_controls.md) · [Source-isolated readings, energies and hashes](../results/4090-aula-077-source-controls.json)

## 400 ms timing control on the frozen Bella program

With the full Aachen room response, delaying only the native 077 command by 400 ms recovered “Turn down the hall light” on the raw microphone in Qwen. In both peer sends Qwen read “Turn down the whole light.” Delaying both the target command and the opposing speaker together produced “What time is it?” on both cropped peer reads from both models. Shifting the entire target including the wake cue recovered the exact command in Qwen only on the delayed, cropped send. All four native presentations opened on time through the fixed marker. These comparisons do not support prescribing a fixed speaking pause. The exact texts, source clock offsets and separate observed cue and command boundaries are retained. [Timing procedure](aula_077_command_gap.md) · [Native events, transcripts and member hashes](../results/4090-aula-077-command-gap.json)

The four stereo programs use the same Bella cue and gain, and can be played from the same two speakers on Mac. The 687,192-byte release and its five members matched their SHA256 values on anonymous download. Archive SHA256 is `488b36fe705bf5ed6057ea6cb14153db63d550622d4ca2cd95f2b608011280f9`. [Aachen 077 timing comparison](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-aula-077-gap-v1.zip)
