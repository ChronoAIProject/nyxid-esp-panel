# External +2 dB Hye Nyx WAV for Mac

The file contains the original 1.45 s af_bella playback sound at +2 dB rounded to PCM16. It has 23,200 samples at 16 kHz, a peak of 19,123 counts and a size of 46,444 bytes. SHA256 `0e8fe35d87247bded97335f5e8d81ffbf65fe8f82e1a4b070c4f1b0c248c5a7d`. Its samples exactly equal the isolated Nyx contribution in each of the three +2 dB playback files. Preserve the same external player setting when comparing with the original. [Download isolated WAV](../results/hye-nyx-af-bella-plus2db.wav) · [Original-level WAV](../results/hye-nyx-af-bella-marker-rms-v1.wav) · [Three-scene gain comparison](bella_address_gain_results.md)

The standalone P4 simulated receiver opened before the command in all 96 historical interruptions, including the three previously used to select this gain. The original-level marker had opened in 93. The other 93 cases remained timely. Across the same 96 sources with the address erased there were zero openings. The four panel-only room paths gave zero self-opens with the synchronous playback reference and four opens with silent reference. All 200 successful UARTs and their digests are available. These scenes reuse the 24 simulated room/command combinations, three source levels, and 32 interruption texts. [Frozen playback comparison](bella_plus2_playback_protocol.md) · [Per-input clocks and levels](../results/4090-bella-plus2-playback.json) · [Raw UARTs](../results/4090-bella-plus2-playback.zip)

## Complete received commands

Each external microphone with the +2 dB cue used its own first native open, original electrical reference, P4 capture, Opus encoder and sender. All 96 enter the counts below, including the three full-mixture cases reused from gain selection. The two values in each cell count complete commands according to Qwen3-ASR and Whisper turbo. The two simulated service conditions used the same source recordings.

| External address | Opened before command | Complete, service ready | Complete, delayed service |
| --- | ---: | ---: | ---: |
| Original level | 93 | 34 / 23 | 42 / 16 |
| +2 dB | 96 | 38 / 23 | 37 / 17 |
| +5.678 dB | 96 | 31 / 19 | 33 / 13 |

Against the original, immediate Qwen gained eight individual successes and lost four. Delayed Qwen gained four and lost nine. Immediate Whisper gained and lost four, and delayed Whisper gained four and lost three. All three recovered weak openings still lost the full first command under both readers. The 96 sets of capture, codec and sender UARTs were downloaded anonymously and verified member by member. ZIP SHA256 `fda5d66b94cb446101aebba76fe7296d62899b94d976b1a9caa185d531c5b3e9`. [Per-case clocks and complete transcripts](../results/4090-bella-plus2-commands.json) · [Download 288 UART logs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-plus2-commands-uart.zip)

## Twelve prospectively generated rooms

After the gain was frozen, 12 new rectangular rooms were generated with one direct path, six wall reflections, low microphone noise and 12 existing synthetic conversation recordings from the training inventory. Each room used five address conditions twice, once quiet and once with a simultaneous competing voice. All 120 microphones had zero clipped samples. Both detectors ran on every microphone. These trials measure when a detector opens, using the arrival of the direct-path command at the mic. [Source placement and design](bella_prospective_room_protocol.md) · [Source and waveform hashes](../results/4090-bella-prospective-rooms-plan.json)

| In time for command, out of 12 | Quiet, original | Quiet, +2 dB | Competing voice, original | Competing voice, +2 dB |
| --- | ---: | ---: | ---: | ---: |
| Hye Nyx fixed-sound P4 receiver | 12 | 12 | 12 | 12 |
| Official Hey Willow DEFAULT | 12 | 12 | 1 | 4 |

Neither detector opened on the other address or on an erased control. Nyx reached the ceiling at both source levels in this simpler digital room model. The earlier gain advantage came from the historical weak interruptions with music and panel playback. The Mac board session supplies acoustic validation and the physical heap and sender measurements. [120 native comparisons](../results/4090-bella-prospective-rooms.json) · [480 UART, return and report members](../results/4090-bella-prospective-rooms.zip)
