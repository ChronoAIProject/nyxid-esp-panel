# Address-only gain for external playback

The two generated addresses were reconstructed from the three released weak-scene speaker pairs. Each track retained the original command, concurrent speech, simulated music and panel output. We tested 0 to 5 dB in 1 dB steps and the published 5.678 dB peak setting. Nyx first opened before the command at +1 dB in 077 and 090, and at +2 dB in 079. Across every higher tested setting the first open stayed at samples 44,288, 47,872 and 45,312, respectively. This left 32, 33 and 43 ms before the first command. The three files used to select the level share historical development rooms. [Source and playback protocol](bella_address_gain_protocol.md) · [Individual levels, microphone inputs and timing](../results/4090-bella-address-gain-wake.json)

The standalone P4 receiver and frozen C161-plus-marker guest agreed on every opening. The C161 neural path emitted zero events. The native official Willow DEFAULT receiver on the classic ESP32 guest missed its 21 matched Willow trials on these three previously missed sources. The 21 Willow presentations and three erased controls gave zero Nyx openings, and all digital microphone inputs had zero clipped samples. The room, player and input amplitudes here are simulation settings. [Native return sequences and both P4 UARTs](../results/4090-bella-address-gain-wake.zip)

| Address gain | Nyx timely opens / 3 | Whole command, immediate Qwen / Whisper turbo | Whole command, delayed Qwen / Whisper turbo |
| --- | ---: | ---: | ---: |
| 0 dB | 0 | 0 / 0 | 0 / 0 |
| +1 dB | 2 | 0 / 0 | 0 / 0 |
| +2 dB | 3 | 0 / 0 | 0 / 0 |
| +3 dB | 3 | 0 / 0 | 1 / 0 |
| +4 dB | 3 | 1 / 0 | 1 / 0 |
| +5 dB | 3 | 0 / 0 | 0 / 0 |
| +5.678 dB | 3 | 0 / 0 | 0 / 0 |

All 17 Nyx opens have complete capture, Opus and sender evidence including 14 fresh paths. The three peak-gain paths reuse the earlier full-mixture logs on identical M/R and opening clocks. The two independent GPU readers processed the full received stream. For comparison, when both address and command were previously doubled together while the competing speaker stayed at the original gain, Qwen read two of the three immediate commands and Whisper read one. [Address-only transcripts](../results/4090-bella-address-gain-commands.json) · [Earlier address-and-command comparison](../results/4090-c161-bella-weak-speaker-gain.json)

The public playback ZIP contains 45 mono 16 kHz PCM24 WAVs and a manifest. Each external track uses the same quarter-scale headroom as the existing room playback pack. The original music and timed panel tracks are shared across levels. Start each source at sample zero with the same physical player volume and positions. The three level series include the target names, all original suffixes and an erased-address control per scene. Anonymous download matched the archive and all 45 WAV hashes. ZIP SHA256 `3128f9cdb1ad233bf1b30f8ffb1509d1f45186093024a707c2aceacef3e11d3b`. [External playback WAVs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/hye-nyx-willow-address-gain-v1.zip) · [Room tracks](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/c161-bella-weak-speaker-gain-room-audio.zip)

The separate sender ZIP has SHA256 `2df9dce38ee670edbfd2c86875841c297b598480002c2c89d04ce733063892fd`. Its 51 UART members matched their recorded hashes after anonymous download. [Download sender UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-bella-address-gain-sender-uart.zip)
