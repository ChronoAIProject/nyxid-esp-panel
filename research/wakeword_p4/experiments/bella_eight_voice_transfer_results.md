# The delivered early marker on eight synthetic voices

The known-room result transferred to all eight Kokoro voice carriers. Each spoke one pre-existing synthetic address. It was joined to the same hall-light command while the opposite speaker asked the time. The table counts opens before the command.

| Measured path and target-player gain | Single triplet | Transferred double triplet | Plain Nyx | Same-voice official Willow |
| --- | ---: | ---: | ---: | ---: |
| Aachen hall 0°, original level | 8/8 | 8/8 | 0/8 | 0/8 |
| Surrey 240° front, original level | 0/8 | 0/8 | 0/8 | 0/8 |
| Surrey 240° front, target +6 dB | 8/8 | 8/8 | 0/8 | 1/8 |

In the hall the early triplet moved the opening forward exactly 6,400 samples (400 ms) for each voice. In Surrey at +6 dB the two versions always opened on the same read, 1,123 samples (70 ms) before the command. All Nyx openings used the marker route. C161's model route produced zero events on these 99 microphones. On Willow's own address it also opened late on one Surrey original-level and three Surrey +6 dB presentations. Each receiver stayed closed on the other address and on all three shared erased controls. The original-level inputs included one clipped sample. The Surrey +6 dB inputs had zero clipped samples. [Original inputs and native clocks](../results/4090-bella-eight-voice-transfer.json) · [66 paired native comparisons](../results/4090-bella-eight-voice-transfer.zip) · [Target-level step](../results/4090-bella-eight-voice-level.json) · [33 paired native comparisons](../results/4090-bella-eight-voice-level.zip)

The source fit transferred the released Bella early marker to the available PCM16 headroom of each existing synthetic carrier. The original speech and later markers remained identical to each voice's single-triplet control outside the three added windows. Lily could accommodate 89.3% of the proposed increment, the other seven accepted the full increment. The exported Bella PCM matched the delivered A samples exactly. With the two whole-prefix readers River transcribed as “Hey, Nick.” from both. Echo transcribed as “Hey, next is.” and “Hey next is”. The application can reproduce these sounds directly. The experiments measure playback of eight fixed WAVs. Recognition of ordinary spoken Nyx in these two tested placements was 0/8. [Source procedure](bella_eight_voice_transfer_protocol.md) · [Level procedure](bella_eight_voice_level_protocol.md) · [16 raw standalone readings](../results/4090-bella-eight-voice-readings.json)

The eight mono PCM16 WAVs and the manifest fit in 207,710 bytes compressed. Archive SHA256 is `67bd9b15d8100852ac5e97c0663880506d7cf4de5f69de956a22a63f7bad8fd6`. The anonymous download matched the archive and all eight WAV hashes. Mac's primary physical trial still uses A and its synchronized stereo playback from the existing handoff. [Eight-voice audio](../results/hye-nyx-eight-voice-transferred-cues.zip) · [Per-voice gain and source identities](../results/4090-bella-eight-voice-transfer-plan.json) · [Board handoff](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7033946232)

At the two placements where all eight double-triplet voices opened (Aachen original level and Surrey target +6 dB), both the 48 kHz PCM24 path and AAC-LC stereo at 128 kbit/s preserved every Nyx and Willow event clock from direct 16 kHz playback. Per player path the marked files opened before the command 16/16, official Willow opened before the command 1/16 and three more opened late. The two erased tracks stayed closed on both receivers. This software player check covers these eight two-group files and one command. [Registered codec test](bella_eight_voice_codec_protocol.md) · [68 microphone records](../results/4090-bella-eight-voice-codec.json) · [272 evidence files](../results/4090-bella-eight-voice-codec.zip)

## Three marker groups in a 1.45-second address

The three-group sound recovered all eight original-level Surrey 240° front presentations of command 077. The target speaker plays “turn down the hall light” and the opposing speaker asks the time. With two groups the registered receiver opened in 0/8 presentations here. The 24 new microphones all opened through the fixed marker, with zero C161 model events, zero reciprocal official Willow opens and zero clipped samples. The room responses, gains, command and noise match the previous experiments. Each placement reuses the same eight voices. [Procedure](three_triplet_eight_voices_protocol.md) · [Native opens and clocks](../results/4090-three-triplet-eight-voices.json) · [P4 and Willow evidence](../results/4090-three-triplet-eight-voices.zip)

| Placement | Two groups | Three groups | Same-voice official Willow | New lead before command |
| --- | ---: | ---: | ---: | ---: |
| Surrey 240° front, original level | 0/8 | 8/8 | 0/8 | 470 ms |
| Surrey 240° front, target +6 dB | 8/8 | 8/8 | 1/8 | 870 ms |
| Aachen hall 0°, original level | 8/8 | 8/8 | 0/8 | 873 ms |

Each exported WAV contains 23,200 samples of mono 16 kHz PCM16. All eight carriers accepted the full proposed pair of early triplets at their own source-relative gain. Bella exactly matches the three-group WAV already issued to Mac. On the complete standalone River file Qwen and Whisper both transcribed “Hey, Nick.” On Echo they returned “Hey, next is.” and “Hey next is”. The released pack is 264,200 bytes. The anonymous download matched the ZIP and every listed WAV, SHA256 `04b5ce086293e14568fdfe6963a35284904bc7b876dc31ff7d15c7bb9fd828d0`. [Eight standalone WAVs and manifest](../results/hye-nyx-eight-three-triplet-voices.zip) · [All 16 transcriptions](../results/4090-three-triplet-eight-voices-readings.json)
