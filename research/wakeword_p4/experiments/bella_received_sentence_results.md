# Keeping the first command in a full received transcript

On the 96 historical +2 dB Nyx interruptions with competing voice, music and panel playback, the full and cropped readings gave complementary evidence. Keep the full reading unless the shorter reading matches its final words and the omission ends at a sentence boundary or is the configured address. Each comparison uses one reader twice. Qwen3-ASR and Whisper turbo were scored separately on the RTX 4090. [Registered rule](bella_received_suffix_protocol.md) · [Per-input texts, decisions and source identities](../results/4090-bella-confirmed-sentences.json)

| 96 historical interruptions | Full reading | Audio crop | Word-midpoint filter | Two readings with boundary check |
| --- | ---: | ---: | ---: | ---: |
| Only the command, immediate, Qwen / Whisper | 22 / 9 | 35 / 16 | 14 / 13 | 34 / 16 |
| Complete command suffix, immediate | 38 / 23 | 35 / 18 | 17 / 16 | 38 / 23 |
| Only the command, delayed | 18 / 7 | 37 / 13 | 15 / 9 | 32 / 12 |
| Complete command suffix, delayed | 37 / 17 | 37 / 15 | 19 / 11 | 37 / 17 |

The word-midpoint filter used Qwen's published forced aligner on its own predicted words and Whisper's own word timestamps. Both used the same previously tested boundary 100 ms before the first open, measured on decoded audio. In quiet rooms the filter preserved every complete Willow suffix at both readiness settings. On the historical interruptions it removed desired words from 21 previously complete immediate Qwen readings and seven immediate Whisper readings. [Word-time protocol](bella_received_word_time_protocol.md) · [Estimated intervals and both sets of matched outcomes](../results/4090-bella-received-word-times.json)

| +2 dB address, each out of 12 rooms | Opened | Only the command, immediate full → boundary check, Qwen / Whisper | Delayed full → boundary check |
| --- | ---: | --- | --- |
| Nyx, quiet | 12 | 1 / 1 → 10 / 10 | 2 / 1 → 10 / 10 |
| Nyx, competing voice | 12 | 0 / 1 → 10 / 8 | 1 / 1 → 11 / 8 |
| Willow, quiet | 12 | 0 / 3 → 9 / 6 | 0 / 3 → 11 / 7 |
| Willow, competing voice | 4 | 0 / 0 → 3 / 3 | 0 / 0 → 1 / 3 |

On the separate ordinary spoken synthetic recordings, frozen C161 opened in 5/12 Nyx cases, 150 ms detector confirmation in 9/12, and official Willow in 8/12 on its own address. Under immediate readiness the whole transcript matched in 4 / 3, 6 / 4 and 8 / 8 cases with the boundary check, up from zero for both readers on each full-transcript group. Under delayed readiness the corresponding counts were 4 / 1, 6 / 0 and 8 / 7. Across the three routes, 20 distinct waveform/open-clock combinations supplied the received streams. These observations reuse the same 12 Nyx and 12 Willow inputs. [Ordinary speech clocks and native receiver evidence](c161_parts_confirmation_results.md)

Every previously complete final command survived the boundary check in these cohorts. The known false open on the unaddressed case 072 still produced a complete command suffix in both full and confirmed readings under both schedules. Its competing “Hey Nick” and an intelligible unaddressed command appear in the same send. No method here measured a real service action. All target commands in these cohorts contain a single sentence, and the score compares lower-case alphabetic words. [Control microphone and source mutes](../results/4090-c161-bella-gain-false-wake.json)

To repeat the offline comparison on Mac, transcribe both the complete decoded receive WAV and the 100 ms crop with the same reader. Give the two actual texts to `confirm_received_sentence.py --address nyx --full 'I have finished washing the dishes. Can you unlock the side door?' --cropped 'Can you unlock the side door?'`. The tool returns the original-cased suffix and the reason for its selection. Physical board clocks, receiver recordings and service latency remain to be measured. [Standalone text comparison](../first_command_p4/mac_board_marker/confirm_received_sentence.py) · [WAV crop tool and checked listening example](bella_plus2_playback_results.md#starting-recognition-100-ms-before-the-first-open)
