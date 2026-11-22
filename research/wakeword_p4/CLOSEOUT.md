# Hye Nyx GPU research closeout

Local work on the RTX 4090 ended on 2026-11-22 at the owner's request. Published experiments and release assets are available on the research branch and the research release. The local-only training materials, checkpoints and scratch runs were designated for deletion.

## Recorded outcome

- On the ESP32-P4 rev1, the C161 firmware with the fixed marker opened the microphone and delivered the full command in all four Bella/Echo playback trials across two volume settings. The marked and erased 079 left-channel examples produced the expected open and closed outcomes. Same-board Hey Willow comparison, measured opening lead time, overlapping playback/listening and long-duration stability were still open at closeout. [Mac board receipt](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7041479995)
- C161 remains the unmarked baseline at threshold 0.91796875 and smoothing 5. Digital replay of the three original Mac recordings opened for Marin and Sage and missed Echo. Binary40 opened only for Sage. Both downstream recognizers received the complete command on every simulated opening under both sender timing conditions. [Voice-by-voice results](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7042347237)
- At closeout, an anonymous download of the published C161 model decompressed to 60,952 bytes with SHA256 `c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81`. The original v30 handoff downloaded anonymously with archive SHA256 `60000e37f1b01c408f48f6310bfa7c6d0e837fe6d808127aab9de9a88e1e3cb5`. [C161 model](https://raw.githubusercontent.com/ChronoAIProject/nyxid-esp-panel/4815901d6598e35bd64913cb38c70ebd202a95e7/research/wakeword_p4/results/4090-c161-mac-development-model-v1.tflite.gz) · [Frozen v30 handoff](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-v30-p4-handoff-v1.tar.gz) · [Board firmware and playback instructions](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/4815901d6598e35bd64913cb38c70ebd202a95e7/research/wakeword_p4/first_command_p4/mac_board_marker/README.md)

## Local storage

All 1,972 inventoried paths were absent after cleanup. Their pre-cleanup allocation was 283,738,144,768 bytes (264.25 GiB), counted once across the inventory with GNU du. The research scratch directory accounted for 258.79 GiB. This measurement follows the earlier 63.37 GiB cleanup.

| Local copies cleared | Allocated bytes before cleanup |
| --- | ---: |
| Training data, downloaded models, checkpoints, simulators and six nested board worktrees | 277,873,377,280 |
| Ignored outputs in the research source directories | 32,034,816 |
| Voice-model and ESP-IDF component caches outside the worktree | 1,268,043,776 |
| Research temporary files and downloads | 1,063,833,600 |
| Published results and fixture omitted from the local checkout | 3,500,855,296 |

Downloaded models included Qwen, Whisper, Parakeet, Chatterbox, SpeechT5, Parler and Kokoro. This worktree uses Git sparse checkout for the published results. After cleanup the checkout occupied 106.21 MiB, excluding the repository's shared Git database. The filesystem had 744.31 GiB available at verification. The system-wide PyTorch installation was retained.
