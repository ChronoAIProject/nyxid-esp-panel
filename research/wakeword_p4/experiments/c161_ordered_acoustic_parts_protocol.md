# Ordered acoustic parts for the spoken address

On the existing four-voice connected synthesis set, frozen C161 opened on 5 of 12 literal “Hey Nyx” and 5 of 12 “Hey Nicks” utterances. All ten requests came after the command started. A learned score for a whole fixed window can depend on where each sound falls inside that window. Test whether allowing the time between three short acoustic patterns to vary helps on the same connected commands and Joe's cross-voice recordings.

Train two compact networks on RTX 4090 with the same synthetic training recordings, seed, optimizer and source draws. Each observes the preceding five 30 ms native 20-band frames. Both learn three local pattern scores with a shared 16-unit hidden layer. For the fixed-spacing arm, sum the scores in order with gaps of five and six frames. For the variable-spacing arm, maximize this sum over gaps of two to eight and three to nine frames, respectively. All samples used for a decision precede the current read. The patterns have weak keyword supervision and carry no individual phone labels.

Use the prior 918-record word training inventory, deduplicate exact PCM and split whole conditioning voices by sorted SHA256, every fifth voice for calibration. Include 1.5 s silence before each original source and 1 s after it, following the standard 81,600-sample continuous prime. Learn from the strongest two-consecutive-frame response in the final 270 ms before the supplied training address boundary. Negatives contribute their strongest response over the whole real stream. Fit 700 AdamW steps per arm. Export signed int8 weights, 32-bit biases, clamped uint8 hidden outputs and integer part scores. Set each cutoff one integer unit above the strongest negative calibration recording. Keep both cutoffs frozen throughout development. Register separate raw detections and their union with C161 under its 67-call cooldown.

Use previously consumed ordinary commands, all Joe and Jenny synthetic speech and recorded daily sentences to compare target coverage, timing, common-word conflicts and reciprocal triggers. Check exposure on the already-consumed full background files. Keep all repetitions within their original cohorts. An additional branch that causes household false wakes needs further work before board integration. Preserve both trial weights and the original C161. [Original connected measurement](../results/4090-c161-connected-nick-willow.json) · [Training source inventory](../results/4090-c161-music-proposal-training-v1.json)

The two 1,724-byte integer heads used 723 distinct fitting waveforms and 192 calibration waveforms from three other training voices. The strict cutoff was set by “Hey Mix” for fixed spacing and “Hey Nick” for variable spacing. At those cutoffs neither arm accepted any of the 44 calibration positives. The follow-up kept both weight arrays and set the cutoff from 82 calibration negatives without elicited Hey-addresses. “We need six mugs for breakfast” set both cutoffs. [Training and strict thresholds](../results/4090-c161-ordered-acoustic-parts-fit.json) · [Conversation thresholds](../results/4090-c161-ordered-acoustic-parts-conversation-calibration.json)

| Development outcome with C161 active | C161 | Fixed spacing | Variable spacing | Both heads at the same time |
| --- | ---: | ---: | ---: | ---: |
| Normal “Hey Nyx” before the command, 12 trials | 0 | 5 | 11 | 4 |
| “Hey Nicks” before the command, 12 trials | 0 | 5 | 11 | 4 |
| Joe target, 72 trials | 0 | 7 | 0 | 0 |
| Joe constructed confusers, 216 trials | 0 | 6 | 0 | 0 |
| Jenny target, 36 trials | 33 | 33 | 33 | 33 |
| Openings on 96 daily sayings | 0 | 1 | 1 | 0 |
| Background events on 7.903 / 10.484 / 9.614 hours | 0 / 0 / 0 | 13 / 4 / 4 | 16 / 12 / 1 | 2 / 0 / 0 |

All three combined routes retained the existing “six cups” false wake on its shifted input. The variable-spacing arm also triggered on one rendition of “Thank you, that sounds good”. These results use historical C161 and official Willow events on identical PCM. The time comparisons use the first request rounded to a 256-sample panel read. The previous recordings remain development data. [913 speech records](../results/4090-c161-ordered-acoustic-parts-conversation-speech.json) · [66 full background recordings](../results/4090-c161-ordered-acoustic-parts-conversation-background.json)

## Received first commands and verification on P4 QEMU

Four synthetic voices each spoke the same three commands after Nyx or Willow. Each distinct microphone-open sample on a given recording went through P4 QEMU capture, Opus and the sender. The official Willow opening times came from its native classic ESP32 QEMU detector. The table counts successful whole commands among all 12 attempts for each wake word. The two numbers in each ASR column are Qwen3-ASR and Whisper turbo. The delayed sender used the existing 500/1500 ms transport schedule.

| Route | Opened | Opened before command | Whole command, immediate | Whole command, delayed |
| --- | ---: | ---: | ---: | ---: |
| Frozen C161 | 5 | 0 | 4 / 5 | 4 / 5 |
| C161 plus fixed spacing | 9 | 5 | 7 / 9 | 7 / 7 |
| C161 plus variable spacing | 11 | 11 | 10 / 11 | 10 / 10 |
| C161 plus both parts heads in agreement | 8 | 4 | 7 / 8 | 7 / 7 |
| Official Hey Willow | 8 | 2 | 8 / 8 | 8 / 8 |

The 27 distinct replay paths produced 81 successful UARTs with zero watchdogs. An anonymous download matched all 81 member hashes and archive SHA256 `2d0cca12f639e9548bb2cbbe60ed7b52601d8f2c4fab1b1380fe5394e9d4de5d`. [Per-command transcripts and timing](../results/4090-c161-ordered-acoustic-parts-first-commands.json) · [Download capture, Opus and sender UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-ordered-acoustic-parts-first-commands-uart.zip)

The research P4 guest executed frozen C161 and both integer parts heads on its own streaming frontend. Across 48 unmarked connected utterances, two daily false-wake cases and the shifted “six cups” case, all 9,642 complete live frames agreed with the host for both scores. Every threshold crossing after cooldown and every C161 event agreed to the sample. The diagnostic guest emits UART scores after restoring interrupts. An earlier instrumented build aborted when it tried to print inside the masked detector call. The final 51 logs passed and an anonymous download matched their member hashes and archive SHA256 `a24b165c1e8abbdcec6b74e25d0833d588dc3fb321db673bbab3641d679c7d07`. Board acoustics, minimum internal heap and real receiver audio still require the Mac board session. [Per-input integer comparisons](../results/4090-c161-ordered-acoustic-parts-p4.json) · [Download P4 score UARTs](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/4090-c161-ordered-acoustic-parts-p4.zip)

Both GPU recognizers transcribed 48 distinct five-second neighborhoods around the 21 fixed-spacing, 29 variable-spacing and two simultaneous events. In one simultaneous case both recognizers heard “photos” and “exactly.” In the other both heard “canker sores and cold sores ... the same thing.” These events occur in conversational material. ASR describes the surrounding audio and leaves the precise triggering sound unresolved. [Event times and both transcripts](../results/4090-c161-ordered-acoustic-parts-background-context.json)
