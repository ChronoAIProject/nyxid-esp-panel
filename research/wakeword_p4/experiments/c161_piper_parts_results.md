# Voice coverage and ordinary speech triggers after broader training

A matched 700-step RTX 4090 run added 640 fit and 160 calibration waveforms from 40 historical Piper training voices to the original 723 fit and 192 calibration waveforms. Both columns below use the same new 1,724-byte head. The original 82 ordinary calibration negatives set the cutoff to −2,217 (“Our best students finished early”). Adding 40 Piper calibration negatives raised it to −1,560 (“Hello”). Original-voice calibration targets accepted were 28/44 and 26/44 respectively. Both cutoffs accepted all 32 Piper calibration targets.

| Development, C161 active | Previous variable head | New, original cutoff rule | New, expanded rule | Both heads, same frame, expanded |
| --- | ---: | ---: | ---: | ---: |
| Ordinary Nyx opened before command / 12 | 11 | 10 | 8 | 6 |
| Ordinary Nicks opened before command / 12 | 11 | 10 | 9 | 6 |
| Joe target opened / 72 | 0 | 8 | 8 | 0 |
| Joe constructed near words opened / 216 | 0 | 0 | 0 | 0 |
| Jenny target opened / 36 | 33 | 33 | 33 | 33 |
| Daily sayings opened / 96 | 1 | 7 | 7 | 0 |
| Events on 7.903 / 10.484 / 9.614 hours | 16 / 12 / 1 | 68 / 284 / 238 | 58 / 253 / 206 | 1 / 0 / 0 |

Six of the new daily openings were explicit addresses to Nick, one was “Hey Mike, good morning.” The earlier “Thank you, that sounds good” example stopped triggering. The shifted “We need six cups of water” still triggered C161 under every policy. Broadening the synthetic training pool reached eight previously missed Joe utterances. It substantially increased triggering in continuous background, even with the expanded calibration rule. Requiring same-frame agreement suppressed most background events and also lost all eight Joe gains. The original-cutoff same-frame results are included in the raw report.

Every GPU score stream for the previous detector matched its saved host SHA256. New GPU integer scores were checked against a separate host implementation on all 913 speech streams and the first complete background. The 66 complete recordings cover the same three previously consumed cohorts. [Frozen protocol](c161_piper_parts_protocol.md) · [Training and both calibrations](../results/4090-c161-piper-parts-fit.json) · [913 speech records](../results/4090-c161-piper-parts-speech.json) · [66 background records](../results/4090-c161-piper-parts-background.json)
