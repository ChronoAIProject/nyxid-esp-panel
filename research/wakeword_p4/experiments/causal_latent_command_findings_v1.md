# Existing complete-command latent support: fixed experiment result

This experiment improves consumed Jenny command hits while increasing nearword false wakes; it is not an overall replacement candidate. Frozen v30 is retained. No threshold, calibration, checkpoint, seed or recipe rescue was performed.

Original training session83064/PID2341885 exited0 after20000 fixed updates on RTX4090,1670.689794918988s. Final weights SHA25684080b3341beceee64647199bc8539243ead0cfd5fd8b08574175cef05e38eaa. Export session8805 exited0; original INT8 is60952bytes, SHA256e8e083687126d9076b8e9c6b590c447d6d66b212fdeb61187fc9f6806370c3a3. Basic interface/reset/function checks passed, with one functional replay; no FP32 or numerical consistency comparison gate.

The new supervision uses existing training-only Piper/Kokoro complete commands, selecting the best eligible five-output mean within generated-container support. Original scalar architecture, initialization, optimizer, source balance and other strong supervision remain fixed. No new TTS, human recording, cropped evaluation waveforms or development-set training was used. Loss reduction does not establish wakeword localization and this experiment does not isolate a causal explanation for its behavior.

All three original collection sessions exited0: legacy88076/372 inputs, Jenny84657/408 inputs, Joe46251/360 inputs. Each uses the fixed0.91796875 threshold, smoothing5, continuous81600sample zero-PCM prime and full original waveform. Baseline c161 traces were reused without repeated inference.

| Consumed development metric | c161 | Latent command |
|---|---:|---:|
| Legacy target hits |35/36|34/36|
| Legacy command hits |26/27|26/27|
| Legacy common-negative activated clips |4/288|3/288|
| Jenny target hits |33/36|35/36|
| Jenny command hits |24/27|26/27|
| Jenny common-negative activated clips |44/324|52/324|
| Joe target hits |0/72|0/72|

Jenny false clips are Nick1/Next13/Mix25/Muse13. Paired legacy events lose1 target hit, remove2 false clips and introduce1 false clip. Paired Jenny events gain2 target hits, remove3 false clips and introduce11 false clips, with41 retained false clips. These are correlated consumed-development observations, not fresh acceptance.

Legacy generated-prefix-complete hits are33/36 (commands26/27), with1 before-only target clip; Jenny26/36 (commands17/27), with9 before-only command clips. These generated metadata boundaries are descriptive and are not human phonetic truth or an automatic rejection gate. Command signed median latency among detected clips is149.625ms legacy and51.40625ms Jenny, against c161153ms and66.40625ms, respectively; changed hit populations limit comparisons. The hypothesis that weak complete-command support would improve discrimination is not supported by the increased Jenny false wakes; the results do not prove or rule out a command-tail shortcut.

Both existing continuous backgrounds had0 events over10.483944826388889h and9.614419583333333h, respectively (one-sided95% Poisson upper bounds0.2857447576425148/h and0.31158742840255416/h). Reused exposures are not independent or additive and do not establish new qualification.

Lossless original reports and raw UINT8 traces for all1140 inputs are recorded in the three `4090-causal-latent-command-*-evidence-manifest-v1.json` files, totaling491442bytes. Event comparisons are in `4090-causal-latent-command-behavior-comparison-v1.json` and `4090-causal-latent-command-event-delta-v1.json`. This is a completed fixed experiment, not completion of the wider independent qualification and matched Hi ESP research goal. c161 supplied to Mac remains an explicitly limited reversible development-board trial, independent of this rejected replacement.
