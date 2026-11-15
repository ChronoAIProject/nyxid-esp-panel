# Character auxiliary v1 findings

The fixed 20000-update RTX 4090 run completed in 18984.523038716 seconds (session 85591 exit 0). The single binary-only INT8 export completed (session 53839 exit 0): 60952 bytes, SHA256 `446f56e7e3bb749b138a29a274d78676dedb06f065a2fb0515975dc72bfacff0`. The training-only character head is discarded.

All three complete waveform collectors exited 0: legacy 61181, Jenny 98052, Joe 85307. Fixed threshold 0.91796875, smoothing 5, continuous 81600-zero-PCM priming and retained startup events/state were used. Existing source-balanced c161 results were reused without baseline reruns.

| Consumed development group | c161 source-balanced | Character auxiliary |
| --- | --- | --- |
| Legacy target / command | 35/36; 26/27 | 30/36; 22/27 |
| Legacy common negative activated | 4/288 | 0/288 |
| Jenny target / command | 33/36; 24/27 | 29/36; 22/27 |
| Jenny scorer common bucket activated | 44/324 | 31/324 |
| Jenny Next / Mix / Muse false wakes | 8 / 25 / 11 | 11 / 18 / 2 |
| Joe target | 0/72 | 0/72 |

Jenny's scorer common bucket has 324 clips; the historical 288 nearword comparison and the additional 36 clips must not be silently conflated. The activated nearword counts above are retained directly from the phrase comparison report. Jenny had 6 target clips whose events were only before the generated wake-prefix reference (c161: 8); this is descriptive synthetic-boundary timing, not automatic failure or phonetic truth. Legacy command median signed latency increased 153 to 217.84375 ms; Jenny increased 66.40625 to 106.5625 ms. These medians cover different hit subsets.

Both reused background streams had zero events: 10.483944826388889 hours (one-sided 95% FAR upper bound 0.2857447576425148/h), and 9.614419583333333 hours (0.31158742840255416/h). Do not add these reused streams as independent exposure. Cross-target, pre-source and priming event counts were zero. Joe common false wakes remained 0/216 and cross-target 0/72.

This hypothesis did not yield an upgrade: lower aggregate nearword activation came with lost target coverage, greater command latency, persistent Joe blindness and increased Next confusion. Do not rescue it by changing coefficient, threshold, checkpoint, seed or calibration. Preserve frozen v30. Native sequence-reference performance does not establish that a training-only character loss transfers sufficient lexical discrimination into this compact binary model.

All 1140 complete UINT8 traces and original reports/point JSON are published losslessly, totaling 518968 bytes across the three evidence manifests, without PCM/features duplication. These are consumed development regressions, not independent acceptance or threshold selection. Independent threshold selection, final INT8 qualification and final official comparison remain outstanding for the full goal.

Paired saved-event delta (`4090-causal-character-event-delta-v1.json`) shows no newly recovered target in either legacy or Jenny: all 5 legacy losses were previously hit (one bare, four light commands), and all 4 Jenny losses are variant2 (two bare, one time, one timer). Jenny's net 44→31 false clips hides 19 removed, 6 new and 25 retained; all six new false clips are Next, including the same slow bare source in quiet, noise and noise+RIR variants. Thus the Next regression is not exclusively an augmentation artifact, and the target losses are not a recovery trade between different missed targets. This saved behavior does not establish the causal mechanism or justify changing a loss coefficient.
