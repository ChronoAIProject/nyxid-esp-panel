# Complete-container phrase auxiliary findings

Original RTX4090 session99251 exited0 after20000 fixed updates in1455.7116674530116 seconds. Final weights SHA256 b12f3b1642b4babec0738cec2b7e0951b45545430da189d1c3431b01fec34497. Bootstrap failure before model/optimizer construction is retained separately in the launch report. Fixed final-only binary INT8 export session23779 exited0:60952 bytes, SHA256171b28955c6366ac54ed55400b6cc52ffff94c54082bf0731eb59daf3c2fc7b0. Training-only19class auxiliary head is discarded in deployment.

All original full-waveform sessions47587/50460/98118 exited0,1140 complete records, fixed .91796875/smoothing5/continuous81600zeroPCM prime with frontend/model/decoder state retained. Existing c161 baseline events are reused, no baseline rerun.

| Consumed development behavior | c161 binary | Phrase auxiliary |
| --- | --- | --- |
| Legacy target / command |35/36;26/27|31/36;22/27|
| Legacy common bucket false clips |4/288|3/288|
| Jenny target / command |33/36;24/27|33/36;24/27|
| Jenny common bucket false clips |44/324|42/324|
| Jenny Nick / Next / Mix / Muse false clips |0/8/25/11|1/15/21/5|
| Joe target |0/72|0/72|

Jenny common324 is not the historical288 nearword-only denominator. Eight Jenny target clips have events only before the generated wake-prefix reference in both conditions; this is descriptive synthesis timing, not automatic failure or human phonetic truth. Legacy command median signed reference latency153ms unchanged; Jenny66.40625→74.625ms. Medians describe hit subsets and do not establish paired speed changes. Legacy saved per-target false counts include one hi_esp-labelled record and two common_negative records; keep original scorer bucket definitions rather than silently recategorizing their denominators. Dedicated cross-target buckets are zero in all groups, as are pre-source and priming events. Joe common0/216/cross-target0/72 does not compensate for missing every target.

Both reused backgrounds have zero events:10.483944826388889h (one-sided95% FAR upper .2857447576425148/h) and9.614419583333333h (.31158742840255416/h). Reused exposure is nonadditive and not independent qualification.

This clip-level supervision did not produce an upgrade. It avoids the large Jenny coverage loss of character/phone CTC, but still loses four legacy hits relative to c161, increases Next confusion and leaves Joe completely undetected. Slight aggregate false-count reduction is insufficient. These behaviors do not identify a causal mechanism or prove clip supervision cannot work generally. No coefficient/LR/checkpoint/threshold/seed/calibration or head-selection rescue; preserve frozen v30.

All1140 UINT8 traces and original report/point JSON bytes preserved losslessly through three manifests,494556 published bytes without PCM/features duplication. Actual deployment inference uses CPU after GPU training, not CPU training fallback. Training and export checks are not recognition qualification. Full goal remains incomplete: independent threshold selection, final INT8 qualification and official matched final evaluation remain outstanding. No existing final data accessed or candidate promoted.
