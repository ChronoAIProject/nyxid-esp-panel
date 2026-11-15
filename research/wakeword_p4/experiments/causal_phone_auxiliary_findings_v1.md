# Pronunciation-codepoint auxiliary v1 findings

The original RTX 4090 training session12500 exited0 after20000 updates in21068.812981168994 seconds. Final weights SHA256 `4fc854aa882a57efaaf6ce76f6d9f28dca04355bd8fa39e8ff469bc7f6d0f6c1`. The once-only binary INT8 export session45541 exited0:60952 bytes, SHA256 `d94d9e9a96135f59ce6c12ecf85a59fbcdae877d9c421b3cd0829eee9cc4ddb5`. The auxiliary head is training provenance only and is discarded for deployment.

All three original complete-waveform collectors exited0 (legacy48472, Jenny33089, Joe33734). They cover1140 inputs at fixed threshold0.91796875/smoothing5, continuous81600-zero-PCM prime with model/frontend/decoder state retained across the boundary. No baseline was rerun; existing c161 source-balanced results were reused.

| Consumed development group | c161 binary | Phone auxiliary binary |
| --- | --- | --- |
| Legacy target / connected command | 35/36;26/27 | 27/36;18/27 |
| Legacy common negative activated | 4/288 | 1/288 |
| Jenny target / connected command | 33/36;24/27 | 26/36;19/27 |
| Jenny scorer common bucket activated | 44/324 | 39/324 |
| Jenny Next / Mix / Muse false wakes | 8 /25 /11 | 14 /20 /5 |
| Joe target | 0/72 | 0/72 |

Jenny324 common records must not be conflated with the historical288 nearword denominator. Jenny has5 target clips with events only before the generated wake-prefix reference (baseline8); these synthetic reference positions are descriptive, not phonetic truth or an automatic failure gate. Legacy command median signed latency increased153 to202.875ms; Jenny increased66.40625 to86.25ms. These medians describe different hit subsets, not paired latency improvement.

Both reused backgrounds remain zero events:10.483944826388889h (one-sided95% FAR upper0.2857447576425148/h),9.614419583333333h (0.31158742840255416/h). They are consumed, reused, nonadditive exposure. Cross-target/pre-source/priming events are zero. Joe common negative0/216 and cross-target0/72 do not compensate for missing every Joe positive.

This intervention does not support an upgrade. Lower aggregate negative activation accompanies substantially lower target/command coverage, persistent Joe blindness and increased Next confusion. It also fails to improve the earlier character auxiliary target coverage (legacy30/36,Jenny29/36) or common false activation(0/288,31/324). This does not identify the causal mechanism or prove that pronunciation supervision cannot work generally. Do not rescue this run by changing coefficient, threshold, checkpoint, seed, calibration, decoder or head. Preserve frozen v30.

All1140 complete UINT8 traces and original report/point JSON bytes are preserved losslessly through the three evidence manifests, totaling509699 published bytes, without duplicating PCM/features. The comparison report uses saved original events only. These are consumed development behavior screens, not independent threshold selection, final qualification or proof of Hi ESP parity. The full research goal remains incomplete.

Paired saved-event analysis (`4090-causal-phone-event-delta-v1.json`, no model calls) separates net counts from changed cases. Legacy loses8 previously hit targets, all connected light commands, with no gained target; it removes4 false clips but adds1 new false clip. Jenny loses8 targets and gains1; target losses include two bare variant2 inputs and six command inputs, including quiet timer input. Jenny removes15 false clips, retains29 and adds10 (8 Next,2 Mix). The new slow bare Next false case occurs in quiet and noise, so the regression is not exclusive to RIR. These source-correlated development cases describe behavior and do not prove a cause or authorize replay-driven tuning.
