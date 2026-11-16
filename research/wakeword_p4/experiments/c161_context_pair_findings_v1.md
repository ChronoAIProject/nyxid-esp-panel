# Paired continuous-history training: initial fixed-profile results

Both original-initialization RTX4090 arms completed20000 updates and exited0. All reported artifact hashes verified. Silence training1557.083s; speech1637.348s. Each final checkpoint was exported once using the unchanged original calibration representative batch; both60952-byte INT8 exports passed static/runtime ABI checks. Threshold .91796875, smoothing5, refractory67 and half-threshold rearm remain fixed. C161 remains baseline.

Final export SHA256:
- silence: `7d2efe1089d819efe2a81c717af3f4557134c4c029a8e8c51861003c5c0159e7`
- speech: `1456b1acda0462cb4072107136ded55a660ca7ea5bfde9c12cd1ea3abdf67ffc`

Builtin36, daily96 and both continuous-session conditions completed exit0. Original reports/raw UINT8 traces archived with member hashes verified; event steps recomputed from saved UINT8. Existing consumed development inputs only; no threshold selection or acceptance.

|Diagnostic|Silence control|Speech history|
|---|---:|---:|
|Builtin isolated Nyx|11/12|10/12|
|Builtin isolated Willow|0/12|0/12|
|Builtin isolated everyday negatives|0/12|0/12|
|Continuous .5s Nyx containers|4/12|12/12|
|Continuous2s Nyx containers|11/12|11/12|
|Daily conversation|0/24|0/24|
|Daily household|0/24|0/24|
|Daily next/mix|1/24|1/24|
|Daily name/address|0/24|0/24|

Both daily events are index69, “We need six cups of water.” seed726, the previously ASR-reviewed consumed clip. The common-language conflict remains unresolved. Continuous counts attribute event block-return to containers, not phonetic truth or independent per-utterance accuracy. Different session conditions reuse audio; no additive exposure/FAR claim. This initial paired result supports a short-interval benefit of speech-history training, while isolated target coverage is lower than the matched silence arm. It does not prove uniform improvement, first-command PCM retention, real-human/P4 quality, or qualification.

Full Melo474 replay is in progress; do not promote either arm before checking British bare regression and other coverage/negative groups. Legacy/Jenny/Joe/Mac comparisons and practical P4 feedback remain outstanding. Preserve both arms and original C161/bc01 evidence regardless of outcome.
