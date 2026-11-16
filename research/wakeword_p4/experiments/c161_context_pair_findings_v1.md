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

## Full consumed Melo474 comparison

Both final models completed all474 inputs with actual replay exit0; saved-trace scoring also exited0. C161 saved ledgers and every compared trace hash were authenticated. Raw new traces/ledgers/result archived and every member size/hash verified. No threshold selection. EN-BR denotes British.

|Melo target|C161|Silence control|Speech history|
|---|---:|---:|---:|
|US bare|9/9|9/9|9/9|
|US command|24/27|25/27|25/27|
|British bare|7/9|3/9|4/9|
|British command|14/27|16/27|18/27|
|All target|54/72|53/72|56/72|

Speech history loses three British bare targets relative to C161, with no gains in that subgroup. British command gains6/loses2, US command gains3/loses2. These paired changes show why a higher aggregate56/72 cannot establish uniform improvement or erase the material standalone British regression. It is not enough to promote the model.

British official reciprocal containers trigger0/36,8/36,6/36 respectively; US reciprocal is0/36 for all. Constructed nearwords: US38/144,46/144,39/144; British13/144,7/144,7/144. These nearword counts remain diagnostic, not zero-required rejection criteria. All three models emit zero events on the same reused7.902862222h background; this is not newly acquired or additive exposure. The separate daily “six cups” conflict remains for both trained arms.

Decision: retain C161 baseline. Preserve the paired short-interval benefit as evidence for continuous-history training, but do not swap either new model into production/P4 baseline. Further consumed Legacy/Jenny/Joe/Mac coverage comparisons and Mac/P4 feedback remain useful to establish scope. No additional training or threshold rescue is justified by merely wanting the current candidate to pass.

## Legacy consumed coverage

Complete372-input double-model replay and saved-trace comparison exited0; C161 report and compared trace hashes were authenticated. Both arms target36/36 and command27/27 versus C16135/36 and26/27, respectively (gain1/loss0). Target standalone9/9 for all. Other synthetic containers with events:7 for each new arm versus4 C161, including constructed-nearword diagnostics. All models have zero events on the identical reused10.483944826h background; this is nonadditive exposure. New744 traces plus ledgers/reports archived with member sizes/SHA verified. Jenny and Joe are pending; Legacy results do not erase British bare regression or daily six-cups conflict. C161 remains baseline.

## Jenny, Joe and Mac synthetic witnesses

Jenny408 and Joe360 complete deployment-byte replays exited0. Jenny scoring exited0. Joe initial scoring failed because its preserved source metadata has no `carrier`; no inference was repeated. The repaired scorer uses `whole-utterance-unclassified` instead of inventing a bare/command label. A subsequent Git-path readiness check also failed before scoring; the path check was corrected, then saved-trace scoring exited0. Both failure logs/hashes and the repair receipt are preserved. Source labels, raw bytes, threshold and events remain unchanged.

|Target diagnostic|C161|Silence control|Speech history|
|---|---:|---:|---:|
|Jenny all|33/36|32/36|34/36|
|Jenny bare|9/9|8/9|9/9|
|Jenny command|24/27|24/27|25/27|
|Joe whole utterance, unclassified|0/72|0/72|0/72|
|Mac synthetic Nyx|1/4 (prior saved evidence)|2/4|2/4|

Speech history gains one Jenny command without losing targets relative to C161; silence loses one Jenny bare without gains. Jenny other synthetic containers with events are44/360,65/360,60/360 respectively (includes constructed negatives and reciprocal targets). These are diagnostic counts, not common-language frequency or a zero-required gate. Jenny reused9.614419583h background has zero events for all, nonadditive to previous evaluations. Joe negatives0/288 for all, with no background exposure denominator.

Both Mac12 replays exited0: Samantha Nyx2/2, Daniel Nyx0/2; other targets0/8 for each arm. These are the existing consumed synthetic PCM witnesses on the host, not fresh voices or ESP32-P4 execution. Coverage remains voice-dependent; neither model resolves Joe or Daniel failures. Jenny/Joe complete original traces and ledgers were archived with member sizes/hashes verified; Mac24 traces similarly archived and saved event steps recomputed. C161 remains baseline. Next mechanism work should address remaining pronunciation/source gaps without training on these consumed development voices; Mac/P4 operational feedback is still outstanding.
