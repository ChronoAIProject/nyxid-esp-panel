# Matched long-window cross-engine experiment: coverage gain, still unqualified

Both locked5000update RTX4090 arms completed. Control resumed3501..5000 from the saved3500model/Adam/all RNG after an infrastructure restart; original interrupted outputs and complete recovery history remain. Pooled completed original5000updates in646.45s. Final-only single builtinINT8 exports are115408bytes each, control44782a4b… and pooleddeafd3c6…. Full948actual recording streams genuinely exited0 (control729.09s, pooled860.00s). Every original trace SHA verified; complete records preserved losslessgzip. No checkpoint/threshold/window/calibration rescue or fresh qualification data used.

| Fixed .91796875, smoothing5 | SpeechT5-only control | Engine-balanced pooled |
|---|---:|---:|
| Complete targets /72 |3|35|
| Bare targets /18 |2|9|
| Commands /54 |1|26|
| EN-US bare /9 |2|9|
| EN-US command /27 |0|19|
| EN-BR bare /9 |0|0|
| EN-BR command /27 |1|7|
| Common confusion /288 |0|9|
| Cross Hi ESP /72 |0|0|
| Prime events |0|0|
| Whole background events /7.902862222h |0|0|

Both background one-sided95% Poisson upper bounds are .3790692776004911/h. These are the SAME reused background recordings, not additive exposure or independent acceptance. Both use44800pastPCM/280MFCC,480cadence,continuous81600zero prime without transition reset, originalfiveUINT8/67refractory/halfthresholdrearm; allprime/finish outcomes retained. CUDA computes bounded saved-buffer MFCC and CPU executes actual exportedINT8 bytes singlethread. No neural CPU/GPU equality admission or P4 frontend equivalence claim.

Paired events retain all3 controlhits and add32targets; add8Next falseclips and1Mix falseclip, noNick/Muse/crossfalse. Pooled Nearword false carrier breakdown:Nextbare5/timer3,Mixtime1. Pooled target carriersbare9/light8/time10/timer8 outof18each. Voices/variants are correlated generated conditions, not population samples. These event changes do not prove a unique acoustic cause or missing phoneme.

Saved execution audit checks all5000batches/arm: commonbackground/ambient schedules identical; allinitial tensors identical; fullsource+RIR preserved; control480000SpeechT5classdraws, pooled120000perengine acrossSpeechT5/Piper/Kokoro/Qwen. Endpoint group identity/class contracts pass. Cross-engine source/task support at the SAME window improved consumed recall markedly, but confidence/calibration remain recipe-dependent and this is not a universal source-only or pretraining result. Against the prior shortwindow2/72run, the longwindowcontrol3/72does not suggest window length alone solved the problem.

The pooled result is still inadequate:37/72misses,0/9EN-BRbare and9nearword falseclips. It does not establish HiESP parity. The existing fixed official reference on these consumed inputs has33/72targets,18/54commands and0/288commonfalse; distinct targets and detector event functions limit causal comparisons. A two-hit aggregate advantage is not practical dominance and gives no permission to open sealed acceptance. No threshold selected from these results.

Deployment remains unresolved:approximately20.1Mconv+denseMAC/call at30ms (~670M/s) plusexternalfloatMFCC, not an incremental frontend/encoder; P4arena/whole-systemRAM/realtime notmeasured. No firmware handoff or suggestion to replace Macc161. Originalv30 bytes/settings stay frozen. All old failures remain; Mike, existingfinalsets, MeloIndia/Australia and reservedbackgrounds unopened.

All generated-boundary signed delays remain in original event receipts, not humanphonetic endpoints. Hit-conditioned medians must not be compared as if the hit sets matched. The broad objective remains incomplete. Next decision should distinguish remaining source/voice coverage, complete-command versus bare support and representation/task generalization using existing training-only evidence before another run; do not blindly extend budget, add samevoices or sweep thresholds to rescue this run.

Authoritative reports:4090-matchbox-long-{control,pooled}-consumed-melo-original-v1.json; fullpaired summary4090-matchbox-long-behavior-comparison-v1.json; executionaudit4090-matchbox-long-execution-audit-v1.json. Original reports remain unchanged. Public evidence packaging can preserve model/training states/rawUINT8traces/metadata without duplicating PCM, with anonymous archive/member hash verification; no new qualification implied.
