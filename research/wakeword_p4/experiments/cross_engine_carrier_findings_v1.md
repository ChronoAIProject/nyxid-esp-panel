# Cross-engine carrier trial

Keep C161 as the deployment baseline. The fixed cross-engine appended-pair recipe improves several consumed synthetic voices and connected contexts, but loses important BR bare-wake coverage. This is a mixed research candidate, not a replacement qualification or a reason to reject all cross-engine training.

Protocol db33e609: original seed726/128-row sampler unchanged; append four positive/Nick-negative pairs, engine uniformly Piper/Kokoro, conditioning uniformly within engine, then 27 carrier/speed/render cells uniformly. Adam1e-3, 20000 final-only updates, loss sum/128. Original training exit0; independent reconstruction of all20000 batches/2720000 descriptors exit0. Initial/final finite checks and optimizer step checks pass; optimizer arithmetic was not replayed. Historical control/additive arms are not equal-compute/source-support causal controls.

Final INT8:64800bytes, SHA256 d0012b8e9eb4e596231cdefde7e7a340c08194b5cb57c393763e3699b8d31c08. One final export with original calibration; export exit0. All17 evaluation stages and independent1787archive-member/three-context completeness review exit0. Original evidence published93d56b57. Fixed235/256 threshold, smoothing5; no heldout/threshold/checkpoint rescue.

| Consumed target set | C161 | Cross-engine |
|---|---:|---:|
| Legacy /36 |35|36|
| Jenny /36 |33|35|
| Joe /72 |0|17|
| Melo US /36 |33|33|
| Melo BR /36 |21|16|
| Mac synthetic /4 |1|2|

BR bare falls7/9→1/9. BR light4→5/9, time7→9/9, timer3→1/9. Aggregate gains cannot erase bare/timer losses. Mac Daniel remains0/2 and Samantha2/2. These are consumed synthetic diagnostics, not independent acceptance or human-generalization evidence.

Complete connected contexts (12targets each), C161→cross-engine: builtin5→11, US2→12, BR0→3. Historical original Binary40 is8/11/6 respectively: BR context still regresses against that reference. Ordinary groups are zero except one US name-address hit: “Hey Max, can you carry the other bag upstairs?” This common-name collision matters separately from rare constructed nearwords. Daily96clips yield0events; the context collision shows this does not establish universal everyday rejection.

Reused backgrounds: Legacy10.483944826h, Jenny9.614419583h, Melo7.902862222h: both C161/candidate0events in each source. No new background exposure; never pool these hours with earlier runs. No P4 recognition, resource, realtime, first-command or stability qualification follows from host replay.

Next: retain original model and raw context traces for diagnosis/Mac review, and investigate whether the BR bare/context misses precede quantization using a separately locked zero-update float diagnostic. Do not tune thresholds, oversample the failed consumed BR clips or repeat this recipe with a rescued seed/checkpoint. C161 and existing Mac firmware handoffs remain the baseline.

## Locked float context attribution

Plan de778cd6; GPU4090 original inference and independent saved event/archive review both actual exit0, zero updates. All144 context clip decisions match float/INT8 exactly: target builtin11/11, US12/12, BR3/3 out of12 each. US common-name collision persists in both; other ordinary groups remain0. Thus BR continuous-context misses and this name collision precede quantization. This scope does not explain isolated BR bare/timer misses, Joe or background behavior. No calibration/QAT rescue is justified by these context results; original reports and four-member evidence archive are preserved.
