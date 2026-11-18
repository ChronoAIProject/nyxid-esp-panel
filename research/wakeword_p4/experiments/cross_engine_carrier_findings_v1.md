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

## Saved cell coverage audit

The exhaustive72target-cell audit preserves PCM identity and all speed/carrier/render cells. BR bare loses six C161 hits with no gains; BR light gains two and loses one, time gains two with no losses, timer loses two with no gains. BR quiet speed1.1 bare is itself a C161-hit/candidate-miss, so this regression is not confined to the noise/RIR render. Do not infer acoustic cause from this association: speeds are separately synthesized and render variants share native speech. US aggregate33/36 conceals two new time hits and two losses (bare/timer). Native source metadata still says unconsumed-threshold-development in places; these records are already consumed diagnostics and cannot be treated as fresh acceptance. No new inference or training was performed by this audit.

## Exhaustive original bare training support

Prelocked dd2c4cb3, RTX4090 float run actual exit0, zero updates; all31644 original broad bare rows in metadata order. Original Binary40→cross-engine Piper positives: valid-source events5934→5808/6048; positive-mask-contained crossing5918→5775/6048. Kokoro180/180 and Qwen216/216 retain crossings. Piper Mix valid events431→244/6048; Nick5→1, Next53→40, Muse8→2. These are training-only cold-cache float fit metrics, not generalization or comparable positive/negative error definitions. The appended recipe therefore changes original bare-source fitting as well as unseen voice outcomes; not solely an unseen BR phenomenon. No proof that added negative labels cause all losses: positive append, optimization and history also differ.

Independent saved-probability event decoder checked every row and aggregate; positive-mask crossing lists were aggregated as reported, not independently regenerated. No new synthesis, threshold selection or protected data. Raw probabilities and per-source records are preserved. A future proposal must preserve original training positive support rather than assuming it is already solved, and cannot simply repeat bare-slot quotas, historical retention losses or gradient projection already rejected by earlier evidence.

## Independent support-window reconstruction and paired transitions

A separate reviewer reconstructs all31644 original temporal masks directly from frame/sample geometry and verifies every saved five-call crossing for both models, plus row identity and raw probability hashes. This closes the earlier positive-mask verification limitation. Piper positive-mask transitions:5722retained,53gained,196lost,77neither (6048rows); net loss143 therefore conceals substantial turnover. Kokoro180 and Qwen216 remain fully retained. Negative rows have no positive mask by definition; their all-neither entries are not negative rejection evidence. This remains training-only support evidence, not proof of the acoustic cause or a license to mine consumed evaluation into training. Earlier independent decoder verifies valid-source negative events separately.
