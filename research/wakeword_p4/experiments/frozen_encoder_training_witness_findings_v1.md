# Final training-witness findings

Execution6ca82070 was published before original session71262; actual4090 read-only run exited0. Independent saved-array audit also exited0. All four final10000 models scored the first100 authenticated original batches twice, identical per target,12800 seen draws/model (3200positive/9600negative). No optimizer/gradient or other checkpoint used.

| Target / condition | Positive windows p>=0.5 | Negative windows p>=0.5 | Seen-window AUC | Weighted mean BCE |
|---|---:|---:|---:|---:|
| hey_nyx / full | 1989/3200 | 32/9600 | 0.994271 | 0.346880 |
| hey_nyx / frozen | 57/3200 | 2/9600 | 0.950489 | 0.798131 |
| hey_willow / full | 2470/3200 | 15/9600 | 0.997357 | 0.234053 |
| hey_willow / frozen | 0/3200 | 0/9600 | 0.933420 | 0.864672 |

Frozen positive median probabilities are Nyx0.111644 / Willow0.095828; full medians0.628850 /0.839353. Frozen heads have some score separation (AUC0.950489/0.933420), so do not describe them as mathematically constant or entirely unlearned. They do not establish useful detection in the original consumed recognition protocol. No bias correction or lower threshold is authorized to rescue them.

Willow full has strong ranking and substantial fixed0.5 positive support on these training windows, yet complete0/36 on consumed cross-source waveforms at every FAR-qualified point. Thus its failure cannot be summarized as absence of training-window discrimination. This does not isolate voice, engine, carrier, source quality, timing or overfitting as the cause; training draws and waveform-event decisions differ. Nyx full also fits these sampled windows more strongly than its frozen counterpart, while retaining previously reported common-word errors.

The next evidence needed is source/engine/carrier-stratified training-witness versus consumed-diagnostic support under matched functions, without selecting samples by scores. Existing metadata/source/window census remains authoritative and should be reused, not regenerated. Do not change negative20, budget or learning rate merely because frozen positives are low. Source coverage/generalization is a testable direction, not a proven cause or yet a training recipe.

These100 early historical batches evaluated at final weights are correlated training witnesses, not unique-source full-fit statistics or independent holdout. AUROC is a window ranking summary, not event recall or FAR. Independent auditor verifies stored arrays/reduction/receipt identity; runtime model execution/repeats remain attributed to the original run. Synthetic quality, stream INT8, untouched-development/acceptance and same-protocol Hi ESP/v30 gates all remain outstanding. No threshold selected, no promotion, v30 unchanged.
