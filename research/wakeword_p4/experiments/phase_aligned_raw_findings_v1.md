# Actual matched phase-supervision result

Neither nominal nor phase final has a point satisfying the original background FAR upper95<=0.5/h together with zero common-negative, cross-target and pre-source events. No working point was selected; fresh development and acceptance were not opened; frozenv30 remains unchanged.

Both original10000-update runs genuinely completed. Independent training audit passed. Both new exports passed all135original complete waveforms/model (270total) with original1e-4FP32/phase gates, fullstartup/tail, resets/twofresh and clocks. Both then consumed all372reused diagnostic inputs (360synthetic,12wholebackground=10.483944826388889h), twofresh. All744traces and276original fixed-grid points/events independently audited.

| Arm | FAR-qualified points | Complete phrase at all FAR-qualified points | Bare | Command | Joint-qualified points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 116/138 | 27–35/36 | 5–9/9 | 22–26/27 | 0 |
| Phase | 60/138 | 33–34/36 | 7–8/9 | 26/27 | 0 |

Ranges are descriptive, not chosen working points. At identical thresholds, phase complete count is higher91/138, equal1, lower46; common-negative eventcount is higher138/138 (by6..11), and background eventcount is higher106/138, equal32, lower0. Thus improved complete counts at some thresholds come with worse discrimination; no coefficient/threshold/extra training or new exposure is used to rescue this run. Hey next activated clips across each arm's FAR-qualified region remain nominal3..6/36 and phase6..7/36. Minimum background upper95 is .2857447576425148/h for both, but this does not make their joint gate pass.

The earlier same-coordinateCPU phase agreement remains true (max1.25170e-6), but numerical supervision alignment alone did not establish recognition improvement. This experiment does not prove that all phase-aligned training is ineffective, that source/phonetic labels are wrong, or that a different initialization/budget/objective would work. Do not rerun the same matched phase intervention with tuned budgets or checkpoints.

Machine evidence: `4090-phase-aligned-comparison-v1.json` retains every newpoint, matched delta and all original raw/official reference curves plus sourceSHA; `4090-phase-aligned-grouped-diagnostic-v1.json` retains all source/carrier/voice/literal groups. One correlated consumed Semaine source limits interpretation, official native512/raw480 policies differ, and generated boundaries are not human phoneme references. This FP32 diagnostic failure is not an INT8 qualification or independent acceptance result. Further research must use a distinct hypothesis; realwaveINT8/freshdev/independentacceptance remain required for any future promotion.
