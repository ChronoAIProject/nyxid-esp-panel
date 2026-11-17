# Paired-carrier context regressions precede quantization

The six-stream prelocked GPU diagnostic (3e3fe9d0) completed with actual exit 0 in session 31629. Independent saved-output review also exited 0, checked seven archive members and decoded float and integer events with a separate cumulative-sum implementation. No optimization, synthesis, export, calibration or threshold selection occurred.

| Complete consumed context | Control float / INT8 | Paired float / INT8 |
| --- | ---: | ---: |
| Builtin targets | 8 / 8 of 12 | 10 / 9 of 12 |
| US targets | 12 / 11 of 12 | 9 / 9 of 12 |
| BR targets | 6 / 6 of 12 | 3 / 3 of 12 |
| US ordinary names | 0 / 0 of 12 | 1 / 1 of 12 |

The US and BR context coverage regressions already exist in the frozen float model. The paired US “Hey Max, can you carry the other bag upstairs?” event also occurs in float. Quantization is not required to produce these failures. Some other float/INT8 differences remain (notably builtin); this does not establish universal quantization equivalence or isolate the training intervention's causal component.

BR lost text indices 2, 6, 8, 9, 10 have control→paired maximum five-score float means .9581→.0106, .9800→.8577, .9862→.8817, .9967→.4652, .9641→.4617. The two gained indices 1 and 4 rise .7387→.9719 and .4847→.9457. This is a substantial redistribution of responses, not solely integer rounding at the event threshold. A score maximum alone is not an event; independent event decoding and complete clip-container outcomes are saved separately.

Both arms load their unchanged final-20000 checkpoint tensors exactly and pass full-versus-three-frame state/output parity on the fixed disposable probe. Each context uses identical original PCM, original frontend, continuous 81,600-sample zero PCM prime, no internal boundary reset, original 480-sample return clock and complete-stream finish. Three-thousand-feature-frame float chunks carry all model caches. The archived raw traces preserve float probabilities, saved UINT8 codes and clocks; original results bind input and checkpoint hashes through the published plan.

This bounded diagnostic covers all three existing context streams for both arms. It does not explain Joe's isolated-clip gains, daily isolated-name results, human pronunciation or P4 behavior. These consumed clips and correlated variants are not new independent acceptance or background exposure. Keep C161 and the existing Mac handoff. Do not launch QAT, recalibration or a quota/threshold sweep as the proposed remedy for these demonstrated pre-quantization context failures. Further training requires a distinct training-only hypothesis and a prelocked comparison rather than another rendering of this failed recipe.
