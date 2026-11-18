# Component float context findings

Frozen diagnostic, CUDA4090, zero optimizer updates. Inference session98663 actual exit0; independent review actual exit0, all7 archive members verified. Fixed235/256 smoothing5, all six complete consumed contexts, original priming and native clocks.

| Arm | Stream | Float targets | INT8 targets |
|---|---|---:|---:|
| positive_only | builtin | 7/12 | 7/12 |
| positive_only | melo-US | 9/12 | 9/12 |
| positive_only | melo-BR | 1/12 | 1/12 |
| negative_only | builtin | 9/12 | 9/12 |
| negative_only | melo-US | 11/12 | 11/12 |
| negative_only | melo-BR | 1/12 | 1/12 |

Both component BR regressions already occur in float: both retain only1/12 targets versus control6/12. Quantization can change individual decisions, but is not established as the main remedy. Compare prior authenticated control float/INT8 contexts (builtin8/8, US12/11, BR6/6); do not pool different utterances or attribute these results to streaming history contamination. No candidate promotion, threshold/calibration search or P4 qualification.
