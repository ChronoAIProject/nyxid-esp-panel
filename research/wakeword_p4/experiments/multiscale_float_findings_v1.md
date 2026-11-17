# Complete multiscale float precision findings v1

The main normal-context regression is already present in float; quantization is not the primary explanation or a sufficient remedy. Keep original C161 and existing Mac handoffs; no multiscale promotion.

| Consumed target scope | Control float / INT8 | Multiscale float / INT8 |
|---|---:|---:|
| builtin | 9 / 8 of 12 | 9 / 9 of 12 |
| primed | 36 / 36 of 36 | 33 / 33 of 36 |
| jenny | 33 / 33 of 36 | 34 / 34 of 36 |
| joe | 20 / 17 of 72 | 22 / 24 of 72 |
| melo/EN-US | 33 / 33 of 36 | 34 / 34 of 36 |
| melo/EN-BR | 20 / 20 of 36 | 26 / 27 of 36 |
| mac | 2 / 2 of 4 | 2 / 2 of 4 |
| context/builtin/nyx_connected_command | 8 / 8 of 12 | 8 / 7 of 12 |
| context/melo-US/nyx_connected_command | 12 / 11 of 12 | 12 / 12 of 12 |
| context/melo-BR/nyx_connected_command | 6 / 6 of 12 | 3 / 2 of 12 |

Multiscale US ordinary-name context has1event in both precisions, versus0control. All other ordinary context groups0. This count does not independently identify the float event as the exact same Max sample; saved identity follow-up is required. BR float3/12 versuscontrol6/12 already regresses, with one additional netINT8miss. Float does not restore Legacy33/36 versuscontrol36/36 either. Joe counts22float/24INT8 show quantization effects need not be uniformly harmful. Counts do not prove perclip identity.

Producer session78530 actualexit0; independent review2264 actualexit0; package review48596 actualexit0. Both strict final20000loads passed exact checkpointtensor equality and disposable full/native-step parity. Both1763streams/3961012calls each,3526streams/7922024calls total. Independent reviewer recomputed float/integer events from every saved array and complete inventory; package reviewer bound all3526INT8arrays to authenticated original traces, archived/reopened eachmember with size/SHA checks. Reports4090-multiscale-float-{original,review,package-review}-v1.json and both perarm evidencearchives.

Fixed threshold0.91796875/smoothing5/refractory67/halfrearm; no output-code rounding for float, no optimizer updates, no new data/exposure, no INT8reinference, no checkpoint/calibration/threshold selection. These consumed synthetic findings do not establish human generalization or P4 resources/realtime/stability. No QAT/calibration rescue as main remedy; learned context support remains unresolved. Continue saved eventidentity attribution and independent Mac/P4 engineering evidence before another training proposal.
