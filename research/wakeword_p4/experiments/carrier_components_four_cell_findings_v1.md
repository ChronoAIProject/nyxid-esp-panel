# Four-cell saved float comparison

All144 clip identities match exactly across control, positive-only, negative-only and joint complete-context records. Source hashes and all per-clip five-window float margins are preserved in4090-carrier-components-four-cell-saved-comparison-v1.json. No new inference, training, synthesis or threshold choice.

Pattern order: control / positive-only / negative-only / joint;1 means a float event inside the target container.

| Stream | Pattern | Target clips |
|---|---|---:|
| builtin | 0000 | 2 |
| builtin | 0010 | 1 |
| builtin | 0110 | 1 |
| builtin | 1000 | 1 |
| builtin | 1011 | 1 |
| builtin | 1110 | 1 |
| builtin | 1111 | 5 |
| melo-US | 1000 | 1 |
| melo-US | 1010 | 1 |
| melo-US | 1011 | 1 |
| melo-US | 1110 | 1 |
| melo-US | 1111 | 8 |
| melo-BR | 0000 | 6 |
| melo-BR | 1000 | 5 |
| melo-BR | 1111 | 1 |

The fixed-seed component contrasts can identify shared and joint-only lost containers, but cannot establish a universal gradient/forgetting mechanism or a reliable replacement recipe. Training supervision and compute differ across128/132/132/136 rows. Compare all targets and ordinary negatives; do not tune a coefficient, voice quota, boundary or checkpoint to these consumed clips. Keep C161.

All five control BR float targets lost by the joint intervention are also lost individually by both components, with exact clip identity agreement. The one surviving BR target is shared across all four models. This is stronger than matching aggregate counts, but does not identify a universal mechanism.
