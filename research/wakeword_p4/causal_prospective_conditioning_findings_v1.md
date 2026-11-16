# Conditioning transfer decomposition

The prospective development replay decomposes as follows (raw float exact matcher):

| conditioning | target hits / 36 | negative events / 180 |
|---|---:|---:|
| af_alloy | 10 | 2 |
| af_kore | 17 | 25 |
| am_echo | 16 | 2 |
| am_santa | 4 | 0 |

The poor47/144 target result is spread across voices, while `af_kore` is a strong confusion outlier (25/180 negatives). `am_santa` has very low target recall but no negative events. This pattern is inconsistent with a single global threshold explanation and shows that conditioning transfer and class confusion interact. No development threshold was selected from this failed candidate.

A future candidate should make conditioning-disjoint performance a preregistered gate and report per-conditioning results. It needs an explicit invariance/data strategy, not another scalar loss adjustment on the same all-conditioning optimization. The split, records and failure remain available for a new model; no protected final material was opened. Frozen v30 remains unchanged.
