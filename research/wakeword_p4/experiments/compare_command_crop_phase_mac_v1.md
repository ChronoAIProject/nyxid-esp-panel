# Independent published sampled-coordinate comparison

Mac ran `compare_command_crop_phase_mac_v1.py` against immutable GPU commit
`2ddb2585078a3d3c0175bf5511dac11623cde572`, without new inference or audio I/O.
Both execution locks and every available committed source/report binding match;
all 27 shared lock entries bind the same inputs. Private model/feature/trace
bytes and the actual remote runtime devices were not independently replayed.

All 12,510 paired published coordinates, including duplicate draws, retain the
same step, slot, original order, endpoint and saved stream score. Every absolute
error and every published quantile was independently recomputed. Same-CPU
nominal225 versus stream maximum error is 0.999996766799244; phase224past+zero
versus stream maximum is 0.0000012516975402832031, below the unchanged 1e-4
bound. Nine coordinates have final nominal <=0.5 and stream >=0.9. GPU/CPU
phase maximum difference is 0.0023148655891418457; the reports alone do not
establish a unique hardware arithmetic cause.

This supports the numerical distinction at the sampled endpoints divisible by
three. It does not prove training-time loss, all other phases, recognition
causality, training benefit, FAR, INT8 suitability or full-panel acceptance.
The prospective all-slot phase-input matched training must produce and assess
its own final weights; neither historical failures nor thresholds are replaced.

```sh
python3 research/wakeword_p4/experiments/compare_command_crop_phase_mac_v1.py \
  --repository REPO --ref 2ddb2585078a3d3c0175bf5511dac11623cde572 \
  --output NEW_REPORT.json
```

Actual output is `../results/mac-command-crop-phase-comparison-v1.json`.
No fresh development/acceptance data, model invocation or production hardware
was accessed. No threshold or candidate was selected.
