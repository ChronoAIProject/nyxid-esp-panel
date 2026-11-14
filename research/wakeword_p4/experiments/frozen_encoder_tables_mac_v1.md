# Actual frozen-encoder table intake and independent arithmetic check

GPU commit `0a94e7f1d78472a79e9513d1e4fb73da2cddcb1e` contains the actual
four-arm comparison introduced at `3f889b8`, after original training/export/
own540/collection/scoring/independent-event jobs terminated. This Mac check
authenticates the committed tables and independently recomputes their arithmetic;
it does not replay private models, H5 histories, saved traces or event arrays.

The existing auditor checked all 552 points, all six group dimensions per point,
276 full/frozen matched deltas, record/summary/execution bindings and the original
grid/material identities. A separate direct Poisson-CDF inversion differs from
the reported 95% upper FAR by at most `2.220446049250313e-16 / hour`.

| Model | FAR points | Negative-only joint points | Complete target range within joint points |
| --- | ---: | ---: | ---: |
| Nyx full | 138 | 30 | 0–14 / 36 |
| Nyx frozen | 138 | 138 | 0–1 / 36 |
| Willow full | 121 | 121 | 0 / 36 |
| Willow frozen | 138 | 138 | 0 / 36 |

"Joint" here only means FAR plus zero common, reciprocal and pre-source events.
It excludes a positive-recall requirement and therefore does not qualify a
silent detector. Neither frozen arm is useful under this experiment. At matched
thresholds, Nyx frozen-minus-full complete recall is -34..0 and common events
-35..0; these tradeoffs are not an upgrade. Willow's positive recall stays zero.
All original points and group regions are retained; no threshold was selected.

The first execution stopped with `KeyError: complete_at_joint_points`: the
prospective old-data parser fixture had assumed a shorter summary field name.
The actual reporter uses `complete_at_all_joint_points`. The Mac checker now
maps that name explicitly and also verifies every declared bare/command range
and minimum FAR. GPU artifacts and diagnostic criteria did not change. The
original failure receipt remains private and is summarized in the public report.

The revised checker was rerun on the real old 552 points and the explicitly
renamed old-data fixture. All 18 corruption cases were rejected, including the
four additional joint/bare/command/minimum-FAR mutations. The 107 Poisson checks
against SciPy still pass. Those fixture results are not new-model evidence.

Two executions of the actual frozen-table audit produce byte-identical full
reports. The 664,513-byte report is retained as lossless gzip, including all
matched deltas and grouped ranges:

- [Summary and full-report hashes](../results/mac-frozen-encoder-diagnostic-table-audit-v1.json)
- [Complete report](../results/mac-frozen-encoder-diagnostic-table-audit-v1.json.gz)
- [Checker regression and rejection results](../results/mac-paired-diagnostic-table-auditor-review-v2.json)

```sh
python3 research/wakeword_p4/experiments/audit_paired_diagnostic_tables_mac_v1.py \
  --gpu-ref 0a94e7f1d78472a79e9513d1e4fb73da2cddcb1e \
  --profile frozen-encoder --output /private/path/new-frozen-table-audit.json
```

No new audio, model call, training, source admission, P4 flash, fresh development
or independent acceptance was performed. The goal remains incomplete. These
failed models do not justify a new P4 adapter or a threshold/budget rescue.
