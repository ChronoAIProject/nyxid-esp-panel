# Independent four-arm table audit

`audit_paired_diagnostic_tables_mac_v1.py` prepares a Mac-side check of the new
matched full/frozen encoder results. The existing experiment, scoring policy,
threshold grid, old scripts and qualification requirements remain unchanged.
The new checker has two explicit profiles: historical raw-context tables for
actual saved-data verification, and the forthcoming frozen-encoder tables.

It authenticates committed public summaries, grouping and event-audit bindings,
checks all four unique model names and all 552 threshold points against the
original materials/grid, and reconciles all six group dimensions. It recomputes
FAR/zero-common/zero-cross/zero-pre-source conjunctions and all 276 same-threshold
paired deltas. It preserves every point and all group ranges. A direct Poisson
CDF inversion independently checks the reported one-sided 95% FAR upper bounds;
the checker needs only the Python standard library and Git. Counts above 99 are
supported, including detectors with frequent background events.

This is a committed-table consistency/arithmetic audit. It does not replace the
GPU's full saved-event oracle, execute a model, inspect private H5/trace arrays,
select a threshold, access fresh data or qualify a candidate. In particular,
zero-negative silent detectors remain insufficient. At pinned GPU commit
`1dced9b`, the new frozen-encoder comparison is absent; that profile correctly
returns exit 2 / `not_ready` and creates no output. No new four-model recognition
result has been audited yet.

## Executed verification

The raw-context profile actually processed all 552 existing real points and all
six grouping dimensions at `1dced9b`. It reproduced the earlier failure:
Willow has no FAR-qualified points; Nyx control/context have 93/138 FAR-qualified
points and complete counts 32..35 / 24..34 of 36, with zero joint points in either
arm. Its largest independent FAR-bound difference was 6.67e-16 per hour.

`review_paired_diagnostic_tables_mac_v1.py` additionally renames those old table
labels into an explicitly constructed future-profile parser fixture. Those
552 fixture points are not new weights, traces, training or recognition. All
14 corruptions were rejected: missing/duplicate points, missing models, bad
group counts, an extra grouping dimension, wrong Poisson bound, audit/summary/
record hash mismatches, wrong declared summary/delta/rejection list, changed
original grid and a selected threshold. Independent Poisson inversion was
checked against SciPy at 107 counts from 0 through 1,000,000; maximum relative
error was 2.82e-13. The absolute error at the largest count was 2.70e-8 per hour,
far outside the admission region; actual old-table differences were much smaller.

Results:

- `results/mac-paired-diagnostic-table-auditor-review-v1.json`: full source/input
  bindings, all corruption outcomes and all 107 numerical comparisons.
- `results/mac-paired-diagnostic-table-raw-profile-replay-v1.json.gz`: complete
  actual raw-profile report, including all matched points and group ranges;
  raw 541,171 bytes, SHA256
  `b71ea806bae7c486c0a105c96365ce0134ef3033d380de87285c53ea9d294a4a`.

Reproduce with Python plus NumPy/SciPy for the review command only:

```sh
WAKE_TABLE_REVIEW_DIR=$(mktemp -d)
python3 research/wakeword_p4/experiments/review_paired_diagnostic_tables_mac_v1.py \
  --gpu-ref 1dced9b --output "$WAKE_TABLE_REVIEW_DIR/review.json"
python3 research/wakeword_p4/experiments/audit_paired_diagnostic_tables_mac_v1.py \
  --gpu-ref 1dced9b --profile raw-context \
  --output "$WAKE_TABLE_REVIEW_DIR/raw-profile.json"
```

For genuine new results, use `--profile frozen-encoder` with the commit that
actually contains all required comparison/grouped/audit/execution files. Missing
files return `not_ready`; a parser review or old-profile pass cannot replace them.
No hardware, microphone, speaker, network configuration or original GPU job is
operated by these tools.
