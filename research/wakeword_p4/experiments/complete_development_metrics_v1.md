# Complete development metrics v1

This additive helper scores the complete-tail collector's original UINT8 outputs.
Historical metrics/results remain unchanged. It imports the existing 138-point
grid, integer five-score event policy, rearm/refractory, sample clocks, Poisson
FAR upper bound, eligibility and selector from `hey_development_metrics_v1`.

## Caller contract

Before calling, authenticate the frozen material manifest, collector execution,
model/training-completion evidence and all trace bytes. Load `expected_rows`
from that authenticated material manifest, independently of collector receipts.
Verify the development roles against the execution/selection protocol. This
helper does not authorize scoring, threshold selection or model promotion.

```python
from complete_development_metrics_v1 import (
    GRID, synthetic_metrics, background_metrics, eligible, choose,
)

# Each clip is (original collector receipt record, original npz['raw']).
# Each expected list is the complete frozen manifest['records'], in order.
synthetic = synthetic_metrics(
    synthetic_clips, 'hey_nyx', threshold,
    expected_rows=synthetic_manifest['records'],
    standalone_carriers=synthetic_manifest['standalone_carriers'],
    details=True,
)
background = background_metrics(
    background_clips, threshold, expected_rows=background_manifest['records'],
)
point = dict(
    threshold=threshold, synthetic=synthetic,
    background_upper95=background['upper95'],
    eligible=eligible(synthetic, background['upper95']),
)
# Run every GRID point. choose(points) preserves the existing selector/null.
```

The authenticated Jenny manifest at GPU `4e08d44` uses `['bare']`. Its 396 rows
yield, per custom target, positive36, standalone9, command27, common324 and
cross-target36. All other literal target aliases, including Hey Muse, remain
common negatives. Denominators are derived from the full frozen rows rather
than changing or truncating that manifest to satisfy the historical360 assert.
Every declared condition needs a nonempty target cell; standalone, commands,
common and cross-target cells must also be nonempty.

Rows must match exactly and have unique `(source_id, variant)` keys. Raw vectors
must be one-dimensional UINT8 with `ceil(real_frontend_frames/3)` steps, where
`real_frontend_frames=max(0, ceil(original_samples/160)-2)`. A missing final
partial group or extra step is rejected. Background exposure is the sum of
original PCM samples, without excluding lead, tail, silence or final padding.

The frozen manifest distinguishes dry prefix/container ends from full-RIR
prefix/container ends. The GPU manifest generator already preserves that
distinction; do not relabel the renderer's RIR-expanded field as a dry boundary.
Detailed events preserve the original one-based raw step, clamped sample end,
unclamped support `(3*step+2)*160` and signed offsets to all four boundaries.
These are generated boundaries, not measured human phoneme endpoints.

## Constructed regression

```sh
python3 research/wakeword_p4/experiments/review_complete_development_metrics_v1.py \
  --gpu-ref 3eff762 \
  --output /tmp/complete-metrics-review.json
```

The review reads exact policy/legacy source from the immutable GPU commit into
a temporary directory; it does not need to install stale local trainer copies.
NumPy and SciPy are required. The output path must be new.

The saved Mac report is
`research/wakeword_p4/results/mac-complete-development-metrics-review-v1.json`.
Constructed396-cell tests cover both targets, explicit bare, added Muse false
activation, early events, distinct dry/RIR recall, signed offsets and a final
step104 event clamped from50240 to50000 samples. Thirteen malformed cases are
rejected. Both targets across all138 points match every shared historical360
metric/eligibility field; detailed event records and an independent slice-based
event oracle match. Complete background exposure, FAR and selector/null checks
pass. No model, frontend, audio, acceptance set or P4 inference was executed;
this result does not establish recognition performance.
