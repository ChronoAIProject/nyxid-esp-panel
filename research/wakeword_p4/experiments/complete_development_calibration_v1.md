# Complete development calibration v1

`calibrate_complete_development_v1.py` reads completed collector artifacts and
produces `calibration.json` in that model's original output directory. It never
loads an interpreter, invokes a model or decodes audio. Historical calibrators
and results stay unchanged.

## Integration before the first raw execution lock

The producer must separately admit the completed fixed-final models, numerical
screen and new development under its existing scoring contract. The collector's
`development-raw-collection-only` scope does not grant that admission. This
consumer validates artifact identity and scoring completeness, not the truth of
the producer's training/source-history evidence.

Bind these consumer files in the same execution `files` map before publishing
the lock and before collection:

- `calibrate_complete_development_v1.py`
- `complete_development_metrics_v1.py`
- `hey_development_metrics_v1.py`
- `uint8_event_policy_v1.py`
- `collect_complete_development_v1.py`
- `complete_feature_stream_v1.py`

The existing versions map already includes NumPy and SciPy; this consumer checks
their versions. Extend each execution model with explicit semantic fields; the
collector preserves these fields in its report without needing modification:

```json
{
  "models": {
    "hey_willow": {"target_id": "hey_willow"},
    "hey_nyx": {"target_id": "hey_nyx"},
    "v30": {
      "target_id": "hey_nyx",
      "fixed_reference_thresholds": [0.91796875]
    }
  }
}
```

These fields supplement each original model's `path`, `sha256`, `bytes` and
`output`; they do not replace them. This avoids inferring a target from a model
name, path or byte size. The producer still verifies frozen v30's original
identity. Do not retrofit a changed lock into already collected reports.

```sh
python3 research/wakeword_p4/experiments/calibrate_complete_development_v1.py \
  --execution-lock research/wakeword_p4/results/4090-jenny-complete-development-execution-v1.json \
  --published-ref ACTUAL_IMMUTABLE_EXECUTION_COMMIT \
  --model hey_nyx
```

Expected rows come independently from the frozen material manifests, including
`standalone_carriers` from the synthetic manifest. Both roles must be
`new-development`. A reused-development regression needs its own named protocol;
this selector cannot silently use it as new threshold-selection data.

The reader verifies the published execution, relevant source bytes, original
model/report identity and scope, full JSONL receipts and hashes, exact frozen
rows, trace paths/hashes, reset repeats, UINT8 steps, complete original INT64
sample clocks and tail counts. It uses every complete input and requires at
least eight original background hours. No lead/tail or silence is excluded.

Output retains all138 registered thresholds and any extra fixed reference
thresholds, complete synthetic event records/signed offsets and explicit
denominators, original background samples/hours/events/FAR, all eligible flags,
the unchanged selector/null result, and separately reported reference points.
A reference that already belongs to GRID retains its ordinary GRID eligibility.
The complete-phrase window still includes saved trailing silence through the
whole original input end; dry/RIR command endpoints remain signed diagnostics.
Detailed offsets must be used when assessing latency rather than interpreting
late complete-phrase hits as fast recognition. Existing calibration files are
never overwritten. No result authorizes independent acceptance or promotion.

## Constructed regression

```sh
python3 research/wakeword_p4/experiments/review_complete_development_calibration_v1.py \
  --gpu-ref 3eff762 --output /tmp/complete-calibration-review.json
```

The saved Mac report is
`research/wakeword_p4/results/mac-complete-development-calibration-review-v1.json`.
Thirteen checks cover a constructed396-cell report and fabricated10h zero score
stream, actual FAR arithmetic and selected/null outcomes, a separate reference,
incomplete/changed records, reset/tail/hash failures, a last clock changed even
with updated trace hashes, reused roles, insufficient exposure, source/execution
binding and CLI preservation of existing output. The integration fixture uses
two grid points plus a reference; the companion metric review covers both
targets across all138 points. Publication lookup is mocked, while file hashes,
receipts, NPZs, clocks and scoring run normally. No actual model or source audio,
development, acceptance, frontend or P4 inference is involved.
