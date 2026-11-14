# Original event ledger read on Mac

The original event packet was delivered at GPU `4701ce39`. Mac anonymously
downloaded 527610 bytes, SHA256
`69308a8be5140770abaa31e214821e3296702bd9c345eb39c10efc52c5b2fd78`.
The intake checks the exact 1249-member archive, the published manifest, all
1243 original private payloads against the original plan/metrics, and all five
original receipts against `ca79386`. No model, score or audio is accessed.

After materialization, the unchanged reader at `9162bdb` ran in its isolated
worktree. Original run session99639 exited0. It read all1242 original event NPZs
and recomputed four boundary descriptions at every existing fixed point.
All original prefix metrics and complete-hit time statistics agree exactly.
Report SHA256:
`50b9c121a320d1a91c21a9c7d4c7aa9fdda6f67a9583b98a5a6a19884d0e665e`.

The additional comparison checks GPU `66cb712` directly against those NPZs:
44712 positive rows and178848 per-row boundary descriptions agree exactly;
all Mac complete-boundary aggregates agree too. GPU's first-event statistic
and Mac's first-after-source statistic intentionally have different meanings.

Descriptive ranges of clips with **any** event over all138 historical points:

| Original model | Path | Any event /36 |
| --- | --- | --- |
| Hey Willow float-control | float | 7–8 |
| Hey Willow float-control | UINT8 | 7 |
| Hey Willow integer-QAT | float | 0 |
| Hey Willow integer-QAT | UINT8 | 0 |
| Hey Nyx float-control | float | 33–35 |
| Hey Nyx float-control | UINT8 | 32–35 |
| Hey Nyx integer-QAT | float | 34–36 |
| Hey Nyx integer-QAT | UINT8 | 35–36 |
| v30 | UINT8 | 22–28 |

In particular, most old Willow failures have no event at all. The original
float-control has at most two after-source detections excluded by the dry-prefix
boundary. Boundary exclusion alone cannot explain most of these misses.
These ranges do not match false-alarm operating points and do not establish
that Nyx meets a mature model's recognition quality. They describe old consumed
models/data only, not the current bounded or one-way experiment. Original
selection/rejection decisions are preserved; fresh development stays closed.

Reproduction, using a repository containing both immutable refs and the
previously prepared isolated reader tree:

```sh
python3 experiments/intake_old_regression_event_packet_mac_v1.py \
  --repository-root REPO --isolated-root ISOLATED_9162BDB \
  --archive ORIGINAL_PACKET --report NEW_INTAKE.json
PYTHON ISOLATED_9162BDB/research/wakeword_p4/experiments/describe_old_regression_boundaries_mac_v1.py run \
  --repository-root ISOLATED_9162BDB --published-ref 9162bdb \
  --gpu-results-ref ca79386ea4082036bae9875a2cfdccb8394f8b6d \
  --report NEW_BOUNDARIES.json
PYTHON experiments/compare_old_regression_boundary_ledgers_mac_v1.py \
  --repository-root REPO --isolated-root ISOLATED_9162BDB \
  --mac-report NEW_BOUNDARIES.json --report NEW_COMPARISON.json
```

The reader requires the original NumPy2.4.6 and the exact prepublished plan,
helper, constructed-check report, parent plan and collection/metrics/audit.
It does not recollect inference, recompute trigger events or choose thresholds.
