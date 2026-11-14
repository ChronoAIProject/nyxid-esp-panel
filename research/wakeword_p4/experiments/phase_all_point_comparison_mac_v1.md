# Mac review of the completed nominal/phase experiment

Neither final qualifies. This review independently recomputed all 276 published
points, all six group dimensions, the two summaries and all 138 matched deltas
from GPU commit `0616933`. It also inverted the Poisson CDF directly rather than
using the GPU's chi-square library; worst upper-bound difference was
`1.6653345369377348e-16/h`. This is published-data arithmetic, not a second
private trace oracle or model run.

| Condition | FAR-qualified points | Complete / 36 | Bare / 9 | Command / 27 | Joint-qualified points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 116 | 27–35 | 5–9 | 22–26 | 0 |
| Phase | 60 | 33–34 | 7–8 | 26 | 0 |

All ranges cover the entire stated region, without selecting a threshold. The
background exposure is the same complete `10.483944826388889 h` per model.

At identical thresholds, phase complete counts are higher at 91 points, equal
at one and lower at 46. Common-negative event counts are higher at **all 138**;
background counts are higher at 106 and equal at 32. In the 60 points where both
arms meet the background FAR bound, phase-minus-nominal complete ranges from
`-2` to `+3`, common events from `+6` to `+8`, and background events from `0` to
`+1`. Cross-target and pre-source counts remain equal. Better target counts
alone therefore do not establish an improvement in discrimination.

The residual errors are broader than one near word. Across each arm's entire
FAR-qualified region, Hey next activates `3–6/36` nominal and `6–7/36` phase;
Hey Muse activates `1/36` nominal and `5–6/36` phase; Hey Nick activates `0/36`
nominal and `0–1/36` phase. Common bare carriers activate `1/72` nominal and
`5–6/72` phase; common command carriers activate `3–6/216` nominal and
`6–8/216` phase. These are correlated, already-consumed Semaine examples, not
independent speakers. They neither establish phonetic causes nor prove errors
occur while the named word is being spoken.

The earlier crop/stream numerical disagreement remains real, and its numerical
correction remains verified. The completed matched intervention shows that this
correction alone does not yield a qualified wake model. Preserve both results;
do not rescue these finals with new thresholds, extra training, checkpoints,
new exposure or P4 integration.

The next useful diagnostic is to locate the retained false events in the actual
full-stream event records, relative to the original generated-support/carrier
metadata. The previously requested four original paired JSONLs are still
missing from this Mac intake; its authenticated reader is already prepared.
Do not rerun inference to obtain them. Generated support is not phonetic truth.
Before another fit, the GPU side should state a distinct hypothesis grounded
in the retained failures and check it against prior rejected temporal/ranking
experiments. This report does not preselect that hypothesis or launch training.

Fresh development and independent acceptance remain closed. Any future recipe
that survives diagnosis must still supply the same-method Hey Willow pairing,
matched baseline/FAR validation and full P4 resource/audio lifecycle evidence.
The predeclared complete FP32 route remains available; all historical INT8
failures and all requirements for an INT8 implementation remain intact.

Reproduce from the repository root:

```sh
python3 research/wakeword_p4/experiments/compare_phase_all_points_mac_v1.py \
  --gpu-ref 0616933 \
  --output /tmp/mac-phase-all-point-comparison-v1.json
```

The output path must be new. The committed result is
`research/wakeword_p4/results/mac-phase-all-point-comparison-v1.json`.
No audio, model inference, hardware modification or new source data is used.
