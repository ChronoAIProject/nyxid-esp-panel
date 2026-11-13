# Complete official reference calibration v1

`calibrate_complete_official_v1.py` reads the new official Jenny grid; it does
not launch QEMU or change the original9672-stream process, files or results.
Run after the producer admits the new materials and full grid. The new official
execution must be published before inference and bind this reader before any
operating-point selection.

## Freeze before the new official execution

Extend the new `jenny-official-development-execution-v1` plan with
`custom_execution_path`, naming the already bound custom complete execution.
The existing registration code already includes that file and both material
manifests in `files`. Also bind these calibration dependencies:

- `experiments/calibrate_complete_official_v1.py`
- `experiments/hey_development_metrics_v1.py`
- `experiments/uint8_event_policy_v1.py`
- `qemu_hey_reference_v2/calibrate_development.py`
- `qemu_hey_reference_v2/run_development_grid.py`

Paths above are relative to `research/wakeword_p4`. Keep the existing new runner,
original runner, main C source, runtime and full WAV identities bound as well.
NumPy/SciPy versions come from the original custom execution. The driver
independently derives expected rows from its two material manifests and requires
the official plan to contain exactly the same complete PCM rows, with original
source identities and explicit `bare` standalone carriers.

```sh
python3 research/wakeword_p4/experiments/calibrate_complete_official_v1.py \
  --execution-lock research/wakeword_p4/results/4090-jenny-official-development-execution-v1.json \
  --published-ref ACTUAL_IMMUTABLE_EXECUTION_COMMIT
```

The reader uses stdlib `wave` to independently check each full original PCM's
encoded hash, decoded hash, count and CRC. It reuses the exact original stream
auditor for every report, complete native return byte, UART,63prime returns,
post-prime getter, stage104 and CRC. Only the imported auditor module's PLAN
variable in this separate reader process points to the new execution; no
original file or running process changes. Each report must also identify the
new execution-bound runner. No wake model, frontend or TTS is executed.

The new dynamic metric consumes native positive returns directly. There is no
custom smoothing, rearm or refractory added to the official model. Willow has
36positives/324common/36cross; HiESP has36positives/288common/72cross. Both have
9standalone/27command. Preserve signed dry/full-RIR offsets and unclamped512-call
support; recall still includes saved trailing silence through the original
whole input end, so late hits cannot be described as fast recognition.

## New same-PCM reference selection policy

Policy is `same-far-native-getter-selection-v1`. All13 original DEFAULT/SET
requests and their actual getter bits remain separate. Eligibility uses the
same macro>=.8, standalone/command>=.75, no pre-source positive or negative
events and nominal95% Poisson FAR upper<=.5/h over at least eight full original
background hours. Rank eligible points by greater macro, lower FAR upper,
greater actual getter threshold, then earlier original request order. Requested
numeric strings do not substitute for actual getter values. Identical getters
are retained, including DEFAULT/SET duplicates. Select null if no point passes.

Output saves all points and native/signed events, explicit denominators,
per-source background events and original samples, stream audit identities and
selected/null results for both official models. The reader refuses to overwrite
`calibration.json`. This new development operating point does not alter the old
diagnostic protocol or establish P4/human/independent acceptance parity.

## Constructed review

```sh
python3 research/wakeword_p4/experiments/review_complete_official_calibration_v1.py \
  --gpu-ref 916fed0 --output /tmp/official-calibration-review.json
```

The saved result is `results/mac-complete-official-calibration-review-v1.json`.
Five groups cover both396 targets, exact denominators/bare/Muse/tail support,
all shared legacy360 fields, invalid rows/native clocks, getter-based ranking
and null/tie behavior, an independent1025-sample numerical WAV decode/CRC, and
the original auditor on constructed complete UART/report/raw evidence with
getter/raw corruption controls. No QEMU/model/frontend/P4/TTS or real development
media was accessed. Production full-grid main still requires the actual future
execution and collected results. The review explicitly stubs the auditor's
unused soundfile import and forbids library audio IO; the production GPU keeps
its original dependency. Mac's first missing-import failure is retained in
`results/mac-complete-official-calibration-initial-import-failure-v1.json`.
