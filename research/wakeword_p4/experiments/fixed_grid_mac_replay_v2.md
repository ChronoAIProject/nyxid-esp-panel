# Native Mac replay of the four original fixed-grid models

This is an executed diagnostic on four predetermined, already-consumed cases.
It does not qualify any model or replace full recognition/FAR/P4 acceptance.

The original handoff is `gpu-fixed-grid-mac-witnesses-v1.tar.gz` on release
`research-hey-nyx-20261110`, 839771 bytes, SHA-256
`27677e306b34471b82d4dd6eb367192e7e389c55ec201f36a887f93cff891d8b`.
The intake checks all 26 payload files against the published manifest, nine
original receipt identities, the original Mac model/H5 plan, and the two prior
input NPZ bytes. It copies files under their original relative names for replay.

Original v1 stopped with `KeyError: 'ops'` after first-target float calculations,
before integer inference or saving a completed case. Its source and plan remain
unchanged; its log and failure receipt are retained. Commit `8dfb234` published
v2 before execution. The only inference-source change reads operator metadata
from the already-bound compile-plan template, checking its SHA against the
recovered receipt. Models, H5 files, inputs, repetitions, phase, arithmetic and
numerical gates are unchanged. The original run was not resumed or overwritten.

Run from the repository root. Use the existing diagnostic Python environment
(TensorFlow 2.21.0, NumPy 2.4.6, LiteRT 2.3.0 and versions in the published lock),
the original SHA-checked upstream20 Python sources, and the prior authenticated
native registered-FC library/report. The latter is available in
`mac-complete-fc-native-replays-v1.tar.gz`, SHA-256
`a24f3206439570de88db79b03a5bdbe15ceed49753d6e95453bf1a040e11e5b1`.
Set `task_python`, `task_root`, `task_upstream` and `task_repo` to their actual
locations. Output directories and report paths must be new.

```sh
python3 research/wakeword_p4/experiments/receive_fixed_grid_witnesses_mac_v1.py \
  --archive "$task_root/gpu-fixed-grid-mac-witnesses-v1-received/gpu-fixed-grid-mac-witnesses-v1.tar.gz" \
  --old-feature-payload "$task_root/gpu-equalized-mac-witnesses-v1/payload" \
  --output-dir "$task_root/gpu-fixed-grid-mac-witnesses-v1-authenticated" \
  --report "$task_root/intake.json"

"$task_python" research/wakeword_p4/experiments/replay_fixed_grid_witnesses_mac_v2.py run \
  --payload-root "$task_root/gpu-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --old-feature-payload "$task_root/gpu-equalized-mac-witnesses-v1/payload" \
  --registered-fc-root "$task_root/complete-fc-registration-replay-v2" \
  --upstream "$task_upstream" \
  --output-dir "$task_root/fixed-grid-native-four-witness-replay-v2"

"$task_python" research/wakeword_p4/experiments/audit_fixed_grid_mac_saved_arrays_v1.py \
  --received "$task_root/gpu-fixed-grid-mac-witnesses-v1-authenticated/archive-members" \
  --replay-root "$task_root/fixed-grid-native-four-witness-replay-v2" \
  --report "$task_root/cross-platform-audit.json"
```

Actual result: four models, 792 steps. Raw/quantized inputs and all UINT8 outputs
match the original GPU arrays exactly. Native float/integer/embedding/FC repeats
are exact; the actual registered TFLM FC path matches all 792 original LiteRT FC
codes. Float streaming cross-platform maximum difference is 2.384185791015625e-6;
offline-phase cross-platform maximum difference is 1.0669231414794922e-5. These
are descriptive maxima, not new tolerances. Native phase checks remain below
the original 1e-4 bound. Original variables remain unchanged.

| Original predetermined case | Native five-step error p99 | Original numerical gate |
| --- | ---: | --- |
| Willow 14, float control | .0670888826 | pass |
| Willow 14, integer QAT | .1461385649 | fail |
| Nyx 48394, float control | .1949088261 | fail |
| Nyx 48394, integer QAT | .0035709143 | pass |

All four case outcomes reproduce GPU originals. A passing witness does not
qualify its model: the original complete540 rejects all four candidates. No
fresh development, independent acceptance, new audio IO, optimizer update,
export, board operation or complete TFLM/P4 model inference occurred here.
