# Current consistency-family Mac diagnostics

The native replay plan was published at `050bf88` before execution. Intake and
saved-array auditing are literal identity substitutions from their previously
executed fixed-grid counterparts, recorded at `391e0ce`; no gates changed.

Using the native Python 3.12 runtime with TensorFlow 2.21.0, NumPy 2.4.6 and
LiteRT 2.3.0, execute from the repository root. `PRIVATE` is an existing local
research output directory, `UPSTREAM` the authenticated microWakeWord checkout.
All output directories must be new; preserve original failures.

```sh
python research/wakeword_p4/experiments/receive_consistency_fixed_grid_witnesses_mac_v1.py \
  --archive "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1.tar.gz" \
  --old-feature-payload "$PRIVATE/gpu-equalized-mac-witnesses-v1/payload" \
  --output-dir "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated" \
  --report "$PRIVATE/mac-consistency-fixed-grid-original-handoff-intake-v1.json"

python research/wakeword_p4/experiments/replay_consistency_fixed_grid_witnesses_mac_v1.py run \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --old-feature-payload "$PRIVATE/gpu-equalized-mac-witnesses-v1/payload" \
  --registered-fc-root "$PRIVATE/complete-fc-registration-replay-v2" \
  --upstream "$UPSTREAM" \
  --output-dir "$PRIVATE/consistency-fixed-grid-native-four-witness-replay-v1"

python research/wakeword_p4/experiments/audit_consistency_fixed_grid_mac_saved_arrays_v1.py \
  --received "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/archive-members" \
  --replay-root "$PRIVATE/consistency-fixed-grid-native-four-witness-replay-v1" \
  --report "$PRIVATE/mac-consistency-fixed-grid-native-saved-array-audit-v1.json"
```

Actual execution completed: all 26 received payloads and nine source receipts
authenticated, 792 native UINT8 steps and raw/quantized inputs exact against the
GPU originals, native repeats and installed TFLM FC registration exact, source
model variables unchanged. The two Willow failures and two Nyx passes on these
predetermined witnesses reproduce; this does not admit any model. Largest
cross-platform float-stream difference is `2.726912498474121e-6`, descriptive.
The original complete540 numerical rejection is unchanged.

Public artifact: `mac-consistency-fixed-grid-native-four-witness-replay-v1.tar.gz`
on release `research-hey-nyx-20261110`, 100849 bytes, SHA256
`d99e0125145976650e7bce3be43078b16df9c1df8573f73a4b5bd0c568a49a06`.
Anonymous redownload authenticated every one of its 15 payloads. Committed
intake/native/audit reports and artifact receipt carry the detailed identities.

The separately frozen `mac-all-final-constant-capture-plan-v1.json` originally
stopped in both settings on the first eager call: passing a Keras variable to
the unchanged primitive triggers TensorFlow's custom-gradient `variables`
requirement. Both original logs, sources and empty outputs are preserved;
there were no saved constant observations. Version 2 explicitly reads each
parameter expression into a Tensor before every method calls that primitive.
The new `mac-all-final-constant-capture-plan-v2.json` retains all 80 constants
and both settings. Execute each setting in a fresh process and output directory:

```sh
python research/wakeword_p4/experiments/diagnose_all_final_constant_capture_mac_v2.py worker \
  --setting default \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --upstream "$UPSTREAM" --output-dir "$PRIVATE/all-final-constant-capture-default-v2"

python research/wakeword_p4/experiments/diagnose_all_final_constant_capture_mac_v2.py worker \
  --setting no_constant_folding_or_arithmetic \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --upstream "$UPSTREAM" --output-dir "$PRIVATE/all-final-constant-capture-no-fold-v2"
```

The diagnostic compares eager/captured/runtime-scale methods
with all allocated compiled constants, saves every difference and repeats, and
performs no activation inference, optimizer update, export, or audio operation.
Isolated graphs do not establish full-training-graph parity, propagation of a
code difference, or recognition/FAR effects. Development and acceptance remain
closed; P4 resource, first-utterance and long-call verification remain pending.

Actual version 2 completed both settings: all 80 constants across three methods
and two settings (480 comparisons) match the original compiled codes on Mac.
The two runs report exact repeats and unchanged variables. An independent
TensorFlow-free saved-array auditor reads raw FlatBuffer buffers, verifies all
480 comparisons and reported differences, and records cross-setting quotient
changes. Only captured-graph quotient arrays change between the two settings;
the saved values, dequantized outputs and integer codes do not change. The
previous GPU tensor20 mismatch does not reproduce in this isolated native Mac
control. Tensor materialization, graph scope and platform differ from the GPU
original; this cannot invalidate its failure or prove the original graph fixed.

```sh
python research/wakeword_p4/experiments/audit_all_final_constant_capture_mac_v2.py \
  --default "$PRIVATE/all-final-constant-capture-default-v2" \
  --no-fold "$PRIVATE/all-final-constant-capture-no-fold-v2" \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --report "$PRIVATE/mac-all-final-constant-capture-saved-array-audit-v2.json"
```

This auditor does not independently verify repeats or variable snapshots, since
those were asserted by the original runner but not separately saved. The public
`mac-all-final-constant-capture-v2.tar.gz` artifact retains all eight actual NPZs,
both reports/logs, audit, and the source/plan snapshot; its committed receipt
records the archive and anonymous redownload hashes.
