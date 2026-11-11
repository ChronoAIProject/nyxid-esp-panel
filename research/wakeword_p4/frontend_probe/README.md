# Cross-host MicroFrontend table diagnostic

Frozen before first execution. GPU receipt at `7f322e7` reports identical PCM
for30 training-only witnesses but14 nonidentical full feature arrays:16 changed
elements, all at channels32/33, maximum raw delta133. This does not establish the
cause or explain the recognition gap. Running v40 retains the original Mac
arrays and seed127726; this diagnostic must not change that protocol.

First compare **initialized filterbank and window tables** without PCM or a
model. Compile the same unmodified pymicro-features2.0.2 C++ source using the
small driver here. Record every included frontend header, the two translation
units, driver/runner/executable hashes, compiler and exact options. Dump actual
integer tables plus separately evaluated float32 mel-frequency bit patterns.
Do not dump raw structs, padding or pointers. On Mac, attempt native arm64 and
x86_64/Rosetta. Preserve build/run failures; they do not justify installing
software or changing feature data. GPU can independently run the same driver
with its source/compiler after receiving it through Issue13.

First Mac run: arm64 succeeds; x86_64 compiles but execution fails with
`Bad CPU type in executable` (no usable Rosetta). Preserve this failure instead
of installing it. `otool` shows fused multiply-adds in the default arm64 table
initializer. Before the alternate run, add the fixed `--fp-contract off`
comparison to isolate whether those integer tables depend on contraction on
this same machine. This tests one cause without changing the original pack or
claiming Linux/installed-wheel behavior.

```sh
python research/wakeword_p4/frontend_probe/run_coefficients.py \
  --source /path/to/pymicro-features-source --output /new/output/directory \
  --arch arm64
```

Use another new directory with `--arch x86_64` on Mac, or omit `--arch` on Linux.
Compare actual arrays, not executable hashes. A mismatch identifies a table
difference in these standalone builds, not automatically the installed wheels
or device. Agreement does not rule out FFT/PCAN/runtime differences. No tolerance
change, audio I/O, model scoring, training, hardware access, NVS or network is
part of the probe. No validation, new development or final dataset is opened.

The unused Mac `training/old_source_replacement.py` and its real-pack probe were
superseded before execution by GPU v40's completed implementation/audit at
`911937b`. Keep them only as provenance; do not integrate their seed137726 or
run them to duplicate v40.

## Full native frontend versus installed wheel

Next, before running, freeze a host-only comparison on the same30 original
training witnesses (first canonical per engine/label, all three variants).
`compare_native.py` builds the full unmodified2.0.2 frontend in isolated shared
libraries, default versus `-ffp-contract=off`. It does not replace the installed
extension. A small C ABI feeds exact160-sample hops and exposes actual coefficient293.
The fixed WAV, rendered PCM and saved full-array hashes are checked; both native
outputs and the current installed wheel are compared to the immutable saved
arrays. Retain every changed element plus all outputs in `arrays.npz`, compiler,
all source/header and binary hashes. No learned model, scoring, audio I/O or
hardware. Differences remain diagnostic and never modify v40's input arrays.

```sh
python research/wakeword_p4/frontend_probe/compare_native.py \
  --source /path/to/pymicro-features-source \
  --study /path/to/old-complete-container-pack-v1 \
  --source-root /path/to/original-workspace --output /new/native-comparison
```

This can connect the standalone coefficient observation to actual feature
outputs on these inputs, without claiming to inspect the installed extension's
private state. Build failures and all nonidentical outputs remain in the new
directory. No threshold or feature tolerance is chosen from this diagnostic.

Mac execution at `97f84c0` completed: all30 rendered PCM and saved-array hashes
verified; installed wheel and default native rebuild each match all30 saved
arrays exactly. No-contraction matches29/30. Its sole changed element is pack466,
frame380/channel39:177→133, with coefficient293 changing1401/2695→1400/2696.
This local contraction experiment still does not explain the Linux32/33
differences. It does show why a one-unit coefficient difference need not mean
a one-unit feature difference near the log/noise floor.

[Complete report](../results/frontend-native-mac-v1.json) and
[public evidence index](../../handoff/assets-frontend-native-mac-v1.json) preserve
all90 output arrays, all differences, source/binary hashes and compiler commands.
The221373-byte archive was anonymously downloaded again; all10 payload members
passed length/SHA and safe-path checks. It contains no audio, firmware, native
binaries, board state or credentials. Original training WAVs/pack and the
unmodified upstream C++ source remain external dependencies. This is numerical
diagnosis on30 training witnesses, not recognition or Hi ESP acceptance.

## Isolated Linux coefficient intervention

GPU `c186b46` reports all original source hashes exact and only one integer
filterbank pair differing from default Mac: index211, Linux2471/1625 versus
Mac2470/1626. The pair is in block33, contributing to output channels32/33.
Before executing this next arm, add `--weight-pair 211 2471 1625` to the same
fixed30-witness comparison. It creates a fresh default native handle per input
and replaces only that pair before the first PCM hop. Original library source,
installed wheel, saved arrays, rendering and all other coefficients stay fixed.
The diagnostic setter rejects invalid indices or noncomplementary Q12 weights.
Keep this operation confined to host probes; do not apply it to training or P4.

Compare all30 resulting difference counts, maxima and first difference values
to GPU's original `4090-old-complete-cross-host-v1.json`, retaining all complete
predicted arrays and hashes. The existing GPU report lacks full Linux output
hashes: matching its summaries alone cannot establish full-array Linux parity.
Ask GPU to compare actual full hashes independently before making that claim.
No model or recognition inference is part of this intervention.

The intervention completed and reproduces all published Linux observations for
all30 inputs:16 unchanged full arrays,14 changed arrays,16 changed elements,
the same maxima and first differences. The default native control and wheel
still match all30 originals. [Report and lossless full-output
deltas](../results/frontend-linux-weight-intervention-v1.json) bind original
saved arrays and predict every complete Linux array hash; all30 predictions
were independently reconstructed from the original arrays plus16 explicit
deltas. This removes the need to download another full-array archive. Complete
actual Linux hashes still need independent comparison; the original GPU report
only supplies summaries. Training and device behavior remain untouched.
