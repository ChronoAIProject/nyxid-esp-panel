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
