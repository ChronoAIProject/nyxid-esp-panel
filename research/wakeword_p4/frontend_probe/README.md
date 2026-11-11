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
