# Context225 preparation and complete finite self-test

This is an independent opt-in research entry for a fixed-final120000-step,
24301-parameter teacher60 Hey Nyx model with225 context frames. The original
150-frame GPU profile and NYXKWS01 packaging path remain unchanged. This is
preparation infrastructure; the current broad-training exports failed numerical
admission and must not receive an actual candidate profile from this work.

The caller must first establish completed-model numerical admission, then freeze
a non-null eligible point using the registered new-development calibration.
The preparer checks the frozen artifact bindings and metadata; it does not
rerun training, numerical admission or the development selection calculations.
It does not authorize independent acceptance or production deployment.

## Profile

Keep models, PCM, calibration and the actual profile outside git. The profile
uses schema `complete-development-p4-profile-v1`, `accepted: false`, an explicit
execution model ID, `target_id: "hey_nyx"`, `input_frames: 225`,
`event_arithmetic: "uint8-sum-binary64-v1"`,
`tail_policy: "whole-features-final-zero-pad-stride3-v1"`,
`smoothing_steps: 5`, `refractory_steps: 67`, and the exact selected `threshold`.

Its `artifacts` object has keys `model`, `training`, `completion`, `calibration`,
`execution`, and `selftest`; each value contains `path` and `sha256`. Relative
paths resolve against the profile directory. The training/completion reports
bind the fixed-final120000 step,100 PTQ batches,120100 actual batch matches,
unchanged BN snapshots and exact model identity. The calibration is the original
`complete-development-calibration-v1` result with the same model/materials as
the complete raw execution; selected must be eligible, registered and present
in the grid. Null selection, a legacy calibration layout or a150-frame report
is rejected. These checks do not independently recompute the reports' evidence.

```sh
python research/panel_wake_runtime/prepare_complete_runtime.py \
  --profile "$COMPLETE_PROFILE" --model "$WAKE_MODEL" \
  --calibration "$WAKE_CALIBRATION" --selftest "$MODEL_BOUND_SELFTEST" \
  --output "$PRIVATE_GENERATED"
```

For the full application, add the following selector to the existing model,
calibration and self-test build arguments:

```sh
idf.py -B "$PRIVATE_BUILD" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE="$COMPLETE_PROFILE" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_MODEL="$WAKE_MODEL" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION="$WAKE_CALIBRATION" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST="$MODEL_BOUND_SELFTEST" build
```

This selector cannot be combined with the legacy candidate profile or embedding
bundle. It registers all profile/artifact files as CMake configuration
dependencies. The generated manifest preserves the profile and artifact hashes;
the configuration uses a binary64 threshold without a float suffix and68000
priming samples. Exact existing generated files are idempotent; different,
partial or stale outputs are refused. Use a fresh build directory after changes.

## NYXKWS02 finite input

The binary format is8-byte magic `NYXKWS02`,32-byte model SHA256, little-endian
UINT32 original sample count and output count, original mono PCM16 at16kHz,
then one raw UINT8 byte for every model invocation. Header size is48 bytes.
The bounded informative fixture contains at least80 outputs and at most16s of
PCM. Tests with header-shaped fake models check metadata only; they cannot
produce a deployable candidate.

For N samples, the complete frontend frame count is `max(0, ceil(N/160)-2)`;
the model call count is `ceil(frames/3)`. The runtime feeds every original
sample once, pads only the final waveform hop to160 samples, and pads only
the final incomplete model group with quantized zero-valued features. It
checks every raw output, including a last invocation that occurs after the
last complete480-sample chunk. Header size, bounded counts and payload length
are validated before sample pointers are constructed. The self-test detector
is destroyed before the normal microphone detector is created.

The public microphone wrapper still consumes480 samples per call, and never
calls finite-input `finish()`. No continuous microphone flush, extra reader,
playback, recording, I2S/AEC/AGC/FIFO or PCM-speed change is introduced.

## Complete fixture generation and UART audit

After numerical admission and a frozen eligible development point, generate
the private fixture from one explicitly selected, permitted source WAV. The
old `training/prepare_runtime_selftest.py` remains the V1 generator; it must not
be used to generate this complete contract.

```sh
python research/panel_wake_runtime/prepare_complete_selftest.py \
  --model "$WAKE_MODEL" --audio "$SOURCE_WAV" \
  --calibration "$WAKE_CALIBRATION" --output "$PRIVATE_FIXTURE.bin"
```

The new generator preserves the entire original 16kHz mono PCM16 source and
prepends the application's actual 68,160 samples of silence. The requested
prime is 68,000, rounded by the existing 480-sample application loop. No source
tail is truncated, and no PCM is padded at the prime/source boundary or to a
480-sample chunk. The whole stream is replayed with a fresh MicroFrontend and
reference LiteRT interpreter twice. Only the final 160-sample waveform hop and
the final three-frame model group are padded; raw UINT8 outputs are saved
without float reconstruction. Model operators and INT8/UINT8 tensor interfaces
must pass the original model inspector. The binary and JSON report use fresh
paths; different/partial files cannot be overwritten.

The complete packager embeds `PANEL_WAKE_SELFTEST_SHA256` in the configuration.
The actual C++ V2 self-test logs that fixture SHA, original sample count and
output count before its existing verdict. This identifies the exact fixture
instead of relying on model identity and output count alone. Audit the saved
UART with the actual binary, generated JSON and unchanged calibration:

```sh
python research/panel_wake_runtime/audit_complete_selftest.py \
  --uart "$PRIVATE_UART" --fixture "$PRIVATE_FIXTURE.bin" \
  --fixture-report "$PRIVATE_FIXTURE.json" \
  --calibration "$WAKE_CALIBRATION" --output "$PRIVATE_NEW_AUDIT"
```

The audit checks complete binary lengths and hashes, actual prime/source bytes,
tail policy, selected threshold and every declared activation step recomputed
from raw UINT8 bytes. It requires exactly one matching fixture identity before
the verdict, then reuses the original UART model/output/activation audit. A
successful audit only authenticates the reported diagnostic for this input;
it does not independently observe a model invocation or prove general P4
equivalence. Constructed backend tests use real NumPy with explicit frontend/
interpreter doubles; actual learned-model generation and hardware audit remain
pending a qualified candidate.

## Verification and limits

```sh
cd research/panel_wake_runtime
python -m unittest -v test_complete_selftest test_prepare_complete_runtime \
  test_prepare_runtime test_audit_runtime_selftest
python review_complete_runtime_flow.py \
  --legacy-ref 68d27387e9916f350bf733aa1b04e271885d5f80 \
  --output "$PRIVATE_NEW_FLOW_OUTPUT"
```

The host flow runner compiles the actual C++ runtime with explicitly constructed
frontend/interpreter doubles, compares complete/legacy/original continuous
prefixes, checks original-sample ordering and final padding under27 input/chunk
combinations per new/legacy arm, and exercises V2 success/rejection plus V1
self-test. It uses undefined-behavior/bounds sanitizers with immediate failure.
`--sanitizers address,undefined` is available on compatible hosts. On this Mac,
the initial Apple ASan runtime spun recursively during initialization before
main; a native stack sample records the cause and the original terminated
attempt is preserved. The successful run does not claim ASan coverage.

CMake tests execute the actual component selector with a test-only IDF wrapper,
not an ESP-IDF full-app configure/link. A configured P4 application that already
includes the experimental component supplies the real compile database:

```sh
python review_complete_runtime_cross_compile.py \
  --compile-database "$EXISTING_P4_BUILD/compile_commands.json" \
  --output "$PRIVATE_NEW_COMPILE_OUTPUT"
```

All three actual P4 C++
translation units (complete/selftest, legacy/selftest, legacy/continuous) compile
with the installed ESP-SR/TFLM/MicroFrontend headers, without warnings. Neither
the host doubles nor compilation establishes FFT equivalence, learned-model
numerics, arena allocation, recognition, device resources or voice acceptance.
Actual qualified-model fixture generation, full-app link and hardware tests
remain required after external candidate admission.
