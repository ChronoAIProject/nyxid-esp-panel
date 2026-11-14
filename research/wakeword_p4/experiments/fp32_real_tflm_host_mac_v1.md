# Original FP32 vectors through the real MicroInterpreter

The USB development board is currently absent. This bounded host experiment
checks the actual installed TFLM interpreter/allocator/resource variables and
all eleven registered operators against the already exported random FP32
model and original1024 synthetic feature steps. It accesses no trained model,
audio, recognition/development/acceptance records or board.

The plan is published before compilation/execution. It binds the original
model, NPZ and export receipt; the complete installed component tree including
inline headers; every selected translation unit; compiler/version/flags; and
both probe sources. The implementation uses the real MicroInterpreter and
three installed `esp_nn` convolution/depthwise/FC source files. `ESP_NN=0`
excludes the unexecuted integer accelerator paths for this FP32-only host
build. The only ESP platform adapter is a chrono implementation of the timer.
No kernel math, model operator or state handling is replaced.

Each original stream is run with two fresh interpreters, with internal
256KiB main and64KiB variable arena capacities. All2048 actual output floats
are saved. Fresh repeats must be exact; original TensorFlow and LiteRT outputs
must each be within the predeclared maximum absolute error1e-4. Native64-bit
allocation numbers are descriptive and cannot be presented as P4 RAM usage or
latency. P4 allocation/Invoke and full-panel resources still require hardware.

```sh
PYTHON experiments/run_fp32_real_tflm_host_mac_v1.py plan \
  --repository-root REPO --prior-root ORIGINAL_FP32_RUN --plan NEW_PLAN.json
# Publish the plan and sources before running:
PYTHON experiments/run_fp32_real_tflm_host_mac_v1.py run \
  --repository-root REPO --prior-root ORIGINAL_FP32_RUN --plan NEW_PLAN.json \
  --output-dir NEW_DIR
```

This does not authorize another training/scoring run or reopen any integer
admission gate. The existing full integer rejections, frozen v30 and all
source/data reservations are preserved.

## Actual result

The plan/source were published at `d8ca154` before original session80355. That
session exited0: all56 real runtime translation units compiled and the actual
interpreter executed2048 steps. The original1024 inputs were each replayed in
two fresh interpreters; saved repeat bytes are exact. Maximum absolute error is
`5.960464477539063e-8` against the original LiteRT output and
`1.1920928955078125e-7` against TensorFlow, within the predeclared1e-4 ceiling.

Each of the eight actual host cases reported44256 main-arena bytes used and
1040 variable-allocator bytes used. These are native64-bit host measurements;
the P4 build enables ESP-NN and has a different ABI. They are not MCU or panel
memory measurements. Fresh host arena allocations are value-initialized by
the C++ harness; this run does not test poisoned/uninitialized memory.

The independent saved-byte audit uses struct-unpacked raw C++ output rather
than the producer's array reduction. It verifies all2048 captured float bytes,
the original input bytes/references, all56 actual translation-unit hashes, the
eight captured allocator records and the exact repeat/numerical results. It
does not perform another inference or broaden the evidence to real models.

```sh
PYTHON experiments/audit_fp32_real_tflm_host_mac_v1.py \
  --run-root ORIGINAL_HOST_RUN --prior-root ORIGINAL_FP32_RUN \
  --plan ORIGINAL_PLAN.json --report NEW_AUDIT.json
```

The next hardware stage is separately frozen in
`fp32_p4_resource_plan_mac_v1.md`. It uses the original P4 image and the existing
app-only restore runner. No development USB serial device was present during
this host run; no board operation occurred.
