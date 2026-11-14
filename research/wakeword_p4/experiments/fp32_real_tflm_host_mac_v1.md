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
