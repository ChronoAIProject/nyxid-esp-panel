# Candidate-free FP32 runtime feasibility

The user objective does not require INT8. Before undertaking another numerical
adaptation, check whether the existing context225 architecture can run as FP32
with the installed P4 runtime. This investigation does not revise any previous
integer gate or rejection, interrupt GPU experiments, or select a model.

The prospective plan binds the GPU architecture helper at `f9e773a`, all twenty
upstream Python sources, installed versions, both component trees, this exporter,
the standalone C++ probe and sdkconfig defaults. Only the pure architecture
function and arguments are imported. No candidate weights or data are read.
The seed is 20261114. BN is frozen, the original streaming conversion has six
zeroed states, and conversion permits FP32 builtins without optimization or
quantization. Four deterministic 256-step feature streams compare TensorFlow to
two fresh BUILTIN_REF LiteRT interpreters at the predeclared maximum absolute
error `1e-4`. This is a synthetic runtime check, not recognition evidence.

```sh
PYTHON experiments/probe_fp32_runtime_feasibility_mac_v1.py plan \
  --repository-root REPO --upstream UPSTREAM --plan PLAN.json
# Publish the plan and sources before this operation:
PYTHON experiments/probe_fp32_runtime_feasibility_mac_v1.py export \
  --repository-root REPO --upstream UPSTREAM --plan PLAN.json --output-dir NEW_DIR
```

The export creates an isolated full IDF project. Runtime kernel sources are
copied unchanged; its TFLM CMake file gains only an explicit `esp-nn` dependency,
since the build disables dependency management and uses the installed copies.
With ESP-IDF 5.5.4 activated:

```sh
IDF_COMPONENT_MANAGER=0 idf.py -C NEW_DIR/project -B NEW_DIR/build \
  -D IDF_TARGET=esp32p4 build
```

Only build/link is authorized by this protocol. The probe reserves internal
256 KiB main and 64 KiB variable arenas, but actual allocation, arena usage,
latency and panel headroom remain unknown until a separately planned P4 run.
It has no sensors, networking, credentials or persistent writes. **Do not flash
this standalone image using the panel partition table or operate the dining
room production board.** A future board experiment must verify development
board identity/active slot and back up/restore the original app, following the
repository's existing app-only rules.

The resulting model, arrays and linked executable are synthetic artifacts.
They cannot establish Hey Nyx quality, satisfy the official Hey Willow
comparison, permit fresh development scoring, or qualify any rejected model.
