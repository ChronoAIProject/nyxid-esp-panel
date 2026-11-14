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
  -D IDF_TARGET=esp32p4 reconfigure
cmake --build NEW_DIR/build --parallel 4
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

## Actual export and complete link

The plan/source were published at `db2839c` before original export session60054,
which exited0. The random context225 model has24301 parameters, six FP32 state
buffers totaling17360 bytes, and a121596-byte TFLite graph with11 builtin types.
All1024 synthetic steps passed the predeclared `1e-4` TensorFlow/LiteRT ceiling;
the maximum actual error was `1.1920928955078125e-7`. The two fresh interpreter
repeats were exact. Original initialized variables were unchanged.

Original complete-build session79066 exited0 using ESP-IDF5.5.4 and
riscv32-esp-elf-g++14.2.0, `-mabi=ilp32f` and
`-march=rv32imafc_zicsr_zifencei_xesppie`. It completed1258 Ninja build steps and
linked an actual P4 ELF and478112-byte app image. The image contains the exact
original121596-byte synthetic FP32 model. All641 installed TFLM/ESP-NN source
files match the frozen trees, apart from the documented CMake dependency line.
Static DRAM data+bss is14796 bytes; this excludes all dynamic arenas/stacks.

An initial CLI attempt (session9143) exited2 because this idf.py does not accept
`-j`; it never configured or compiled the project. Its log is retained. The
corrected command above controls parallelism through CMake, with the original
model, sources and configuration unchanged. The postprocessing collector uses
the actual `-DIDF_VER` compiler argument because IDF5.5.4 project_description
does not expose an `idf_ver` field.

```sh
python3 experiments/collect_fp32_feasibility_build_mac_v1.py \
  --repository-root REPO --run-root NEW_DIR --report NEW_LINK_REPORT.json
```

The result establishes export, synthetic host numerical compatibility and full
P4 link feasibility. It does not establish AllocateTensors/Invoke on P4,
dynamic RAM, latency, incremental panel flash usage, or recognition quality.
No board was accessed. A prospective development-board run remains necessary
before selecting FP32 as the deployment route.
