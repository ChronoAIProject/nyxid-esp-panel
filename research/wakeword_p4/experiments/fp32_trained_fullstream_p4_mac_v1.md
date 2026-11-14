# Original trained FP32 P4 preparation

This standalone numeric/resource probe uses the four unchanged original trained
raw-FP32 models and complete saved feature streams from the admitted original
fullstream packet. It does not generate, export, quantize or invoke a model
during preparation. It has no microphone, speaker, network or NVS operations.

Selection is fixed by original request metadata: the first variant-zero stream
for each Willow/Nyx target and Piper/Kokoro/Qwen engine. All six full streams,
including startup and tail, remain unchanged. Four models × six streams × two
fresh interpreters give 48 executions and **8,336 Invoke calls**. Integer-trained
raw reconstructions remain a distinct observable from their quantized training
execution. These selected streams are numerical witnesses, not recognition
acceptance or all 540 original cases.

```sh
PYTHON research/wakeword_p4/experiments/prepare_fp32_trained_fullstream_p4_mac_v1.py \
  --repository-root REPO --packet-root ORIGINAL_FULLSTREAM_PACKET \
  --output-dir NEW_PRIVATE_RUN --plan NEW_PUBLIC_PREPARATION.json
```

Preparation authenticates the original intake/packet manifest, every used
model/NPZ and shared input identity across models. It copies the existing pinned
TFLM/ESP-NN components unchanged except for the previously documented CMake
dependency declaration. The generated project contains only exact original
model/input/reference bytes and the numeric probe. `fixture-plan.json` records
all source, runtime, asset and generated-file hashes.

The probe uses the same 12 registered operators and value-initialized 256 KiB
main / 64 KiB variable arena capacities as the admitted host implementation.
Both capacities are allocated in internal RAM. Each execution creates fresh
allocator, interpreter and variable state; no mid-stream resets occur. Every
output is emitted as its exact float32 bits and every Invoke duration is kept,
including the cold first call. UART output and task yields are outside Invoke
timing. Per-case reports retain actual main/variable allocator use, interpreter
setup time (excluding arena allocation/zeroing), maximum error against original
TF and maximum Invoke time. The unchanged gates are absolute error ≤1e-4 and
every Invoke ≤30,000 us. Capture reduction must additionally compare all outputs
against original TF/LiteRT and both fresh repeats byte-for-byte.

The generated partition CSV is **only a build layout** allowing the diagnostic
assets to link. Never flash that table, a bootloader, NVS or filesystem. No board
operation is part of preparation/build. A future app-only execution requires a
fresh confirmed development-board identity, current app/slot and verified
backup, a probe image fitting the confirmed slot, and finally restoration plus
independent original-app/service verification. The dining-room production board
is excluded. Do not infer a current serial port, IP, offset or backup from old
runs. The existing random-weight probe capture validator does not understand
this new trained fullstream protocol and must not be used to certify it.

Expected build environment is the previously linked ESP-IDF v5.5.4,
`riscv32-esp-elf-g++` esp-14.2.0_20260121, `ilp32f`, configured 360 MHz and the
existing P4 revision defaults. Build-only success does not measure real P4
arena/latency, panel headroom, audio integration, recognition/FAR or acceptance.
Actual target/ABI/toolchain and all embedded bytes must be checked after link.

## Completed cross-build

Preparation was published as `a828342` before build. Original build session
`7488` exited 0 under ESP-IDF v5.5.4 / esp-14.2.0_20260121. The independent
build collector authenticates every prepared generated file, copied runtime
component, all four model/six stream/reference assets inside the actual image,
the RISC-V ELF target and `ilp32f` ABI. The 1,331,120-byte image has 14,780 bytes
of static DRAM data/bss, excluding all runtime arena allocations and stacks.
See `../results/mac-fp32-trained-fullstream-p4-link-v1.json`.

No hardware was accessed. The prepared 8,336 calls have not run on P4, so this
does not establish target timing, arena use, repeat equality or numeric parity.
The original build log, ELF, image, section sizes and fixture assets remain in
the private `fp32-trained-fullstream-p4-v1` run directory. Reproduce collection:

```sh
python3 research/wakeword_p4/experiments/collect_fp32_trained_fullstream_p4_build_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN --report NEW_REPORT.json
```
