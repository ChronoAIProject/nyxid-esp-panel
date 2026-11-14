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

## Saved-capture reducer

The trained protocol now has its own reducer. It authenticates the original
fixture plan, all generated files, original TF/LiteRT references and the exact
linked image. It requires one complete BEGIN, every ordered step and case,
and one DONE across all 48 executions and 8,336 calls. Every output must be
finite and within probability bounds, match both references to absolute1e-4,
and match the other fresh repeat byte-for-byte. Every Invoke, including cold
calls, must be positive and at most30,000us. Raw outputs/timings must reproduce
the printed summaries; duplicate JSON keys, malformed, extra or missing protocol
records are rejected. Reported allocator use must fit the prepared capacities.

The error gate promotes the original float32 values before subtraction. The
device's rounded float32 error or printed `passed` cannot mask an over-bound
error. The reducer separately checks consistency of the device's rounded
summary. It records raw-output hashes, cold Invoke/setup times and allocator
use per execution; it does not infer full-panel headroom from these numbers.

```sh
python3 research/wakeword_p4/experiments/reduce_fp32_trained_p4_capture_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN \
  --capture SAVED_UART.log --report NEW_REDUCTION.json
```

The constructed checker uses the original authenticated reference values with
invented timings/resources. All 26 invalid-input cases were rejected, including
a one-ULP fresh-repeat difference inside numeric tolerance, a LiteRT-only
discrepancy, cold-call overrun and a float32-subtraction rounding counterexample.
Its valid full-protocol case and CLI completed; these are **zero actual P4
Invokes** and provide no hardware performance evidence. See
`../results/mac-trained-p4-capture-reducer-constructed-checks-v1.json`.

```sh
python3 research/wakeword_p4/experiments/check_trained_p4_capture_reducer_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN --output-dir NEW_CHECK_DIR
```

Capture origin is deliberately a separate unresolved requirement: the future
app-only execution/restore receipts must bind the actual development board,
confirmed slot and flashed image to the UART capture. A passing parser does not
authenticate those facts or qualify a candidate. That runner integration and
actual P4 execution remain pending; the old random-weight validator is still
unsuitable for this protocol.
