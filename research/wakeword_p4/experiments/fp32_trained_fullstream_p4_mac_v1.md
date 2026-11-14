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

No hardware was accessed during preparation/build. At that stage the prepared
8,336 calls had not run on P4; the actual execution below was completed later.
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

Capture origin is a separate requirement: app-only execution/restore receipts
must bind the actual development board, confirmed slot and flashed image to
the UART capture. A passing parser does not authenticate those facts or qualify
a candidate. The runner below supplies that integration; its actual execution
is recorded below. The old random-weight validator is still unsuitable.

## Trained app-only runner

`run_trained_p4_capture_mac_v1.py` uses the authenticated prepared image and the
new reducer. All device inputs must be current observations; there are no
default MAC, IP, app address, slot size or partition-table address. Use only the
development USB board, excluding the dining-room production board. Keep its
output outside the checkout: it contains raw console, device identity, flash
readbacks and a full original app-slot backup.

Enumerate with `IDF_PYTHON -m serial.tools.list_ports -v`, including USB VID/PID,
serial number and location. Do not use `/dev/cu.usb*` as an inventory: WCH
devices appear as `/dev/cu.wchusbserial*`, and CP210x drivers may use
`/dev/cu.SLAB_USBtoUART*`. The runner validates the explicitly supplied port
against current USB enumeration metadata; it does not select one automatically.
An empty filename glob does not mean the development board is disconnected.

Before writing, the runner checks USB ownership/MAC, idle app identity and wake
state, fresh on-device partition-table MD5/layout, exact active slot geometry,
known-good app hash/structured ELF descriptor and a matching fresh full-slot
readback. A read-only `nyx-status` on the USB console must match the HTTP host
IP, service slug and firmware version. This prevents treating another panel
with the same firmware image as the USB-connected board.

Only the verified app address can reach `write_flash`; the diagnostic partition
table, bootloader, NVS and filesystem are not write inputs. After writing, the
entire probe image is read back and compared before capture. UART capture opens
before reset and retains partial lines across timeout reads. It has a bounded
600-second default for all8,336 calls plus115200-baud output, with progress
messages; no microphone, speaker or audio fixture is used.

Any probe write attempt, including partial failure, capture timeout, parser
failure or normal cancellation, enters restoration after the UART context has
closed. The fresh full-slot backup restores app bytes and original padding.
The whole slot, original partition table and OTA metadata are then read back
and must match exactly. Finally the app ELF/slot, idle voice, unchanged service
selection and listening wake state must recover. `online` additionally requires
Nyx connected; explicit `app-only` preserves the separate offline-engineering
case and records the actual connection state without claiming it is online.
Failed restoration remains a failed execution; it cannot yield a passing run.

Run after publishing the sources and obtaining fresh development-board details:

```sh
IDF_PYTHON research/wakeword_p4/experiments/run_trained_p4_capture_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN \
  --port CURRENT_DEV_USB --expected-mac CURRENT_DEV_MAC --status-url CURRENT_STATUS_URL \
  --restore-app CURRENT_GOOD_APP --expected-restore-sha256 CURRENT_GOOD_APP_BIN_SHA \
  --expected-app-sha256 CURRENT_APP_ELF_SHA \
  --app-offset CURRENT_APP_OFFSET --slot CURRENT_OTA_LABEL --slot-bytes CURRENT_SLOT_BYTES \
  --partition-table-offset CURRENT_TABLE_OFFSET --restore-readiness online \
  --output NEW_PRIVATE_EXECUTION_DIR
```

The runner records source/image/capture/backup/table/OTA-data hashes and current
execution parameters in `report-private.json`. Independent evidence collection
must inspect these raw receipts after an actual run; a constructed backend
always records `capture_origin_bound=false`. Never use a saved earlier IP,
backup or slot as a fresh observation. SIGKILL/power removal cannot be recovered
by a Python `finally` block.

The constructed checker passed25 complete transaction scenarios: preflight
identity/busy/geometry/backup rejection, online versus app-only readiness,
partial probe write, corrupt readback, timeout/cancellation, invalid protocol
and restoration write/readback/table/OTA-data/readiness failures. It also
exercises actual writer dispatch with zero commands for a foreign USB identity,
and actual UART FIFO code with split records and closure on success/error.
The original linked image's structured ELF descriptor matches its actual ELF.
No physical board was accessed. Reproduce in a new private output directory:

```sh
python3 research/wakeword_p4/experiments/check_trained_p4_runner_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN --output-dir NEW_CHECK_DIR
```

See `../results/mac-trained-p4-app-only-runner-constructed-checks-v1.json`.
## Actual P4 execution and restoration

The development ESP32-P4 revision v1.3 was connected through a WCH USB serial
bridge. The earlier `/dev/cu.usb*` inventory missed its
`/dev/cu.wchusbserial*` device. This was an inventory/runner validation bug, not
evidence that the user had disconnected the board. Commit `d9d0ac7` validates
current USB enumeration metadata; six port cases and the existing 25
transaction/restoration cases pass. Actual USB metadata, chip identity, console
IP/slug/firmware, HTTP status, slot geometry and full backup were then checked
fresh before the run.

All **48 executions / 8,336 real P4 Invokes** completed and passed. The maximum
absolute error against original TF was `1.3709068298339844e-6`, and against
original LiteRT `1.5944242477416992e-6`, both below the original `1e-4` gate.
Fresh-repeat output bytes matched exactly. Every Invoke, including cold calls,
met the unchanged 30,000-us deadline; the worst was **5,206 us**.

| Original raw FP32 model | Invokes | Worst Invoke (us) | Main arena used (bytes) | Variable arena used (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Hey Willow float control | 2,084 | 5,206 | 51,800 | 528 |
| Hey Willow integer-trained raw reconstruction | 2,084 | 5,117 | 51,800 | 528 |
| Hey Nyx float control | 2,084 | 5,129 | 51,800 | 528 |
| Hey Nyx integer-trained raw reconstruction | 2,084 | 5,116 | 51,800 | 528 |

These are allocator usage figures inside the probe's **256-KiB main + 64-KiB
variable reservations**, not its whole internal-RAM footprint or full-panel
headroom. The integer-trained rows still execute raw FP32 reconstructions and
do not measure quantized inference. Only the six preselected complete training
feature streams were used; there was no waveform frontend or audio I/O.

The probe image was read back exactly before capture. The original full app
slot, partition table and OTA metadata were restored and read back byte-for-byte.
After the runner exited, a separate HTTP check confirmed the original app ELF,
firmware, valid idle OTA slot, service/agent selection and idle Hi ESP listener;
the serial port was unowned. Nyx remained `connecting`, matching the explicitly
selected `app-only` readiness. This does not claim a working voice connection.
No dining-room board, fresh development or independent acceptance was accessed.

The public result and unchanged raw UART are
`../results/mac-trained-p4-fullstream-hardware-result-v1.json` and
`../results/mac-trained-p4-fullstream-uart-v1.log.gz`. A separate reducer invocation
reproduced the runner's entire reduction exactly. The gzip artifact retains the
original UART bytes, including CRLF, without line-ending normalization:

```sh
gzip -dc research/wakeword_p4/results/mac-trained-p4-fullstream-uart-v1.log.gz > /tmp/trained-p4-uart-original.log
python3 research/wakeword_p4/experiments/reduce_fp32_trained_p4_capture_mac_v1.py \
  --repository-root REPO --run-root ORIGINAL_PRIVATE_RUN \
  --capture /tmp/trained-p4-uart-original.log \
  --report NEW_REDUCTION.json
```

This closes the standalone original-model P4 numerical/resource subset and its
restoration check. Full 540-case P4 coverage, new continuation weights,
full-panel headroom, recognition/FAR, first-utterance, voice connectivity and
long-call/reconnect acceptance remain outstanding.
