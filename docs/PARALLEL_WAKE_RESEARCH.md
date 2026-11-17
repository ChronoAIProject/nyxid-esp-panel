# Optional C161/B40/history research runtime

The default firmware continues to embed only C161. To build the original optional
three-model handoff, select a separate build directory and pass
`-D NYX_PANEL_WAKE_RESEARCH_HANDOFF=/absolute/path/to/triple-handoff` to `idf.py`.
Use the normal `PANEL_BOARD_PROFILE` and isolated `SDKCONFIG` settings for the
target board. The handoff directory contains `models/c161.tflite`,
`models/binary40.tflite` and `models/history_expert.tflite`; packaging verifies
their original identities and emits a build manifest. Rejected paired-carrier
and multiscale exports are not accepted by this frozen research path.
Automatic OTA updates are disabled in this optional build.

This runtime uses the application's existing single capture/AEC/AGC/FIFO path.
One frontend processes native 16 kHz PCM. Each three-frame feature chunk is
quantized separately for each model. C161 has its own internal-variable arena;
B40 and history each own seven retained cache buffers. Logical signature names
resolve tensors instead of assuming positional input/output cache order. Cache
feedback checks metadata before copying, and any interpreter/cache failure
stops the combined detector without feeding a partial score to arbitration.
The original portable three-detector policy owns smoothing, rearm, cooldown and
priority; priming continues for 81,600 samples without a state reset at live join.

Each interpreter currently reserves 64 KiB of internal arena memory, with 16 KiB
additional C161 variable storage and separate explicit-cache allocations. These
are allocation budgets, not measured P4 peak usage or realtime guarantees.
Startup logs report used arenas, variable allocator metadata and retained bytes.
The C161 `variables_used` field describes its separate allocator metadata; the
resource tensor buffers allocated by the interpreter are included in `arena_used`.
The adapter logs internal heap free/largest-block sizes before and after create,
failure cleanup and destruction; its minimum-free value is a system lifetime
low-water mark, not this detector's isolated peak. Concurrent allocations can
affect these snapshots, so differing heap totals alone do not prove a leak.
Progress logs separately report combined model inference and complete detector
call timing (frontend included), free task stack and the count of calls over the
native 30 ms block duration. The overrun count is diagnostic and never changes
the input speed, event rules or FIFO. It excludes upstream I2S/AEC/AGC, downstream
call startup and the periodic timing-log emission itself. Any activation/error
log emitted inside processing is included. The model lifecycle releases all buffers
on initialization failure and normal destruction.

This is an opt-in research implementation, not a release candidate or evidence
of overall recognition quality. The original combination retains a disclosed
daily Nick conflict. Before reversible hardware trials, check model provenance,
the exact development-board identity/profile, available memory, the original
firmware backup and protected NVS/storage handling. No production board is
authorized. Record firmware/model hashes and actual rollback evidence.

Still required: full-panel P4 resources and timing, actual natural wake and
first-command retention, common daily/Bot-playback interference and long-call/
reconnect/audio stability. Host cache checks and target compilation do not prove
those requirements. Numerical self-tests diagnose ABI/cache faults; promotion
uses practical behavior and does not acquire a numerical-equality gate. Keep
the existing no-new-Mac-training/baseline/TTS/recording limits.

For the host check of the actual TFLM model wrapper, install the project's
managed dependencies and run:

```sh
python3 tests/parallel_wake/run.py --handoff /path/to/triple-handoff --output /new/test/output
```

This compiles the same model wrapper with the project's TFLM sources and Undefined
Behavior Sanitizer, invokes two independent sets using constructed feature chunks, and
injects failure into every wrapper allocation. It does not read PCM or claim
P4 equivalence, daily wake quality, or device memory peaks.
`--sanitizer address` is also available where the host ASan runtime works.

Add `--frontend` to exercise the same shared frontend and detector used in the
firmware. It generates in-memory zero/constructed PCM, primes exactly 81,600
samples, checks frame accounting at the live join and injects a bad cache ABI
to verify that a partial model failure stops the shared detector. It also
checks fresh initialization after every wrapper-allocation failure. This is a
software flow check; it adds no recorded or spoken acoustic baseline.

The frontend check also intercepts only the frontend C allocator calls, fails
each of its 17 allocation sites, and verifies that all tracked frontend and
wrapper storage is released. The standard managed-component patch set now
checks `filterbank.work` alongside the other allocations: version 1.2.3 omitted
this null check and could reach `FrontendReset` with a null work buffer. This
changes the low-memory failure path for both engines, without changing feature
processing when allocation succeeds. Configure the project first so its managed
patches are applied before running the host check. The unpatched failure at
allocation site 9 was reproduced under UBSan and retained separately from the
successful patched run. This does not establish P4 peak memory or heap behavior.
