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
Startup logs report used arenas and retained bytes; progress logs report combined
inference timing and free task stack. The model lifecycle releases all buffers
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
