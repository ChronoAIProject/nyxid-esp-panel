# Classic ESP32 Hi ESP runtime probe

ESP-SR 2.6.1 ships a classic ESP32 WakeNet9 library although its Hi ESP menu
is enabled only for S3/P4/S31. The public external-model directory setting packs
the unmodified `model/wakenet_model/wn9_hiesp` folder for this diagnostic.
The probe changes no engine arithmetic, and uses ordinary emulated PSRAM.
It is not a supported-device claim or a recognition benchmark.

Use official ESP-IDF v5.5.4 and Xtensa QEMU
`esp-develop-9.2.2-20261109` (x86_64 Linux). Install the esp32 IDF tools.
Copy this project to `.local/wakeword/wakenet-esp32-project`, append
`CONFIG_ESP_SR_EXTERNAL_MODEL_PATH="/absolute/path/to/parent"` to the copied
sdkconfig.defaults, with only the official `wn9_hiesp` directory in that parent.
The dependency is pinned to ESP-SR 2.6.1; build/model/library hashes are in
`../results/4090-esp32-hiesp-build.json`. Existing downloaded managed components
can be copied to avoid downloading them again; preserve all upstream content.

With the installed IDF environment exported:

```bash
idf.py -C .local/wakeword/wakenet-esp32-project \
  -B "$PWD/.local/wakeword/wakenet-esp32-build" set-target esp32 build
```

In that build directory merge, without flashing hardware:

```bash
python -m esptool --chip esp32 merge_bin \
  --output qemu-flash.bin --fill-flash-size 4MB @flash_args
```

From the repository root, substitute the installed Xtensa nm path:

```bash
python3 research/wakeword_p4/qemu_esp32_probe/run_probe.py \
  --qemu .local/wakeword/qemu-xtensa-release/qemu/bin/qemu-system-xtensa \
  --elf .local/wakeword/wakenet-esp32-build/nyx_wakenet_esp32_probe.elf \
  --nm /absolute/path/to/xtensa-esp32-elf-nm \
  --flash .local/wakeword/wakenet-esp32-build/qemu-flash.bin \
  --output /tmp/nyx-esp32-fresh-run
```

The runner requires model identity, ordinary 4MB PSRAM, 100 consecutive real
zero-PCM detect calls, and stage104 read from the ELF symbol through the QEMU
monitor. Stage104 follows destroy/deinit. Snapshot mode preserves the flash.
Output must be a fresh short path because the monitor uses an AF_UNIX socket.
Two independent runs passed. These are zero-input compatibility probes only:
no real-audio accuracy, negative-audio false-wake rate, P4 equivalence, hardware
latency, or Hey Nyx parity has been measured by this probe. Final audio is unused.
