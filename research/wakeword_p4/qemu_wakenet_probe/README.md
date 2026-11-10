# P4 WakeNet runtime compatibility probe

**Simulation only. Do not deploy this external-memory configuration to a board.**
The model and Espressif inference archives are unchanged. No audio fixture or
final test set is scored. Zero PCM is only intended to exercise runtime code.

The exact registry component ESP-SR 2.6.1 matches the main project's component
hash. `dependencies.lock` pins its resolved dependencies. Hi ESP `wn9_hiesp` is
packed into the custom model partition by the component's official build logic.

Build a copy under ignored `.local/` so component downloads and sdkconfig stay
out of the source directory:

```bash
cp -r research/wakeword_p4/qemu_wakenet_probe .local/wakeword/wakenet-qemu-repro
export IDF_TOOLS_PATH="$PWD/.local/wakeword/idf-tools"
source .local/wakeword/esp-idf/export.sh
idf.py -C .local/wakeword/wakenet-qemu-repro \
  -B "$PWD/.local/wakeword/wakenet-qemu-repro-build" set-target esp32p4 build
```

Merge `@flash_args` from the generated build directory as in the boot-probe
README, then run official QEMU `esp-develop-9.2.2-20261109` with machine `esp32p4`,
`-snapshot`, UART to a file, and the merged image as a raw MTD drive. The inherited
application target is `nyx_qemu_boot_probe`; its main is this WakeNet probe.

Two simulation adaptations are explicit in main.c: register part of the QEMU
plain external RAM window after a sparse page check, and attempt to activate
IDF's standard lazy FPU path before metadata parsing. These do not implement
PSRAM hardware initialization or emulate missing PIE arithmetic instructions.

Observed outcome: model partition and `wn9_hiesp` discovery succeed; creation
traps in `dl_esp32p4_sr_memcpy` at `esp.ld.128.usar.ip`. **No detect call
completed.** The ROM `frm` CSR fault also occurred once in this run. The current
emulator cannot supply an operational Hi ESP comparison using this binary.
Details, hashes and exceptions are in
`../results/4090-qemu-p4-wakenet-create-probe.json`.
