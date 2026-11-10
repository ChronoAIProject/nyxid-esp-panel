# Board-free P4 boot probe

This project tests application startup and a FreeRTOS delay/esp_timer interval.
It does not load audio or test wake recognition. Official QEMU release
`esp-develop-9.2.2-20261109` and ESP-IDF `v5.5.4` were used.

From the repository root, with the installed tools under ignored `.local/`:

```bash
export IDF_TOOLS_PATH="$PWD/.local/wakeword/idf-tools"
source .local/wakeword/esp-idf/export.sh
idf.py -C research/wakeword_p4/qemu_probe \
  -B "$PWD/.local/wakeword/qemu-probe-build" \
  -D SDKCONFIG="$PWD/.local/wakeword/qemu-probe-sdkconfig" \
  set-target esp32p4 build
```

In the generated build directory, merge the generated offsets (no hardware):

```bash
python -m esptool --chip esp32p4 merge_bin \
  --output qemu-flash.bin --fill-flash-size 4MB @flash_args
```

From the repository root:

```bash
python3 research/wakeword_p4/qemu_probe/run_probe.py \
  --qemu .local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32 \
  --nm .local/wakeword/idf-tools/tools/riscv32-esp-elf/esp-14.2.0_20260121/riscv32-esp-elf/bin/riscv32-esp-elf-nm \
  --elf .local/wakeword/qemu-probe-build/nyx_qemu_boot_probe.elf \
  --flash .local/wakeword/qemu-probe-build/qemu-flash.bin \
  --output .local/wakeword/qemu-boot-markers-v1
```

The runner uses a snapshot so the flash image is preserved. It captures UART to
a file, reads six memory words at the ELF's `nyx_qemu_probe` symbol, and requires
both application markers and positive elapsed time. The report includes binary
hashes. A previous bounded process captured only QEMU's host startup message;
that observation did not prove that the firmware had failed to boot. UART file
capture and actual memory inspection established startup, including a second run.

Results: `../results/4090-qemu-p4-boot-markers.json`. Passing this probe does not
establish compatibility with WakeNet's optimized inference instructions.
