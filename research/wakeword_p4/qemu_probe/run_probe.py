"""Read boot/timer markers from real P4 firmware under official QEMU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qemu", "nm", "elf", "flash", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("seconds must be in (0, 60]")
    args.output.mkdir(parents=True, exist_ok=True)
    symbols = subprocess.check_output([str(args.nm), str(args.elf)], text=True)
    address = next(int(row.split()[0], 16) for row in symbols.splitlines()
                   if row.endswith(" nyx_qemu_probe"))
    serial = args.output.resolve() / "serial.log"
    command = [str(args.qemu), "-machine", "esp32p4", "-display", "none",
               "-serial", "file:" + str(serial), "-monitor", "stdio", "-snapshot",
               "-drive", f"file={args.flash.resolve()},if=mtd,format=raw"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(args.seconds)
        monitor = f"stop\nxp /6wx 0x{address:x}\ninfo registers\nquit\n"
        output = process.communicate(monitor.encode(), timeout=10)[0]
    except BaseException:
        process.kill()
        process.communicate()
        raise
    (args.output / "monitor.raw.log").write_bytes(output)
    # HMP echoes commands using terminal editing escapes; retain response lines.
    clean = "\n".join(line for line in output.decode(errors="replace").splitlines()
                      if "\x1b" not in line)
    (args.output / "monitor.log").write_text(clean + "\n")
    words = []
    for line in clean.splitlines():
        if re.match(r"^[0-9a-fA-F]+:", line):
            words.extend(int(word, 16) for word in re.findall(r"0x[0-9a-fA-F]+", line))
    if len(words) != 6:
        raise RuntimeError(f"Expected six memory words, got {len(words)}")
    report = {
        "scope": "Boot/timer only; no audio, WakeNet or recognition evidence",
        "command": command, "wait_seconds": args.seconds,
        "exit_code": process.returncode, "symbol_address": hex(address),
        "memory_words": [hex(word) for word in words],
        "boot_pass": words[0] == 0x4E595842,
        "timer_pass": words[1] == 0x4E595854 and (words[2] | words[3]) != 0,
        "elapsed_us": words[2] | words[3] << 32, "chip_revision": words[4],
        "serial_output": serial.read_text(errors="replace"),
        "artifacts": {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "bytes": path.stat().st_size}
                      for name, path in (("elf", args.elf), ("flash", args.flash),
                                         ("qemu", args.qemu))},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if process.returncode or not report["boot_pass"] or not report["timer_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
