"""Boot the built P4 application with an ephemeral flash and save its actual UART."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=Path)
parser.add_argument("qemu", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("uart", type=Path)
a = parser.parse_args()
build = a.build.resolve()
args = json.loads((build / "flasher_args.json").read_text())
files = {}
with tempfile.TemporaryDirectory() as tmp:
    flash = Path(tmp) / "selftest.bin"
    image = bytearray(b"\xff" * (4 << 20))
    for address, name in args["flash_files"].items():
        raw = (build / name).read_bytes()
        start = int(address, 0)
        assert start + len(raw) <= len(image)
        image[start:start + len(raw)] = raw
        files[name] = hashlib.sha256(raw).hexdigest()
    flash.write_bytes(image)
    log = Path(tmp) / "serial.log"
    cmd = [str(a.qemu.resolve()), "-machine", "esp32p4", "-display", "none", "-serial", "file:" + str(log),
           "-monitor", "stdio", "-snapshot", "-drive", f"file={flash},if=mtd,format=raw"]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as child:
        deadline = time.monotonic() + 90
        match = None
        while time.monotonic() < deadline and child.poll() is None:
            if log.exists():
                match = re.search(rb"^(NYX_DUAL_(?:PASS|FAIL)[^\r\n]*)\r?\n", log.read_bytes(), re.M)
                if match or b"Guru Meditation Error" in log.read_bytes():
                    break
            time.sleep(.1)
        _, err = child.communicate(input=b"quit\n", timeout=15)
    a.uart.write_bytes(log.read_bytes() if log.exists() else b"")
    evidence = dict(target="esp32p4", physical_board=False, idf="5.5.4", images=files,
                    qemu_sha256=hashlib.sha256(a.qemu.read_bytes()).hexdigest(),
                    uart_sha256=hashlib.sha256(a.uart.read_bytes()).hexdigest(),
                    marker=match[1].decode() if match else None, exit_code=child.returncode)
    a.report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(evidence["marker"] or err.decode()[-1500:])
    if not match or not match[1].startswith(b"NYX_DUAL_PASS") or child.returncode:
        raise SystemExit(1)
