"""Cold-boot the board receiver with the paired digital inputs and retain its UART."""
import argparse
from array import array
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from zipfile import ZipFile

from check_replay import BELLA, BELLA_SHA, PACK, PACK_SHA, RESULTS, mono


def cases():
    known = json.loads((RESULTS / "4090-c161-mac-board-marker-replay.json").read_text())
    for path, digest in ((PACK, PACK_SHA), (BELLA, BELLA_SHA)):
        assert sha256(path.read_bytes()).hexdigest() == digest
    with ZipFile(PACK) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for case, row in zip(manifest["cases"], known["captures"], strict=True):
            assert case["label"] == row["label"]
            yield row["label"], mono(archive, case["m"]), mono(archive, case["r"]), row["live_no_tx"], None
        yield ("self-restored-R", mono(archive, "digital/self-zero-R-M.wav"),
               mono(archive, "digital/g0-balanced-real-R-R.wav"), {"M": [], "C": [], "O": []}, None)
    with ZipFile(BELLA) as archive:
        for case in json.loads(archive.read("manifest.json"))["records"]:
            mic = mono(archive, case["file"])
            ref = bytes(len(mic))
            t = case["command_at"]
            yield "bella/"+case["file"], mic, ref, None, t
            yield "erased/"+case["file"], bytes(2*t)+mic[2*t:], ref, None, t


def payload(mic, ref):
    assert len(mic) == len(ref) and sys.byteorder == "little"
    count = (len(mic)//2 + 255)//256 * 256
    mr = array("h", [0]) * (2*count)
    mr[:len(mic):2] = array("h", mic)
    mr[1:len(ref):2] = array("h", ref)
    return struct.pack("<I", count) + mr.tobytes()


def boot(qemu, image, log):
    command = [qemu, "-machine", "esp32p4", "-display", "none", "-monitor", "stdio",
               "-serial", f"file:{log}", "-snapshot", "-drive", f"file={image},if=mtd,format=raw"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE) as proc:
        try:
            until = time.monotonic()+70
            while time.monotonic() < until:
                uart = log.read_bytes() if log.exists() else b""
                if b"BOARD PASS\r\n" in uart or b"Rebooting..." in uart or proc.poll() is not None:
                    break
                time.sleep(.1)
            _, err = proc.communicate(b"quit\n", timeout=5)
            assert proc.returncode == 0 and b"BOARD PASS\r\n" in uart, (err, uart[-1800:])
        finally:
            if proc.poll() is None:
                proc.kill()


def replay_cases(guest, sim_build, qemu, work, inputs):
    flash = bytearray([255]) * (4<<20)
    config = json.loads((sim_build / "flasher_args.json").read_text())
    for offset, name in config["flash_files"].items():
        at = int(offset, 16)
        data = (guest if name == config["app"]["file"] else sim_build/name).read_bytes()
        assert at+len(data) <= 2<<20
        flash[at:at+len(data)] = data
    logs = work.resolve() / "uart"
    logs.mkdir(exist_ok=True, parents=True)
    image = logs.parent / "image.bin"
    seen = set()
    rows = []
    for label, mic, ref, expected, command_at in inputs:
        block = payload(mic, ref)
        key = sha256(block).hexdigest()
        log = logs / (key+".log")
        if key not in seen:
            flash[2<<20:(2<<20)+len(block)] = block
            image.write_bytes(flash)
            log.unlink(missing_ok=True)
            boot(qemu, image, log)
            seen.add(key)
        uart = log.read_text()
        count, = struct.unpack_from("<I", block)
        assert f"BOARD begin prime=81600 bytes=13304 count={count}" in uart
        assert "watchdog got triggered" not in uart and "Guru Meditation" not in uart
        hits = [list(map(int, line.split())) for line in re.findall("^BOARD decision (.*)$", uart, re.M)]
        events = dict(M=[m for m, _, _ in hits if m], C=[c for _, c, _ in hits if c],
                      O=[hits[0][2]] if hits else [])
        if expected is not None:
            assert events == expected, (label, events, expected)
        rows.append(dict(label=label, input_sha256=key, uart_sha256=sha256(log.read_bytes()).hexdigest(),
                         events=events, command_at=command_at))
    return dict(guest_sha256=sha256(guest.read_bytes()).hexdigest(),
                qemu_sha256=sha256(qemu.read_bytes()).hexdigest(), records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guest", type=Path)
    parser.add_argument("sim_build", type=Path)
    parser.add_argument("qemu", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_cases(args.guest, args.sim_build, args.qemu, args.work, cases()), indent=2))


if __name__ == "__main__":
    main()
