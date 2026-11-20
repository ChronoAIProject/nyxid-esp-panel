"""Replay a whole mono recording through the unchanged P4 listeners, with zero reference."""
from pathlib import Path
import re
import struct
import subprocess
import time
import zlib

import soundfile as sf

import eight_local_fixed_rearm as rearm
from build_rev1_dev import REFERENCE_BARGE, check_sources

P, fixed = rearm.P, rearm.fixed
WORK = Path(".local/wakeword/fixed-rearm-demand").resolve()


def build():
    builds = {}
    for name, board in (("fixed", rearm.WORK / "src"), ("c161", rearm.prior.dual.WORK / "boards-src")):
        standalone = name == "fixed"
        check_sources(board, extra_patches=(rearm.FIXED, *REFERENCE_BARGE) if standalone else REFERENCE_BARGE)
        dest = WORK / name
        flash = fixed.build(board.resolve(), dest, standalone, rearm.prior.dual.test.IDF, resident_mono=True)
        (dest / "flash.bin").write_bytes(flash)
        builds[name] = dict(guest=P.sha(dest / "board.bin"), elf=P.sha(dest / "board.elf"), flash=fixed.digest(flash))
    P.save(WORK / "build.json", dict(guests=builds, qemu=P.sha(fixed.QEMU), model=P.sha(fixed.MODEL)))


def replay(name, row, folder):
    build = P.read(WORK / "build.json")
    flash = WORK / name / "flash.bin"
    assert P.sha(flash) == build["guests"][name]["flash"] and P.sha(fixed.QEMU) == build["qemu"]
    assert P.sha(Path(row["path"])) == row["sha256"]
    pcm, rate = sf.read(row["path"], dtype="int16")
    raw = pcm.tobytes()
    samples = len(pcm)
    assert pcm.ndim == 1 and rate == 16000 and fixed.digest(raw) == row["pcm_sha256"] and samples == row["samples"]
    crc = zlib.crc32(raw)
    content = struct.pack("<II", samples, crc) + raw + bytes(2 * (-samples % 256))
    folder.mkdir(parents=True, exist_ok=True)
    stamp = fixed.digest(bytes.fromhex(build["qemu"] + build["guests"][name]["flash"]) + content)
    log = folder / (stamp + ".log")
    if not log.exists() or b"BOARD PASS\r\n" not in log.read_bytes():
        if log.exists():
            log.rename(folder / (stamp + ".failed-" + P.sha(log) + ".log"))
        payload = folder / "input.bin"
        payload.write_bytes(content)
        command = [fixed.QEMU, "-machine", "esp32p4", "-m", "16M", "-display", "none", "-monitor", "stdio",
                   "-serial", f"file:{log}", "-snapshot", "-drive", f"file={flash},if=mtd,format=raw",
                   "-device", f"loader,file={payload},addr=0x48400000,force-raw=on"]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as proc:
            try:
                deadline = time.monotonic() + max(90, samples / 16000)
                while time.monotonic() < deadline and proc.poll() is None:
                    data = log.read_bytes() if log.exists() else b""
                    if any(token in data for token in (b"BOARD PASS\r\n", b"Rebooting...", b"Guru Meditation")):
                        break
                    time.sleep(.25)
                _, err = proc.communicate(b"quit\n", timeout=8)
                if proc.returncode or b"BOARD PASS\r\n" not in log.read_bytes():
                    (folder / (stamp + ".failed-" + fixed.digest(err) + ".stderr.log")).write_bytes(err)
                    raise AssertionError((err, log))
            finally:
                if proc.poll() is None:
                    proc.kill()
        payload.unlink()
    text = log.read_text()
    assert not any(token in text for token in ("Guru Meditation", "watchdog got triggered"))
    assert re.findall(r"^PROFILE input (\d+) (\w+)$", text, re.M) == [(str(samples), f"{crc:08x}")]
    (count, size), = re.findall(r"^PROFILE time 0 (\d+) (\d+) ", text, re.M)
    assert int(count) == (samples + 255) // 256 * 256 and int(size) == (9488 if name == "fixed" else 13304)
    events = [list(map(int, t)) for t in re.findall(r"^PROFILE event 0 (\d+) (\d+) (\d+)$", text, re.M)]
    return dict(input_crc32=f"{crc:08x}", consumed_samples=int(count), events=events), log
