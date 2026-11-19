"""Boot the P4 application and pair online keyword returns with the frozen host clocks."""
import argparse
import hashlib
import json
import re
import subprocess
import time
import zlib
from pathlib import Path

from prepare import WORK, sha, read

QEMU = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")


def boot(build, output):
    files = read(build / "flasher_args.json")["flash_files"]
    image = bytearray(b"\xff" * (16 << 20))
    for address, name in files.items():
        data = (build / name).read_bytes()
        start = int(address, 16)
        assert start + len(data) <= len(image)
        image[start:start+len(data)] = data
    flash = output / "flash.bin"
    flash.write_bytes(image)
    log = output / "uart.log"
    log.unlink(missing_ok=True)
    args = [str(QEMU), "-machine", "esp32p4", "-serial", f"file:{log.resolve()}", "-display", "none",
            "-monitor", "stdio", "-snapshot", "-drive", f"file={flash},if=mtd,format=raw"]
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as child:
        try:
            deadline = time.monotonic() + 300
            while child.poll() is None and time.monotonic() < deadline:
                text = log.read_text(errors="replace") if log.exists() else ""
                if re.search(r"LEX_DONE cases=-?\d+\n|Guru Meditation|abort\(\) was called|Rebooting\.\.\.", text):
                    break
                time.sleep(.4)
            _, err = child.communicate(b"quit\n", timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
    text = log.read_text(errors="replace")
    assert child.returncode == 0 and re.search(r"LEX_DONE cases=\d+\n", text), (err[-250:],text[-1200:])
    return files, log, text


def main(mode):
    mapped = mode in ("mapped", "compact")
    build = WORK / ("build" if mode == "read" else f"build-{mode}")
    output = WORK if mode == "read" else build
    files, log, text = boot(build, output)
    assert "LEX_DONE cases=62\n" in text
    source = (WORK / mode if mode in ("compact", "q8") else WORK) / "inputs.json"
    manifest = read(source)
    inputs = manifest["records"]
    mappings = {}
    storage = (build / "storage.bin").read_bytes()
    for name, offset_text, size_text in re.findall(r"LEX_MAP file=/vol/(\S+) offset=(\d+) length=(\d+)", text):
        offset, size = int(offset_text), int(size_text)
        digest = hashlib.sha256(storage[offset:offset+size]).hexdigest()
        assert name not in mappings and digest == manifest["files"][name]
        mappings[name] = dict(offset=offset, size=size, sha256=digest)
    music = {}
    if mode == "q8":
        assert "Task watchdog got triggered" not in text
        for size, crc, name in re.findall(r"LEX_MUSIC size=(\d+) crc=([0-9a-f]+) name=([^\n]+)", text):
            content = (Path("spiffs/music")/name).read_bytes()
            assert len(content) == int(size) and zlib.crc32(content) == int(crc,16)
            digest = hashlib.sha256(content).hexdigest()
            assert digest == manifest["files"]["music/"+name]
            music[name] = dict(bytes=len(content), crc32=crc, sha256=digest)
        assert len(music) == 3
        assert len(storage) == 0x200000 and files["0xe00000"] == "storage.bin"
        expanded = re.findall(r"LEX_UNPACK file=/model/(\S+) bytes=(\d+) expanded=(\d+)", text)
        assert sorted(name for name, _, _ in expanded) == ["am/means", "am/variances"]
        for name, size, decoded in expanded:
            assert int(decoded) == 838732 and (WORK/"q8/volume"/name).stat().st_size == int(size)
    assert set(mappings) == ({"am/mdef", "am/sendump"} if mapped else set())
    cases = []
    pattern = (r"LEX_CASE id=(\d+) count=(\d+) confirmed=(-?\d+) history_us=(-?\d+) "
               r"compute_us=(\d+) confirm_us=(-?\d+) max_chunk_us=(\d+)")
    matches = {}
    for nums in re.findall(r"LEX_MATCH id=(\d+) start=(\d+) end=(\d+) returned=(\d+)", text):
        idx, start, end, returned = map(int, nums)
        matches.setdefault(idx, []).append(dict(start=start,end=end,returned=returned))
    for nums in re.findall(pattern,text):
        idx, count, decision, history, total, confirmed_us, maximum = map(int, nums)
        assert idx == len(cases)
        src = inputs[idx]
        actual = matches.get(idx, [])
        expected = [{k:m[k] for k in ("start","end","returned")} for m in src["matches"]]
        assert len(actual) == count
        cases.append(dict(index=idx, cohort=src["cohort"], key=src["key"], proposal=src["proposal"],
                          matches=actual, matches_equal=actual==expected, host_confirmed=src["confirmed"],
                          confirmed=None if decision<0 else decision, history_us=history, compute_us=total,
                          modeled_confirm_us=None if confirmed_us<0 else confirmed_us, max_chunk_us=maximum))
    assert len(cases) == len(inputs) == 62
    assert all(c["matches_equal"] and c["confirmed"] == c["host_confirmed"] for c in cases)
    cost_rows = re.findall(r"LEX_COST id=(\d+) setup=(\d+) close=(\d+) blocks=([\d,]+)\n", text)
    if mapped or cost_rows:
        assert len(cost_rows) == len(cases)
        for (idx, setup, close, blocks), case in zip(cost_rows, cases, strict=True):
            assert int(idx) == case["index"]
            costs = list(map(int, blocks.split(",")))
            src = inputs[int(idx)]
            assert len(costs) == (src["end"] - src["start"] + 479) // 480
            assert sum(costs) + int(setup) == case["compute_us"] and max(costs) == case["max_chunk_us"]
            case.update(setup_us=int(setup), close_us=int(close), block_cost_us=costs)
    heaps = {stage:dict(zip(("ext","int","min_ext","min_int"), map(int, nums))) for stage,*nums in
             re.findall(r"LEX_HEAP stage=(\w+) ext=(\d+) int=(\d+) min_ext=(\d+) min_int=(\d+)", text)}
    assert set(heaps) == {"before","ready","decoded"}
    result = dict(physical_board=False, heap=heaps, init_us=int(re.search(r"LEX_INIT us=(\d+)",text)[1]),
        images={f:sha(build/f) for f in files.values()}, elf_sha256=sha(build/"nyx_lexical_probe.elf"),
        input_sha256=sha(source), qemu_sha256=sha(QEMU), sdkconfig_sha256=sha(WORK/("q8-sdkconfig" if mode == "q8" else "sdkconfig")),
        uart_sha256=sha(log), records=cases)
    result.update(mappings=mappings, stack_remaining_bytes=int(re.search(r"LEX_STACK remaining=(\d+)", text)[1]))
    if music:
        result["music_readback"] = music
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("P4 cases",len(cases),"segment clocks equal",sum(c["matches_equal"] for c in cases),
          "decisions equal",sum(c["confirmed"]==c["host_confirmed"] for c in cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mapped", dest="mode", action="store_const", const="mapped")
    group.add_argument("--compact", dest="mode", action="store_const", const="compact")
    group.add_argument("--q8", dest="mode", action="store_const", const="q8")
    main(parser.parse_args().mode or "read")
