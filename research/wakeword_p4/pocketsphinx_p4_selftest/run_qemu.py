"""Boot the P4 application and pair online keyword returns with the frozen host clocks."""
import json
import re
import subprocess
import time
from pathlib import Path

from prepare import WORK, sha, read

QEMU = Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")


def main():
    build = WORK / "build"
    files = read(build / "flasher_args.json")["flash_files"]
    image = bytearray(b"\xff" * (16 << 20))
    for address, name in files.items():
        data = (build / name).read_bytes()
        start = int(address, 16)
        assert start + len(data) <= len(image)
        image[start:start+len(data)] = data
    flash = WORK / "flash.bin"
    flash.write_bytes(image)
    log = WORK / "uart.log"
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
    assert child.returncode == 0 and "LEX_DONE cases=62" in text, (err[-250:],text[-1200:])
    source = WORK / "inputs.json"
    inputs = read(source)["records"]
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
    heaps = {stage:dict(zip(("ext","int","min_ext","min_int"), map(int, nums))) for stage,*nums in
             re.findall(r"LEX_HEAP stage=(\w+) ext=(\d+) int=(\d+) min_ext=(\d+) min_int=(\d+)", text)}
    assert set(heaps) == {"before","ready","decoded"}
    result = dict(physical_board=False, heap=heaps, init_us=int(re.search(r"LEX_INIT us=(\d+)",text)[1]),
        images={f:sha(build/f) for f in files.values()}, elf_sha256=sha(build/"nyx_lexical_probe.elf"),
        input_sha256=sha(source), qemu_sha256=sha(QEMU), sdkconfig_sha256=sha(WORK/"sdkconfig"),
        uart_sha256=sha(log), records=cases)
    (WORK/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("P4 cases",len(cases),"segment clocks equal",sum(c["matches_equal"] for c in cases),
          "decisions equal",sum(c["confirmed"]==c["host_confirmed"] for c in cases))


if __name__ == "__main__":
    main()
