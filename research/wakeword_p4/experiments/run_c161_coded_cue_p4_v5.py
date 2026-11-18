"""Boot the v5 standalone P4 application and verify both far-field command notifications."""
import json
from pathlib import Path
import re
import subprocess
import time

from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM="4090-c161-coded-cue-p4-v5"
BUILD=Path(".local/wakeword/c161-p4-coded-selftest-v5/build").resolve()
QEMU=Path(".local/wakeword/qemu-p4-release/qemu/bin/qemu-system-riscv32")


def main():
    layout=read(BUILD/"flasher_args.json")["flash_files"]
    flash=bytearray(b"\xff"*(4<<20))
    images={}
    for offset,name in layout.items():
        path=BUILD/name
        blob=path.read_bytes()
        start=int(offset,16)
        assert start+len(blob)<=len(flash) and all(b==255 for b in flash[start:start+len(blob)])
        flash[start:start+len(blob)]=blob
        images[name]=dict(sha256=sha(path),length=len(blob),offset=start)
    image=BUILD/"coded-flash.bin"
    image.write_bytes(flash)
    uart=BUILD/"v5-uart.log"
    uart.unlink(missing_ok=True)
    cmd=[str(QEMU),"-machine","esp32p4","-display","none","-serial",f"file:{uart}","-monitor","stdio",
         "-snapshot","-drive",f"file={image},if=mtd,format=raw"]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE) as emulator:
        try:
            deadline=time.monotonic()+50
            while time.monotonic()<deadline:
                log=uart.read_bytes() if uart.exists() else b""
                if b"\n" in log.partition(b"NYX_CUE_SELFTEST ")[2] or emulator.poll() is not None:
                    break
                time.sleep(.15)
            _,errors=emulator.communicate(b"quit\n",timeout=8)
        finally:
            if emulator.poll() is None:
                emulator.kill()
    assert emulator.returncode==0,errors
    text=uart.read_text(errors="replace")
    marker=re.search(r"NYX_CUE_SELFTEST PASS hits=2 state=(\d+) max_block_us=(\d+) blocks_ge_30ms=0 min_internal_heap=(\d+)",text)
    events=[int(t) for t in re.findall(r"NYX_CUE_EVENT sample=(\d+) bits=8",text)]
    assert events==[102932,278036] and marker, text[-1400:]
    dest=RESULTS/f"{STEM}-uart.log"
    with dest.open("xb") as f:
        f.write(uart.read_bytes())
    save(RESULTS/f"{STEM}.json",dict(target="esp32p4",idf="v5.5.4",physical_board=False,
         uart_sha256=sha(dest),qemu_sha256=sha(QEMU),elf_sha256=sha(BUILD/"nyx_cue_selftest.elf"),images=images,
         flash_sha256=sha(image),fixture_sha256=sha(RESULTS/"4090-c161-coded-cue-v5-command-far.wav"),
         prime=81600,gap=32163,events=events,state_bytes=int(marker[1]),max_block_us=int(marker[2]),
         min_internal_heap=int(marker[3]),blocks_ge_30ms=0))
    print(marker[0],flush=True)


if __name__=="__main__":
    main()
