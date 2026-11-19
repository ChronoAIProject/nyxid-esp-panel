"""Replay all distinct name grammar segments on P4 and compare words, frames, memory and time."""
from collections import defaultdict
from pathlib import Path
import re
import zlib

from prepare import WORK, read, sha
from run_qemu import QEMU, boot

RESULTS = Path("research/wakeword_p4/results")
ROOT = WORK / "addresses-graph"
BUILD = WORK / "build-addresses"


def main():
    host = RESULTS/"4090-c161-address-grammar-graph-windows.json"
    reference = read(host)
    assert sha(ROOT/"cases/cases.bin") == reference["cases_sha256"]
    files, log, text = boot(BUILD, BUILD)
    assert "Task watchdog got triggered" not in text and "Guru Meditation" not in text
    music = {}
    for count, crc, name in re.findall(r"LEX_MUSIC size=(\d+) crc=([0-9a-f]+) name=([^\n]+)",text):
        path = Path("spiffs/music")/name
        assert path.stat().st_size == int(count) and zlib.crc32(path.read_bytes()) == int(crc,16)
        music[name] = sha(path)
    weights = ROOT/"volume/am/sendump"
    assert len(music) == 3 and sha(weights) == reference["packed_sendump_sha256"]
    assert int(re.search(r"FSG_SENDUMP bytes=(\d+)",text)[1]) == weights.stat().st_size
    assert files["0xe00000"] == "storage.bin" and (BUILD/"storage.bin").stat().st_size == 0x200000
    segments = defaultdict(list)
    for idx, a, b, word in re.findall(r"FSG_SEG id=(\d+) begin=(\d+) end=(\d+) word=([^\n]+)",text):
        segments[int(idx)].append(dict(word=word,begin=int(a),end=int(b)))
    rows = []
    cases = re.findall(r"FSG_CASE id=(\d+) samples=(\d+) us=(\d+) text=([^\n]*)",text)
    for old, (idx,n,us,hyp) in zip(reference["windows"],cases,strict=True):
        assert len(rows) == int(idx) == old["index"] and int(n) == old["samples"]
        rows.append(dict(index=int(idx), samples=int(n), compute_us=int(us), text=hyp,
                         text_equal=hyp==old["q8_text"], segments_equal=segments[int(idx)]==old["q8_segments"],
                         segments=segments[int(idx)], input_sha256=old["sha256"]))
    assert f"LEX_DONE cases={len(rows)}\n" in text
    heaps = {stage:dict(zip(("ext","int","min_ext","min_int"),map(int,nums),strict=True)) for stage,*nums in
             re.findall(r"LEX_HEAP stage=(\w+) ext=(\d+) int=(\d+) min_ext=(\d+) min_int=(\d+)",text)}
    assert set(heaps) == {"before","ready","decoded"}
    import json
    out = dict(physical_board=False, reference_sha256=sha(host), model_sha256=reference["model_sha256"],
               heaps=heaps, init_us=int(re.search(r"LEX_INIT us=(\d+)",text)[1]), music=music,
               flash_files={n:sha(BUILD/n) for n in files.values()}, elf_sha256=sha(BUILD/"nyx_lexical_probe.elf"),
               qemu_sha256=sha(QEMU), sdkconfig_sha256=sha(ROOT/"sdkconfig"), uart_sha256=sha(log),
               embedded_sendump_bytes=weights.stat().st_size, application_bytes=(BUILD/"nyx_lexical_probe.bin").stat().st_size,
               records=rows)
    (BUILD/"addresses.json").write_text(json.dumps(out,indent=2)+"\n")
    print("windows",len(rows),"same text",sum(r["text_equal"] for r in rows),"same segmentation",sum(r["segments_equal"] for r in rows))


if __name__ == "__main__":
    main()
