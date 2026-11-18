"""Replay the five discrepant physical clocks and compare host intermediates with the instrumented UART."""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from fit_c161_history_gain_v1 import RESULTS, REF_KERNEL, read, sha
from make_fixture import WORK


def main():
    stem = "4090-c161-midlevel-p4-qemu-ansi-trace"
    local = WORK / "results"
    local.mkdir(exist_ok=True)
    source = local if (local / f"{stem}.json").is_file() else RESULTS
    path = source / f"{stem}-uart.log"
    report = read(source / f"{stem}.json")
    assert sha(path) == report["uart_sha256"]
    expected = {}
    for fields in re.findall(r"MIDLEVEL_TENSOR source=(\d+) step=(\d+) lane=(\d+) tensor=(\d+) bytes=(\d+) "
                             r"fnv=([a-f0-9]+) first=(\d+)", path.read_text()):
        src, step, lane, index, size = map(int, fields[:5])
        expected.setdefault((src, step, lane), {})[index] = (size, int(fields[5], 16), int(fields[6]))
    fixture = WORK / "midlevel.bin"
    assert sha(fixture) == report["fixture_sha256"]
    model = RESULTS / "4090-c161-midlevel-single-v1.tflite"
    assert sha(model) == report["model_sha256"]
    raw = fixture.read_bytes()
    counts, = struct.unpack_from("<I", raw)
    at, rows = 4, []
    for _ in range(counts):
        src, length = struct.unpack_from("<II", raw, at)
        at += 8
        record = np.frombuffer(raw, np.uint8, count=length*124, offset=at).reshape(-1,124)
        at += record.nbytes
        q = record[:,:120].copy().view(np.int8).reshape(-1,40)
        for lane in (0, 1):
            steps = [s for i, s, k in expected if i == src and k == lane]
            if not steps:
                continue
            it = REF_KERNEL(model_path=str(model), experimental_preserve_all_tensors=True)
            it.allocate_tensors()
            inp = it.get_input_details()[0]["index"]
            out = it.get_output_details()[0]["index"]
            repeats = 1 + (np.arange(len(q)) % 2) * lane
            available = repeats.cumsum()[2::3] // 3
            stream = np.repeat(q, repeats, axis=0)[:3*available[-1]].reshape(-1,3,40)
            done = 0
            for step in sorted(steps):
                for frame in stream[done:available[step-1]]:
                    it.set_tensor(inp, frame[None])
                    it.invoke()
                done = available[step-1]
                assert it.get_tensor(out).item() == record[step-1,120+lane]
                tensors = {}
                for idx, (size, p4_hash, first) in expected[src,step,lane].items():
                    data = it.get_tensor(idx).tobytes()
                    assert len(data) == size
                    h = 2166136261
                    for value in data:
                        h = ((h ^ value) * 16777619) & 0xffffffff
                    tensors[idx] = dict(size=size, host_fnv=h, p4_fnv=p4_hash, first_host=data[0], first_p4=first)
                rows.append(dict(source=src,step=step,lane=lane,tensors=tensors))
                print(src,step,lane,[i for i,t in tensors.items() if t["host_fnv"]!=t["p4_fnv"]])
    assert at == len(raw) and len(rows) == len(expected) == 5
    result = dict(uart_sha256=sha(path), fixture_sha256=sha(fixture), records=rows)
    (local / f"{stem}-comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
