"""Reconstruct 12 quantized physical streams using the published WAVs and independent traces."""
import hashlib
import io
import struct
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from audit_c161_history_paths_v1 import decode
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, frontend, read, sha

WORK = Path(".local/wakeword/c161-midlevel-p4")
STEM = "4090-c161-midlevel"


def main():
    source = read(RESULTS / "4090-c161-signed-forward-v1.json")
    archive = RESULTS / "4090-c161-signed-forward-inputs-v1.zip"
    plan = RESULTS / "4090-c161-signed-forward-execution-v1.json"
    assert sha(plan) == source["plan_sha256"] and sha(archive) == read(plan)["archive_sha256"]
    model = RESULTS / f"{STEM}-single-v1.tflite"
    assert sha(model) == read(RESULTS / f"{STEM}-training-v1.json")["reports"][0]["sha256"]
    scale, zero = REF_KERNEL(model_path=str(model)).get_input_details()[0]["quantization"]
    base = RESULTS / f"{STEM}-forward-scores-v1.npz"
    expanded = RESULTS / "4090-c161-stream-stretch-forward-scores-v1.npz"
    assert sha(base) == read(RESULTS / f"{STEM}-forward-v1.json")["scores_sha256"]
    assert sha(expanded) == read(RESULTS / "4090-c161-stream-stretch-forward-v1.json")["scores_sha256"]
    blob = bytearray(struct.pack("<I", 12))
    with ZipFile(archive) as waves, np.load(base) as normal, np.load(expanded) as stretch:
        count = 0
        for i, src in enumerate(source["records"]):
            if src["target_id"] != "nyx":
                continue
            raw = waves.read(Path(src["path"]).name)
            assert hashlib.sha256(raw).hexdigest() == src["sha256"]
            pcm, rate = sf.read(io.BytesIO(raw), dtype="int16")
            assert rate == 16000 and hashlib.sha256(memoryview(pcm)).hexdigest() == src["pcm_sha256"]
            values, clock = frontend(pcm)
            q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
            a, b = normal[str(i)][0], stretch[f"_{i}_midlevel"]
            rows = np.zeros((len(clock), 124), np.uint8)
            rows[:, :120] = q.reshape(-1, 120).view(np.uint8)
            for lane, codes in enumerate((a, b)):
                rows[:, 120 + lane] = codes
                for event in decode(codes, clock):
                    rows[event["step"] - 1, 122 + lane] = 1
            blob.extend(struct.pack("<II", i, len(rows)))
            blob.extend(rows.tobytes())
            count += 1
    assert count == 12
    WORK.mkdir(exist_ok=True, parents=True)
    target = WORK / "midlevel.bin"
    target.write_bytes(blob)
    print(target, len(blob), sha(target))


if __name__ == "__main__":
    main()
