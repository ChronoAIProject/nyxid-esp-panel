"""Run the shipped Hey Willow detector on the matched half-mixture PCM used by Hye Nyx."""
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve()
RESULTS = Path("research/wakeword_p4/results")
STEM = "4090-c161-speex-dialogues"
PLAN = RESULTS / f"{STEM}-willow-plan.json"
CASES = Path(".local/wakeword/c161-first-command-v5/lexical-echo/paired-dialogues/official")
sys.path.insert(0, str((RESULTS.parent / "qemu_hey_reference_v2").resolve()))
import run_willow_mac_matched_prime_v1 as native


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save(path, data):
    if path.exists():
        assert read(path) == data
    else:
        path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def prepare():
    import numpy as np
    import soundfile as sf
    report = RESULTS / f"{STEM}.json"
    before = read(report)
    comparison = RESULTS / "4090-c161-pocketsphinx-willow-plan.json"
    runtime = read(comparison)["runtime"]
    files = {str(path):sha(path) for path in [HERE, native.SOURCE, Path(native.__file__), comparison, report]}
    for name in runtime.values():
        assert sha(Path(name)) == read(comparison)["files"][name]
        files[name] = sha(Path(name))
    archive = RESULTS / f"{STEM}-uart.zip"
    assert sha(archive) == before["uart_archive_sha256"]
    records = before["willow_inputs"]
    assert len(records) == 16
    results = {r["key"]:r for r in before["records"]}
    with ZipFile(archive) as uart:
        for record in records:
            key, path = record["key"], Path(record["path"])
            if not path.exists():
                tracks = {}
                for mode in ("linear", "post"):
                    raw = uart.read(f"{key}-{mode}.log")
                    assert hashlib.sha256(raw).hexdigest() == results[key]["dsp"][mode]["uart_sha256"]
                    frames = re.findall(rb"SPEEX_PCM \d+ \d+ ([0-9a-f]+)", raw)
                    assert len(frames) == 1000
                    tracks[mode] = np.frombuffer(bytes.fromhex(b"".join(frames).decode()), dtype="<i2")
                dry = np.r_[np.zeros(160, dtype=np.int16), tracks["linear"][:-160]]
                half = ((dry.astype(np.int32)+tracks["post"].astype(np.int32))//2).astype("<i2")
                path.parent.mkdir(exist_ok=True, parents=True)
                sf.write(path, half, 16000, subtype="PCM_16")
            assert sha(path) == record["sha256"]
            files[str(path)] = record["sha256"]
    save(PLAN, dict(records=records, files=files, runtime=runtime, threshold_requests=["DEFAULT"]))


def case():
    native.PLAN = PLAN
    native.__file__ = str(HERE)
    native.case()


def run():
    import run_c161_willow_matched_scenes_v1 as runner
    runner.PLAN = PLAN
    runner.OUT = CASES
    runner.__file__ = str(HERE)
    runner.willow()


def summarize():
    import numpy as np
    records = read(PLAN)["records"]
    rows = []
    archive = RESULTS / f"{STEM}-willow-uart.zip"
    with ZipFile(CASES/"uart.zip", "w", ZIP_DEFLATED) as uart:
        for i, src in enumerate(records):
            folder = CASES/f"{i:02}"
            r = read(folder/"report.json")
            assert r["source_pcm_sha256"] == src["pcm_sha256"] and r["execution_sha256"] == sha(PLAN)
            assert r["model"] == "wn9_heywillow_tts" and r["stage"] == 104 and r["total_prime_samples"] == 81600
            assert r["actual_float32_bits"] == r["post_prime_getter_bits"] == "3f24dd2f"
            assert sha(folder/"raw-returns.bin") == r["raw_returns_sha256"] and sha(folder/"serial.log") == r["serial_sha256"]
            codes = np.fromfile(folder/"raw-returns.bin", np.uint8)
            assert len(codes) == r["input_calls"] and np.all(codes <= 2)
            events = (np.minimum((np.flatnonzero(codes==2)+1)*512,r["input_samples"])-49344).tolist()
            assert events == [e["sample_end"]-49344 for e in r["events"]]
            rows.append(dict(key=src["key"], events=events, hits=[t for t in events if src["start"]<=t<src["end"]]))
            for name in ("serial.log","raw-returns.bin","report.json","stage-monitor.txt","host.log"):
                uart.write(folder/name, f"{i:02}/{name}")
    with (CASES/"uart.zip").open("rb") as src, archive.open("xb") as dest:
        dest.write(src.read())
    (CASES/"uart.zip").unlink()
    save(RESULTS/f"{STEM}-willow.json",dict(plan_sha256=sha(PLAN),records=rows,uart_sha256=sha(archive)))


if __name__ == "__main__":
    {"prepare": prepare, "case": case, "run": run, "summarize": summarize}[sys.argv.pop(1)]()
