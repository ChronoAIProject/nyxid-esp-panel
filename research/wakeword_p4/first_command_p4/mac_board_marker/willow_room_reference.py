"""Compare the two board listeners on the exact microphone inputs in the Mac room packs."""
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from check_willow_feeder import build_feeder

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT / "results"
ASSETS = Path(".local/wakeword/c161-bella-weak-speaker-gain")
WORK = Path(".local/wakeword/c161-willow-board/room")
PLAN = RESULTS / "4090-hey-willow-p4-rev1-room-plan.json"
OUT = RESULTS / "4090-hey-willow-p4-rev1-room.json"
TRACE = OUT.with_suffix(".zip")
SCRIPT = Path(__file__)
ROOM = RESULTS / "4090-c161-bella-weak-speaker-gain-room-audio.json"
CONTROL = RESULTS / "4090-c161-bella-072-room-playback.json"
NYX = RESULTS / "4090-c161-bella-room-handoff-replay.json"
GAIN = RESULTS / "4090-c161-bella-weak-speaker-gain.json"
DAILY = RESULTS / "4090-c161-bella-daily-willow.json"
sys.path.insert(0, str((ROOT / "qemu_hey_reference_v2").resolve()))
import run_willow_mac_matched_prime_v1 as official


def digest(raw):
    return sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def inputs():
    for source in (ROOM, CONTROL):
        meta = read(source)
        archive = ASSETS / meta["archive"]
        assert digest(archive.read_bytes()) == meta["archive_sha256"]
        with ZipFile(archive) as z:
            if source == ROOM:
                cases = [(p, s["command_at"], s["reference_sha256"]) for s in meta["scenes"] for p in s["pairs"]]
            else:
                cases = [(p, meta["command_at"], p["reference_sha256"]) for p in meta["records"]]
            for entry, at, ref_sha in cases:
                label = entry["digital"].removeprefix("digital/").removesuffix(".wav")
                with wave.open(BytesIO(z.read(entry["digital"]))) as wav:
                    assert wav.getparams()[:3] == (2, 2, 16000)
                    mr = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, 2)
                mic = mr[:, 0].tobytes()
                assert digest(mic) == entry["mic_sha256"] and digest(mr[:, 1].tobytes()) == ref_sha
                yield label, mic, ref_sha, at


def past_runs(stack):
    prior = {}
    for source, path in ((GAIN, ASSETS / read(GAIN)["archive"]), (DAILY, DAILY.with_suffix(".zip"))):
        assert digest(path.read_bytes()) == read(source)["archive_sha256"]
        archive = stack.enter_context(ZipFile(path))
        for key in archive.namelist():
            if key.endswith("/willow-report.json"):
                rpt = json.loads(archive.read(key))
                prefix = key.removesuffix("report.json")
                prior[rpt["source_pcm_sha256"]] = (archive, prefix, rpt)
    return prior


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    old = read(GAIN.with_name(GAIN.stem + "-plan.json"))
    files = {v:old["files"][v] for v in old["runtime"].values()}
    for p in (SCRIPT, Path(official.__file__), official.SOURCE, ROOM, CONTROL, GAIN, DAILY, NYX):
        files[str(p)] = digest(p.read_bytes())
    with ExitStack() as stack:
        known = past_runs(stack)
        rows = []
        for label, mic, _, _ in inputs():
            if digest(mic) in known:
                continue
            path = WORK / (label.replace("/", "-") + ".wav")
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(mic)
            rows.append(dict(label=label, path=str(path), sha256=digest(path.read_bytes()),
                             pcm_sha256=digest(mic), samples=len(mic)//2))
            files[str(path)] = rows[-1]["sha256"]
    plan = dict(records=rows, runtime=old["runtime"], files=files, threshold_requests=["DEFAULT"],
                scope="Missing official Willow returns for the same 35 consumed digital room inputs at a continuous 81600-sample prime.")
    if PLAN.exists():
        assert read(PLAN) == plan
    else:
        save(PLAN, plan)
    print(len(rows), "fresh official runs")


def run():
    plan = read(PLAN)
    for path, expected in plan["files"].items():
        assert digest(Path(path).read_bytes()) == expected
    for i, item in enumerate(plan["records"]):
        final = WORK / f"run-{i}"
        if final.exists():
            assert read(final / "report.json")["execution_sha256"] == digest(PLAN.read_bytes())
            continue
        with TemporaryDirectory(prefix="willow-") as tmp:
            out = Path(tmp) / "run"
            cmd = [sys.executable, SCRIPT, "case", "--model", "willow", "--threshold", "DEFAULT",
                   "--record-index", str(i), "--output", out, "--timeout-seconds", "70"]
            for k, v in plan["runtime"].items():
                cmd.extend(["--"+k,v])
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            assert read(out / "report.json")["source_pcm_sha256"] == item["pcm_sha256"]
            (out / "input.pcm").unlink()
            shutil.move(out, final)


def score(board):
    nyx = read(NYX)["records"]
    plan = read(PLAN)
    fresh = {r["pcm_sha256"]:WORK/f"run-{i}" for i,r in enumerate(plan["records"])}
    rows = []
    with ExitStack() as stack, TemporaryDirectory() as tmp, ZipFile(TRACE, "w", ZIP_DEFLATED) as trace:
        old = past_runs(stack)
        exe = build_feeder(board, Path(tmp))
        for label, mic, ref_sha, at in inputs():
            key = digest(mic)
            if key in old:
                archive, prefix, rpt = old[key]
                raw, uart = (archive.read(prefix+name) for name in ("raw-returns.bin", "serial.log"))
            else:
                folder = fresh[key]
                rpt = read(folder/"report.json")
                raw, uart = ((folder/name).read_bytes() for name in ("raw-returns.bin", "serial.log"))
            assert rpt["source_pcm_sha256"] == key and digest(raw) == rpt["raw_returns_sha256"]
            assert digest(uart) == rpt["serial_sha256"] and rpt["stage"] == 104
            assert rpt["model"] == "wn9_heywillow_tts" and rpt["actual_float32_bits"] == "3f24dd2f"
            assert rpt["post_prime_getter_bits"] == "3f24dd2f" and rpt["mode"] == "DEFAULT"
            assert rpt["total_prime_samples"] == 81600 and not rpt["additional_priming_events"]
            assert all(v<=0 for _,v in rpt["prime_returns"])
            returns = bytes(v+1 for _,v in rpt["prime_returns"])+raw
            pcm = mic + bytes(rpt["final_padding_samples"]*2)
            ends = subprocess.check_output([exe,str(len(pcm)//2),"256"], input=pcm+returns)
            events = [tuple(map(int,line.split())) for line in ends.splitlines()]
            assert [model for model,_ in events] == [e["relative_sample"] for e in rpt["real_events"]]
            assert nyx[label]["command_at"] == at
            rows.append(dict(label=label, mic_sha256=key, ref_sha256=ref_sha, command_at=at,
                             nyx=nyx[label]["baseline"]["events"], willow_model=[m for m,_ in events],
                             willow_first_read=events[0][1] if events else None, fresh=key in fresh,
                             uart_sha256=digest(uart), raw_returns_sha256=digest(raw)))
            trace.writestr(label+"/serial.log",uart)
            trace.writestr(label+"/raw-returns.bin",raw)
            trace.writestr(label+"/report.json",json.dumps(rpt,indent=2)+"\n")
    assert len(rows)==len({r["label"] for r in rows})==35
    save(OUT,dict(records=rows,archive_sha256=digest(TRACE.read_bytes()),plan_sha256=digest(PLAN.read_bytes()),
                  scope="C161 fixed-marker P4 digital replay and archived or fresh official Xtensa Willow DEFAULT "
                        "returns replayed by the rev1 host-compiled feeder. No new acoustic exposure or physical P4 run."))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        official.PLAN = PLAN
        official.__file__ = str(SCRIPT)
        official.case()
    elif action == "score":
        score(Path(sys.argv[1]))
    else:
        {"prepare":prepare,"run":run}[action]()
