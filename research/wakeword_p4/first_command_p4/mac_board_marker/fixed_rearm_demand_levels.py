"""Run both addresses at the same energy on six indoor microphone backgrounds."""
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import fixed_rearm_demand as long
import package_fixed_rearm_demand as pack

P, p4 = long.P, long.p4
WORK = long.WORK / "levels"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
OUT = long.OUT.with_stem(long.OUT.stem + "-levels")
PLAN = OUT.with_stem(OUT.stem + "-plan")
PROGRAMS = OUT.with_name("hye-nyx-willow-six-indoor-equal-levels.zip")
PROTOCOL = long.PROTOCOL.with_stem("fixed_rearm_demand_levels_protocol")


def prepare():
    import torch
    assert "4090" in torch.cuda.get_device_name()
    WORK.mkdir(parents=True, exist_ok=True)
    parent = P.read(long.PLAN)
    receipt = P.read(pack.RECEIPT)
    assert P.sha(pack.PACK) == receipt["archive_sha256"]
    rows = []
    files = {p:parent["files"][p] for p in parent["runtime"].values()}
    with ZipFile(pack.PACK) as z, ZipFile(WORK / PROGRAMS.name, "w") as dest:
        for row in parent["records"]:
            scene = row["label"]
            raw = z.read(scene + ".wav")
            assert p4.fixed.digest(raw) == row["sha256"]
            wave, sr = sf.read(BytesIO(raw), dtype="int16")
            assert sr == 16000 and P.digest(wave) == row["pcm_sha256"]
            window = slice(44800-23200, 44800)
            calls = {}
            for c in row["calls"]:
                assert c["clipped"] == 0
                calls[c["role"]] = wave[c["command"]-44800:c["end"]].astype(np.int32)
            n = len(calls["nyx"])
            assert len(calls["willow"]) == n and np.array_equal(calls["nyx"][44800:], calls["willow"][44800:])
            bg_end = row["background_samples"]
            noise = wave[bg_end-n:bg_end].astype(np.int32)
            addresses = {k:(v-noise)[window] for k,v in calls.items()}
            blank = calls["willow"].copy()
            blank[window] = noise[window]
            assert np.array_equal(blank[:window.start], noise[:window.start])
            target_levels = {"willow-level":long.rms_dbfs(addresses["willow"]), "zero-snr":long.rms_dbfs(noise[window])}
            conditions = [(mode, role) for mode in target_levels for role in ("nyx","willow")] + [("erased","erased")]
            for mode, role in conditions:
                program = blank.copy()
                actual = gain = None
                clipped = 0
                if role != "erased":
                    amplitude = torch.from_numpy(addresses[role]).to(device="cuda", dtype=torch.float64)
                    gain = 10**((target_levels[mode]-long.rms_dbfs(addresses[role]))/20)
                    cue = (amplitude*gain).round().to(dtype=torch.int32).cpu().numpy()
                    program[window] += cue
                    clipped = int(np.count_nonzero((program < -32768) | (program > 32767)))
                    program = program.clip(-32768,32767)
                    actual = long.rms_dbfs((program-noise)[window])
                    assert abs(actual-target_levels[mode]) < .01
                label = f"{scene}-{mode}-{role}"
                path = (WORK / (label+".wav")).relative_to(Path.cwd())
                sf.write(path, program.astype("<i2"), 16000, subtype="PCM_16")
                digest = P.sha(path)
                files[str(path)] = digest
                dest.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
                rows.append(dict(label=label, scene=scene, mode=mode, role=role, path=str(path),
                                 sha256=digest, pcm_sha256=P.digest(program.astype("<i2")), samples=n,
                                 start=window.start, command=window.stop, gain=gain, cue_rms_dbfs=actual,
                                 background_rms_dbfs=long.rms_dbfs(noise[window]), clipped=clipped))
        dest.writestr(ZipInfo("manifest.json"), json.dumps(dict(rate=16000, encoding="mono PCM16", records=rows,
                      license="CC BY-SA 3.0 Unported", source="https://doi.org/10.5281/zenodo.1227121",
                      authors="Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent",
                      modifications="Channel 1 final window mixed with two generated calls at matched levels."),indent=2)+"\n",
                      compress_type=ZIP_DEFLATED)
    (WORK / PROGRAMS.name).replace(PROGRAMS)
    for path in (SCRIPT, PROTOCOL, long.PLAN, pack.RECEIPT, PROGRAMS, long.wakes.room.official.SOURCE):
        files[str(path)] = P.sha(path)
    assert len(rows) == len(set(r["label"] for r in rows)) == 30
    P.save(PLAN, dict(records=rows, runtime=parent["runtime"], files=files, threshold_requests=["DEFAULT"],
                      gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), p4=parent["p4"]))


def native():
    rows = P.read(PLAN)["records"]
    assert P.read(long.WORK / "build.json") == P.read(PLAN)["p4"]
    for r in rows:
        for name in ("fixed", "c161"):
            p4.replay(name, r, WORK / name / r["label"])
        print(r["label"], flush=True)


def score():
    plan = P.read(PLAN)
    meta = P.read(WORK / "willow.json")
    assert P.sha(WORK / "willow.zip") == meta["archive_sha256"] and meta["plan_sha256"] == P.sha(PLAN)
    official = {r["label"]:r for r in meta["records"]}
    rows, files = [], {}
    with TemporaryDirectory(dir=WORK) as folder:
        folder = Path(folder)
        trace = OUT.with_suffix(".zip")
        with ZipFile(folder / trace.name, "w", ZIP_DEFLATED) as logs, ZipFile(WORK / "willow.zip") as archive:
            for row in plan["records"]:
                r = dict(row)
                w = official.pop(r["label"])
                assert w["pcm_sha256"] == row["pcm_sha256"]
                clocks = {"willow":[o for _,o in w["events"]]}
                r["willow"] = w["events"]
                for name, sha in w["members"].items():
                    key = w["label"] + "/" + name
                    data = archive.read(key)
                    assert p4.fixed.digest(data) == sha
                    logs.writestr("willow/"+key, data)
                    files["willow/"+key] = sha
                for name in ("fixed", "c161"):
                    cache = WORK / name / r["label"]
                    observation, log = p4.replay(name, r, cache)
                    for path in [log, *cache.glob("*.failed-*.log")]:
                        key = f"{name}/{r['label']}/{path.name}"
                        logs.write(path, key)
                        files[key] = P.sha(path)
                    r[name] = observation
                    clocks[name] = [e[2] for e in r[name]["events"]]
                r["timely"] = {k:[end for end in v if r["start"] <= end < r["command"]] for k,v in clocks.items()}
                rows.append(r)
        assert len(rows) == 30 and not official
        with ZipFile(folder / trace.name) as logs:
            assert len(files) == len(logs.namelist()) and not logs.testzip()
            assert all(p4.fixed.digest(logs.read(k)) == sha for k,sha in files.items())
        P.save(folder / OUT.name, dict(records=rows, files=files, archive_sha256=P.sha(folder / trace.name),
                  plan_sha256=P.sha(PLAN), additional_background_hours=0))
        (folder / trace.name).replace(trace)
        (folder / OUT.name).replace(OUT)


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage == "willow":
        long.wakes.willow_cases(PLAN, WORK, SCRIPT)
    elif stage == "case":
        official = long.wakes.room.official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    else:
        {"prepare":prepare, "native":native, "score":score}[stage]()
