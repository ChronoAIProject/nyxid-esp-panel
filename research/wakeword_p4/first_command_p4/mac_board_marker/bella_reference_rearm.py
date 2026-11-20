"""Replay the paired electrical-reference variants on the frozen two-marker WAV."""
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import replay_bella_early_marker as history

early, fixed, P = history.early, history.fixed, history.P
playback = history.playback
WORK = Path(".local/wakeword/bella-reference-rearm").resolve()
STEM = early.ROOT / "results/4090-bella-reference-rearm"
PATCH = Path(__file__).with_name("reference-rearm-80ms.patch").resolve()
PROBE = PATCH.with_name("reference_rearm_probe.cpp")
IDF = Path(".local/wakeword/idf-tools/python_env/idf5.5_py3.11_env/bin/python").absolute()
SUB = Path("components/panel_services/fixed_marker")


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / "src"
    shutil.copytree(fixed.PRIMARY / SUB, source / SUB, dirs_exist_ok=True)
    service = source / SUB.parent
    shutil.copyfile(fixed.PRIMARY / SUB.parent / "panel_voice.c", service / "panel_voice.c")
    runtime = Path("components/panel_wake_runtime")
    if not (source / runtime).exists():
        (source / runtime).symlink_to(fixed.PRIMARY / runtime, target_is_directory=True)
    subprocess.run(["git", "apply", "--directory=" + str(source.relative_to(Path.cwd())), str(PATCH)], check=True)
    for name, src in (("original", fixed.PRIMARY), ("candidate", source)):
        subprocess.run(["c++", "-std=c++17", "-O2", "-I", src / SUB, PROBE, "-o", WORK / ("probe-" + name)], check=True)
    image = fixed.build(source, WORK / "guest", False, IDF)
    binary = (WORK / "guest/board.bin").read_bytes()
    meta = P.read(fixed.REPORT)
    assert P.sha(fixed.QEMU) == meta["qemu_sha256"] and P.sha(fixed.MODEL) == meta["model_sha256"]
    assert fixed.MODEL.stat().st_size == 60952 and binary.count(fixed.MODEL.read_bytes()) == 1
    (WORK / "guest.flash").write_bytes(image)
    headers = {p.name: P.sha(p) for p in (source / SUB).iterdir() if p.is_file()}
    P.save(WORK / "build.json", dict(guest_sha256=fixed.digest(binary), original_guest_sha256=meta["guest_sha256"]["dual"],
           model_sha256=meta["model_sha256"], qemu_sha256=meta["qemu_sha256"], patch_sha256=P.sha(PATCH),
           flash_sha256=P.sha(WORK/"guest.flash"), modified_headers=headers,
           original_headers={k:P.sha(fixed.PRIMARY / SUB / k) for k in headers}))


def trace(which, mic, ref):
    stream = np.stack((mic, ref), axis=1).astype("<i2").tobytes()
    raw = subprocess.check_output([WORK / ("probe-" + which)], input=stream, text=False)
    clocks = defaultdict(list)
    for line in raw.decode().splitlines():
        name, sample = line.split()
        clocks[name].append(int(sample))
    return dict(clocks)


def historical():
    original, _, _ = playback.candidate()
    cue, sr = sf.read(early.WAV, dtype="int16")
    assert sr == 16000 and P.sha(early.WAV) == P.read(early.FIT)["cue_sha256"]
    archived = {r["label"]: r for r in P.read(history.OUT)["records"]}
    for group in (playback.playback.panel, playback.playback.external):
        for key, row, mic, ref in group(original, cue):
            label = key.replace("/plus2", "/early")
            old = archived[label]
            assert old["mic_sha256"] == P.digest(mic) and old["ref_sha256"] == P.digest(ref)
            yield label, dict(group=row["group"], role=old["role"], clipped=row["clipped"],
                              start=row.get("start"), end=row.get("command_at")), mic, ref, old
    for r in P.read(early.REPORT)["records"]:
        mic, sr = sf.read(r["path"], dtype="int16")
        ref, rr = sf.read(r["reference_path"], dtype="int16")
        assert sr == rr == 16000 and P.digest(mic)==r["pcm_sha256"] and P.digest(ref)==r["ref_sha256"]
        yield "rate/"+r["label"], dict(group="rate", role=r["rate"], start=r["start"], end=r["end"], clipped=r["clipped"]), mic, ref, r["native"]


def collisions():
    _, external, _ = playback.candidate()
    panel, sr = sf.read(early.WAV, dtype="int16")
    assert sr == 16000 and len(panel) == len(external)
    rows = P.read(playback.playback.selfplay.OUT)["records"]
    for old in rows:
        if (old["reference"], old["version"]) != ("synchronous", "original"):
            continue
        geom = old["geometry"]
        (ir, _, checksum), _ = playback.playback.daily.tx.diag.room(geom)
        assert checksum == old["room_sha256"]
        p, e = (fftconvolve(x.astype(float), ir) * old["scale"] for x in (panel, external))
        for offset in (-800, -400, 0, 400, 800):
            start = 32000 + offset*16
            n = 64000 + len(p) + max(0, offset*16)
            ext = np.zeros(n)
            ext[start:start+len(e)] = e
            mix = ext.copy()
            mix[32000:32000+len(p)] += p
            mic = np.rint(mix).clip(-32768,32767).astype("<i2")
            ref = np.zeros(n, dtype="<i2")
            ref[32000:32000+len(panel)] = panel
            ext_pcm = np.rint(ext).clip(-32768,32767).astype("<i2")
            zeros = np.zeros_like(ref)
            ext_raw = trace("original", ext_pcm, zeros).get("raw", [])
            panel_pcm = np.zeros_like(ref)
            panel_pcm[32000:32000+len(p)] = np.rint(p).clip(-32768,32767).astype("<i2")
            panel_raw = trace("original", panel_pcm, zeros).get("raw", [])
            yield f"collision/{geom}/{offset}", dict(group="collision", geometry=geom, offset_ms=offset,
                  external_alone_raw=ext_raw, panel_alone_raw=panel_raw, start=start, end=start+len(external),
                  clipped=int(np.count_nonzero((np.rint(mix)<-32768)|(np.rint(mix)>32767)))), mic, ref, None


def replay(part):
    meta = P.read(WORK/"build.json")
    assert P.sha(WORK/"guest.flash") == meta["flash_sha256"] and P.sha(PATCH)==meta["patch_sha256"]
    version = bytearray((WORK/"guest.flash").read_bytes())
    original = history.wakes.fixed_image(P.read(fixed.REPORT), "dual")
    versions = {"candidate": version, "original": original}
    cache = {}
    for name, image in versions.items():
        cache[name] = WORK / part / (name+"-"+fixed.digest(image[:2<<20]))
        cache[name].mkdir(parents=True, exist_ok=True)
    rows, files = [], {}
    out = STEM.with_name(STEM.name+"-"+part+".json")
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        archive = out.with_suffix(".zip")
        with ZipFile(temp/archive.name, "w", ZIP_DEFLATED) as logs:
            for label, row, mic, ref, old in (historical if part=="historical" else collisions)():
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                row.update(label=label, input_sha256=fixed.digest(block), mic_sha256=P.digest(mic), ref_sha256=P.digest(ref))
                if old:
                    assert row["input_sha256"] == old["input_sha256"]
                    row["original"] = dict(events=old["events"])
                instrument = row["group"] in ("panel", "collision")
                for name in ("original", "candidate"):
                    if instrument:
                        row[name+"_probe"] = trace(name,mic,ref)
                    if name == "candidate" or not old:
                        for attempt in range(2):
                            try:
                                epochs, log, failed = fixed.boot(cache[name],versions[name],cache[name],block,1)
                                break
                            except AssertionError:
                                if attempt: raise
                        row[name], = epochs
                        assert b"BOARD PASS\r\n" in log.read_bytes()
                        for p in (log,*failed):
                            member = f"{name}/{label}.log" if p==log else f"failed/{name}/{label}/{p.name}"
                            logs.write(p,member)
                            files[member] = P.sha(p)
                    if instrument:
                        assert row[name+"_probe"].get("allowed",[]) == row[name]["events"]["C"], (label,name)
                rows.append(row)
                if len(rows)%40==0: print(part,len(rows),flush=True)
        assert len(rows)==len({r["label"] for r in rows})==(215 if part=="historical" else 20)
        P.save(temp/out.name, dict(records=rows,files=files,archive_sha256=P.sha(temp/archive.name),build=meta,
               sources={str(p):P.sha(p) for p in (Path(__file__),PATCH,PROBE,history.OUT,early.REPORT,early.WAV,
                          early.ROOT/"experiments/bella_reference_rearm_protocol.md")}))
        (temp/archive.name).replace(archive)
        (temp/out.name).replace(out)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        assert sys.argv[1] in ("historical","collisions")
        replay(sys.argv[1])
