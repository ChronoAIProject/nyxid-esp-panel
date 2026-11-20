"""Compare external indoor recordings on the two Nyx receivers and official Hey Willow."""
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import fixed_rearm_demand_p4 as p4
import eight_local_player_079 as player
import check_address_gain_wakes as wakes

P, WORK = p4.P, p4.WORK
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
OUT = player.OUT.with_name("4090-fixed-rearm-demand.json")
PLAN = OUT.with_stem(OUT.stem + "-plan")
PROTOCOL = player.PROTOCOL.with_name("fixed_rearm_demand_protocol.md")
SCENES = ["DLIVING", "DKITCHEN", "DWASHING", "OMEETING", "PCAFETER", "PRESTO"]
QUIET = OUT.with_name("c161-willow-077-quiet-pair.zip")
QUIET_REPORT = OUT.with_name("4090-hey-willow-p4-rev1-quiet.json")
ISSUED = OUT.with_name("4090-early-groups-eight-local-mac.json")


def render(source):
    import torch
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    scene = source["scene"]
    wav = WORK / (scene + ".wav")
    assert source["member"] == scene + "/ch01.wav" and P.sha(wav) == source["member_sha256"]
    background, rate = sf.read(wav, dtype="int16")
    assert rate == 16000 and background.ndim == 1 and sf.info(wav).subtype == "PCM_16"
    assert P.sha(QUIET) == P.read(QUIET_REPORT)["audio_sha256"]
    with ZipFile(QUIET) as archive:
        import json
        meta = json.loads(archive.read("manifest.json"))
        sources = {}
        for role, filename in (("nyx", "erased.wav"), ("willow", "willow.wav")):
            raw = archive.read(filename)
            assert p4.fixed.digest(raw) == meta["files"][filename]["sha256"]
            sources[role], sr = sf.read(BytesIO(raw), dtype="int16")
            assert sr == rate and P.digest(sources[role]) == meta["files"][filename]["pcm_sha256"]
    end = meta["command_at"]
    cue = p4.rearm.prior.shifts.candidate.WAV
    assert P.sha(cue) == P.read(ISSUED)["files"]["af_bella.wav"]
    bella, sr = sf.read(cue, dtype="int16")
    assert sr == rate and not sources["nyx"][:end].any()
    assert np.array_equal(sources["nyx"][end:], sources["willow"][end:])
    starts = dict(nyx=end-len(bella), willow=int(np.flatnonzero(sources["willow"][:end])[0]))
    sources["nyx"][starts["nyx"]:end] = bella
    roles = ("nyx", "willow") if SCENES.index(scene) % 2 == 0 else ("willow", "nyx")
    parts, spans = [background], []
    at = len(background)
    for i, role in enumerate(roles):
        if i:
            parts.append(np.zeros(rate, dtype="<i2"))
            at += rate
        speech = sources[role]
        noise = background[-len(speech):]
        def gpu(a):
            return torch.from_numpy(a.astype(np.int32)).to("cuda")
        values = gpu(noise) + gpu(speech)
        mixed = values.clamp(-32768, 32767).short().cpu().numpy()
        np.testing.assert_array_equal(mixed, (noise.astype(np.int32)+speech).clip(-32768,32767))
        parts.append(mixed)
        spans.append(dict(role=role, start=at+starts[role], command=at+end, end=at+len(speech),
                          clipped=int(((values < -32768) | (values > 32767)).sum().item())))
        at += len(speech)
    mic = np.concatenate(parts)
    assert len(mic) == at
    path = (WORK / (scene + "-calls.wav")).relative_to(Path.cwd())
    sf.write(path, mic, rate, subtype="PCM_16")
    return dict(label=scene, path=str(path), sha256=P.sha(path), pcm_sha256=P.digest(mic), samples=at,
                background_samples=len(background), background_pcm_sha256=P.digest(background), calls=spans,
                background_rms_dbfs=float(10*np.log10(np.mean((background.astype(float)/32768)**2))), source=source)


def prepare():
    sources = P.read(WORK / "sources.json")
    assert [r["scene"] for r in sources] == SCENES
    rows = [render(source) for source in sources]
    assert len(set(r["background_pcm_sha256"] for r in rows)) == len(rows)
    base = P.read(player.PLAN)
    deps = [SCRIPT, PROTOCOL, Path(p4.__file__).relative_to(Path.cwd()), QUIET_REPORT, ISSUED,
            wakes.room.official.SOURCE, p4.fixed.HERE / "profile_fixed_sound.c", Path(p4.fixed.__file__)]
    files = {p:base["files"][p] for p in base["runtime"].values()}
    files.update({str(path.resolve().relative_to(Path.cwd())):P.sha(path) for path in deps})
    files.update({r["path"]:r["sha256"] for r in rows})
    import torch
    P.save(PLAN, dict(records=rows, runtime=base["runtime"], files=files, threshold_requests=["DEFAULT"],
                      p4=P.read(WORK / "build.json"), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                      attribution="DEMAND, Thiemann, Ito & Vincent. doi:10.5281/zenodo.1227121. CC BY 4.0."))


def compare():
    plan = P.read(PLAN)
    willow = P.read(WORK / "willow.json")
    assert P.sha(PLAN) == willow["plan_sha256"] and P.sha(WORK / "willow.zip") == willow["archive_sha256"]
    official = {r["label"]:r for r in willow["records"]}
    assert plan["p4"] == P.read(WORK / "build.json")
    rows, hashes = [], {}
    trace = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        with ZipFile(tmp / trace.name, "w", ZIP_DEFLATED) as logs, ZipFile(WORK / "willow.zip") as old:
            for row in plan["records"]:
                scene = row["label"]
                record = dict(row)
                for name in ("fixed", "c161"):
                    folder = WORK / name / scene
                    obs, log = p4.replay(name, row, folder)
                    record[name] = obs
                    for file in [log, *folder.glob("*.failed-*.log")]:
                        member = f"{name}/{scene}/{file.name}"
                        logs.write(file, member)
                        hashes[member] = P.sha(file)
                matched = official.pop(scene)
                assert row["pcm_sha256"] == matched["pcm_sha256"] and not matched["reused"]
                record["willow"] = matched["events"]
                for name, checksum in matched["members"].items():
                    member = scene + "/" + name
                    raw = old.read(member)
                    assert p4.fixed.digest(raw) == checksum
                    logs.writestr("willow/" + member, raw)
                    hashes["willow/" + member] = checksum
                rows.append(record)
        assert not official and len(rows) == len(SCENES)
        with ZipFile(tmp / trace.name) as logs:
            assert len(hashes) == len(logs.namelist()) and not logs.testzip()
            for member, sha in hashes.items():
                assert p4.fixed.digest(logs.read(member)) == sha
        P.save(tmp / OUT.name, dict(records=rows, files=hashes, archive_sha256=P.sha(tmp / trace.name),
                                    plan_sha256=P.sha(PLAN), background_hours=sum(r["background_samples"] for r in rows)/57600000))
        (tmp / trace.name).replace(trace)
        (tmp / OUT.name).replace(OUT)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        official = wakes.room.official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif action == "willow":
        wakes.willow_cases(PLAN, WORK, SCRIPT, timeout_seconds=600)
    else:
        {"build":p4.build, "prepare":prepare, "report":compare}[action]()
