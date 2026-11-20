"""Run unchanged Nyx and official Willow receivers on registered real background inputs."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf

import fixed_rearm_demand as indoor

P, p4, wakes = indoor.P, indoor.p4, indoor.wakes
RESULTS = indoor.OUT.parent
WORK = Path(".local/wakeword/fixed-rearm-speech-music")
PLAN = RESULTS / "4090-fixed-rearm-speech-music-plan.json"
OUT = RESULTS / "4090-fixed-rearm-speech-music.json"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = indoor.PROTOCOL.with_name("fixed_rearm_speech_music.md")
VOX = RESULTS / "4090-c161-history-gain-melo-v1.json"
MUSIC = RESULTS / "4090-c161-music-development-sources-v1.json"
EARNINGS = RESULTS / "4090-c161-history-gain-jenny-v1.json"
EVENT = RESULTS / "4090-c161-final-marker-background-event.json"
RATE = 16000


def complete(group, item):
    path = Path(item["path"])
    pcm, rate = sf.read(path, dtype="int16")
    assert rate == RATE and pcm.ndim == 1 and sf.info(path).subtype == "PCM_16"
    assert P.sha(path) == item["sha256"] and P.digest(pcm) == item["pcm_sha256"] and len(pcm) == item["samples"]
    source = item.get("source_path", path.name)
    return dict(label=group+"-"+Path(source).stem, group=group, source=source, source_start=0,
                path=str(path), samples=len(pcm), pcm_sha256=item["pcm_sha256"], sha256=item["sha256"])


def prepare():
    WORK.mkdir(exist_ok=True)
    rows = []
    vox = [r["source"] for r in P.read(VOX)["records"] if r["phase"] == "background"]
    for group, sources in (("vox",vox), ("music",P.read(MUSIC)["records"])):
        rows.extend(complete(group, r) for r in sources if 30*RATE <= r["samples"] <= 300*RATE)
    historical, = P.read(EVENT)["records"]
    earnings, = [r["source"] for r in P.read(EARNINGS)["records"]
                 if r["phase"] == "background" and r["source"]["pcm_sha256"] == historical["source_pcm_sha256"]]
    path = Path(earnings["wav_path"])
    assert P.sha(path) == earnings["wav_sha256"]
    with sf.SoundFile(path) as stream:
        assert (stream.channels, stream.samplerate, stream.subtype) == (1, RATE, "PCM_16")
        stream.seek(390*RATE)
        clip = stream.read(300*RATE, dtype="int16")
    offset = historical["offset"] - 390*RATE
    assert len(clip) == 300*RATE and P.digest(clip[offset:offset+historical["samples"]]) == historical["clip_pcm_sha256"]
    excerpt = WORK / "earnings-4483680.wav"
    sf.write(excerpt, clip, RATE, subtype="PCM_16")
    rows.append(dict(label=excerpt.stem, group="earnings", source=historical["source"], source_start=390*RATE,
                     source_sha256=earnings["wav_sha256"], path=str(excerpt), samples=len(clip),
                     sha256=P.sha(excerpt), pcm_sha256=P.digest(clip),
                     old_marker_in_excerpt=historical["event_in_file"]-390*RATE))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == len({r["label"] for r in rows})
    base = P.read(indoor.PLAN)
    assert base["p4"] == P.read(p4.WORK/"build.json")
    deps = [VOX, MUSIC, EARNINGS, EVENT, PROTOCOL, indoor.PLAN, SCRIPT, Path(p4.__file__), Path(wakes.__file__)]
    files = {str(p.resolve().relative_to(Path.cwd())):P.sha(p) for p in deps}
    files.update({path:base["files"][path] for path in base["runtime"].values()})
    files.update({r["path"]:r["sha256"] for r in rows})
    P.save(PLAN, dict(records=rows, runtime=base["runtime"], files=files, threshold_requests=["DEFAULT"], p4=base["p4"]))
    print(len(rows), sum(r["samples"] for r in rows)/RATE, flush=True)


def run_p4():
    plan = P.read(PLAN)
    assert plan["p4"] == P.read(p4.WORK/"build.json")
    for r in plan["records"]:
        for name in ("fixed", "c161"):
            obs, _ = p4.replay(name,r,WORK/name/r["label"])
            print(name,r["label"],obs["events"],flush=True)


def report():
    plan = P.read(PLAN)
    willow = P.read(WORK/"willow.json")
    assert willow["plan_sha256"] == P.sha(PLAN) and willow["archive_sha256"] == P.sha(WORK/"willow.zip")
    official = {r["label"]:r for r in willow["records"]}
    assert len(official) == len(plan["records"])
    rows, members, counts = [], {}, {}
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(WORK/"willow.zip") as old:
        tmp = Path(folder)
        archive = tmp / OUT.with_suffix(".zip").name
        with ZipFile(archive,"w",ZIP_DEFLATED) as logs:
            for r in plan["records"]:
                label, samples = r["label"], r["samples"]
                item = dict(r)
                decisions = {}
                for name in ("fixed","c161"):
                    folder = WORK/name/label
                    item[name], log = p4.replay(name, r, folder)
                    decisions[name] = [e[2] for e in item[name]["events"]]
                    for path in [log, *sorted(folder.glob("*.failed-*.log"))]:
                        member = f"{name}/{label}/{path.name}"
                        logs.write(path,member)
                        members[member] = P.sha(path)
                w = official.pop(label)
                assert not w["reused"] and w["pcm_sha256"] == r["pcm_sha256"]
                item["willow"] = w["events"]
                decisions["willow"] = [e[1] for e in w["events"]]
                for name, digest in w["members"].items():
                    data = old.read(label+"/"+name)
                    assert sha256(data).hexdigest() == digest
                    member = "willow/"+label+"/"+name
                    logs.writestr(member,data)
                    members[member] = digest
                group = counts.setdefault(r["group"], Counter())
                group.update(recordings=1,samples=samples)
                for name, ends in decisions.items():
                    group[name] += sum(end <= samples for end in ends)
                    group[name+"_padding"] += sum(end > samples for end in ends)
                rows.append(item)
        with ZipFile(archive) as logs:
            assert len(logs.namelist()) == len(members) and not official
            assert all(sha256(logs.read(k)).hexdigest() == v for k,v in members.items())
        P.save(tmp/OUT.name, dict(records=rows, counts=counts, files=members, plan_sha256=P.sha(PLAN),
                                  archive_sha256=P.sha(archive), before_prime_samples=81600,
                                  policy="Zero electrical reference. Only original samples contribute to each group's total."))
        archive.replace(OUT.with_suffix(".zip"))
        (tmp/OUT.name).replace(OUT)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        official = wakes.room.official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif action == "willow":
        wakes.willow_cases(PLAN, WORK, SCRIPT, timeout_seconds=600)
    else:
        {"prepare":prepare,"p4":run_p4,"report":report}[action]()
