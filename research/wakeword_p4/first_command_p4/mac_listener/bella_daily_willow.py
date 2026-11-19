"""Run official DEFAULT Willow on the same interfering speech clock as each physical-R Nyx pair."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import willow_music_cases as runner

WORK = Path(".local/wakeword/c161-bella-daily-willow")
PLAN = runner.ROOT / "results/4090-c161-bella-daily-willow-plan.json"
OUT = PLAN.with_name("4090-c161-bella-daily-willow.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import check_bella_daily_overlap as daily

    p, matched = daily.P, daily.matched
    parent = p.read(matched.PLAN)
    files = {v:parent["files"][v] for v in parent["runtime"].values()}
    for path in (SCRIPT,matched.PLAN,matched.PACK,daily.OUT,Path(daily.__file__).resolve().relative_to(Path.cwd())):
        files[str(path)] = p.sha(path)
    assert files[str(matched.PACK)] == parent["files"][str(matched.PACK)]
    rooms, taps = daily.rooms()
    by_file = {item[0]["file"]:item for values in rooms.values() for item in values}
    nyx = {r["index"]:r for r in p.read(daily.OUT)["records"] if r["role"] == "absent"}
    WORK.mkdir(exist_ok=True,parents=True)
    rows = []
    with ZipFile(matched.PACK) as pack:
        dry = {r["file"]:r for r in json.loads(pack.read("manifest.json"))["records"]}
        for source, key, _, pcm in daily.event_inputs("daily"):
            i = int(key)
            old = nyx.pop(i)
            name = old["file"]
            case, s, scene, ref, active, tracks = by_file[name]
            assert source["text"]==old["text"] and p.digest(pcm)==old["dry_speech_sha256"]
            talk, level = daily.tx.balanced.match_level(active,pcm)
            assert level["gain_db"]==old["speech_gain_db"] and p.digest(ref)==old["reference_sha256"]
            at, stop = old["speech_start"],old["speech_end"]
            willow = daily.study.playback.read_wav(pack,dry[name])
            end = old["command_at"]
            assert np.array_equal(willow[end:],tracks["original"][end:])
            for role, wave in (("willow",willow),("absent",tracks["absent"])):
                external = wave.astype(float)
                external[at:stop] += talk
                mic, clips = scene(daily.board.bella.rear_filter(external,taps))
                digest = p.digest(mic)
                if role == "absent":
                    assert digest==old["mic_sha256"]
                wav = WORK/f"{len(rows):03}.wav"
                sf.write(wav,mic,16000,subtype="PCM_16")
                files[str(wav)] = p.sha(wav)
                rows.append(dict(index=i,target_id=role,group=old["group"],text=old["text"],file=name,db=s["external_db"],
                                 start=dry[name]["start"],end=end,samples=len(mic),clipped=clips,pcm_sha256=digest,
                                 reference_sha256=case["reference_sha256"],path=str(wav),sha256=files[str(wav)]))
    assert not nyx and len(rows)==len({r["pcm_sha256"] for r in rows})==192
    plan = dict(records=rows,files=files,runtime=parent["runtime"],threshold_requests=["DEFAULT"],
                policy="One official original-level Bella Willow and its paired erased control per consumed daily utterance. "
                       "Identical competing-voice samples and placement to the Nyx pair. Erased M hash matches the P4 "
                       "trial. Keep the command tail and room fixed. The DEFAULT detector receives raw microphone M.")
    if PLAN.exists():
        assert p.read(PLAN)==plan
    else:
        p.save(PLAN,plan)


def score():
    import check_bella_daily_overlap as daily
    from score_eight_voice_willow import timing
    from score_handoff_willow import willow_events

    p = daily.P
    rows, counts = [],defaultdict(Counter)
    sha = p.sha(PLAN)
    plan = p.read(PLAN)
    with TemporaryDirectory(dir=WORK) as folder:
        work = Path(folder)
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for i,s in enumerate(plan["records"]):
                hits = willow_events(i,s,sha,z,WORK/"runs")
                result = timing(s,(hits[0]+255)//256*256 if hits else None)
                rows.append(dict(index=s["index"],role=s["target_id"],file=s["file"],group=s["group"],text=s["text"],
                                 db=s["db"],command_at=s["end"],pcm_sha256=s["pcm_sha256"],events=hits,**result))
                t = result["opened"]
                for subset in ("all",s["group"],str(s["db"])):
                    counts[s["target_id"]+"."+subset].update(cases=1,opened=t is not None,timely=result["timely"],
                                              early=t is not None and t<s["start"],late=t is not None and t>=s["end"])
        with ZipFile(work/archive.name) as z:
            assert len(z.namelist())==192*3 and z.testzip() is None
        p.save(work/OUT.name,dict(records=rows,summary=counts,model="wn9_heywillow_tts",actual_threshold_bits="3f24dd2f",
                                  plan_sha256=sha,archive_sha256=p.sha(work/archive.name),nyx_sha256=p.sha(daily.OUT)))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print({k:dict(v) for k,v in counts.items() if k.endswith(".all")})


if __name__ == "__main__":
    runner.PLAN,runner.WORK,runner.RUNNER = PLAN,WORK,SCRIPT
    {"prepare":prepare,"run":runner.run,"case":runner.case,"score":score}[sys.argv.pop(1)]()
