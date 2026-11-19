"""Replay both fixed af_bella calls at matched levels on all 24 weak rear conditions."""
from pathlib import Path
import sys

import willow_music_cases as runner

WORK = Path(".local/wakeword/c161-bella-peak-willow")
PLAN = runner.ROOT / "results/4090-c161-bella-peak-willow-plan.json"
OUT = PLAN.with_name("4090-c161-bella-peak-willow.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    from io import BytesIO
    import json
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile
    import numpy as np
    import soundfile as sf
    import check_bella_peak_playback as peak
    import eight_voice_willow as willow
    from surrey_fixed_cues import PLAN as surrey

    ch, tx, study = peak.ch, peak.tx, peak.study
    known = ch.read(peak.OUT)
    previous = ch.read(surrey)
    files = {p:previous["files"][p] for p in previous["runtime"].values()}
    for p in (SCRIPT,peak.OUT,peak.PACK,peak.weak.PACK,willow.PACK,willow.PLAN,surrey):
        files[str(p)]=ch.sha(p)
    assert ch.sha(willow.PACK)==ch.read(willow.PLAN)["files"][str(willow.PACK)]
    assert ch.sha(peak.PACK)==known["pack_sha256"]
    with ZipFile(willow.PACK) as pack:
        entries = json.loads(pack.read("manifest.json"))["records"]
        anchors = {r["role"]:r for r in entries if r["batch"]=="second" and r["track"]==0}
        cues = {}
        for role in ("willow_matched","nyx_marked"):
            entry = anchors[role]
            data = pack.read(entry["file"])
            assert peak.package.digest(data)==entry["wav_sha256"]
            track, rate = sf.read(BytesIO(data),dtype="int16")
            assert rate == 16000 and ch.digest(track)==entry["pcm_sha256"]
            cues[role] = track[32000:entry["command_at"]]
    assert np.array_equal(cues["nyx_marked"],sf.read(peak.bella.CUE,dtype="int16")[0])
    gain = 10**(known["gain_db"]/20)
    high = np.rint(cues["willow_matched"].astype(float)*gain)
    assert np.max(np.abs(high)) < 32767
    fixed = {"willow_original":cues["willow_matched"],"willow_louder":high.astype("<i2"),
             "nyx_original":cues["nyx_marked"],"nyx_louder":sf.read(peak.CUE,dtype="int16")[0],
             "no_address":np.zeros_like(cues["nyx_marked"])}
    levels = {}
    for role, signal in fixed.items():
        anchor = anchors["willow_matched" if role.startswith("willow") else "nyx_marked"]
        duration = anchor["command_at"]-anchor["start"]
        if role!="no_address":
            levels[role]=tx.balanced.dry_levels(signal[-duration:])
    for tag in ("original","louder"):
        assert abs(levels["willow_"+tag]["active_rms_dbfs"]-levels["nyx_"+tag]["active_rms_dbfs"]) < .01
    original = {r["file"]:r for r in known["records"] if r["source"]["external_db"]==-6}
    low = {r["file"]:r for r in ch.read(peak.weak.OUT)["records"] if r["condition"]=="back_panel1_music1" and r["role"]=="fixed"}
    lengths = {r["source"]["index"]:r["samples"] for r in ch.read(study.timing.OUT)["records"]
               if r["track"]=="marked" and r["timing"]=="tx80" and r["source"]["external_db"]==-6}
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    v5,_ = tx.study.panel_wave()
    bed,_,music_id = tx.study.grid.room.rooms.read_noise("music")
    filters,_ = peak.bella.gains.direction.filters()
    assert rate==16000
    WORK.mkdir(exist_ok=True)
    native = tx.panel.grid.room.rooms.c161_binding()
    rows = []
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(peak.PACK) as pack:
        matcher = tx.single.timed.binding(Path(tmp))
        for entry in json.loads(pack.read("manifest.json"))["records"]:
            r = original.pop(entry["file"])
            s = r["source"]
            end = s["command_at"]
            louder = study.playback.read_wav(pack,entry)
            assert ch.digest(louder[end:]) == entry["command_pcm_sha256"]
            n = len(louder)
            if Path(entry["file"]).parent.name=="first":
                _, (hx, _, _) = tx.diag.room(s["geometry"])
                n = lengths.pop(s["index"]) - 4800 - len(hx) + 1
            mix = study.mixed.room_scene(s, {"length":(entry,range(n))}, panel,v5,bed,music_id)
            for role, cue in fixed.items():
                source = louder.copy()
                source[:end-len(cue)] = 0
                source[end-len(cue):end] = cue
                anchor = anchors["willow_matched" if role.startswith("willow") else "nyx_marked"]
                start = end-(anchor["command_at"]-anchor["start"])
                mic, clips = mix(peak.bella.rear_filter(source,filters["back"]))
                route = study.playback.host_result(matcher,native,mic,s["transmit_event"])
                if role=="nyx_louder":
                    assert route==r["result"] and np.array_equal(source,louder)
                if role=="nyx_original":
                    assert route==low.pop(entry["file"])["result"]
                if role=="no_address":
                    assert route==r["no_address"]
                p = WORK/f"{len(rows):03}.wav"
                sf.write(p,mic,16000,subtype="PCM_16")
                files[str(p)] = ch.sha(p)
                rows.append(dict(file=entry["file"],target_id=role,start=start,end=end,clips=clips,
                    samples=len(mic),path=str(p),sha256=files[str(p)],pcm_sha256=ch.digest(mic),
                    command_pcm_sha256=entry["command_pcm_sha256"],nyx=route))
    assert not low and not original and not lengths and len(rows)==120
    plan=dict(records=rows, files=files, runtime=previous["runtime"],threshold_requests=["DEFAULT"],
              levels=levels, gain_db=known["gain_db"], policy="Same af_bella voice, source gain and command PCM. "
              "Compare the archived active-RMS-matched calls and amplify each complete call by the same factor. "
              "Reuse all weak rear cases, unchanged panel timeline, reference and microphone gain.")
    if PLAN.exists():
        assert ch.read(PLAN)==plan
    else:
        ch.save(PLAN,plan)


def score():
    from collections import Counter, defaultdict
    from tempfile import TemporaryDirectory
    from zipfile import ZIP_DEFLATED, ZipFile
    from score_eight_voice_willow import timing, ch
    from score_handoff_willow import willow_events

    plan = ch.read(PLAN)
    digest = ch.sha(PLAN)
    out_zip = OUT.with_suffix(".zip")
    rows,summary = [],defaultdict(Counter)
    with TemporaryDirectory(dir=WORK) as folder:
        temp=Path(folder)
        with ZipFile(temp/out_zip.name,"w",ZIP_DEFLATED) as archive:
            for i,s in enumerate(plan["records"]):
                hits=willow_events(i,s,digest,archive,WORK/"runs")
                willow=timing(s,(hits[0]+255)//256*256 if hits else None)
                nyx=timing(s,next(iter(s["nyx"]["opened"]),None))
                rows.append(dict(file=s["file"],role=s["target_id"],willow_events=hits,willow=willow,nyx=nyx))
                for name, result in (("willow",willow),("nyx",nyx)):
                    summary[name+"."+s["target_id"]].update(cases=1,timely=result["timely"],
                        within_500ms=result["within_500ms"],opened=result["opened"] is not None)
        with ZipFile(temp/out_zip.name) as archive:
            assert len(archive.namelist())==360 and archive.testzip() is None
        ch.save(temp/OUT.name,dict(records=rows,summary=summary,plan_sha256=digest,archive_sha256=ch.sha(temp/out_zip.name)))
        (temp/out_zip.name).replace(out_zip)
        (temp/OUT.name).replace(OUT)


if __name__=="__main__":
    runner.PLAN, runner.WORK, runner.RUNNER=PLAN,WORK,SCRIPT
    {"prepare":prepare,"run":runner.run,"case":runner.case,"score":score}[sys.argv.pop(1)]()
