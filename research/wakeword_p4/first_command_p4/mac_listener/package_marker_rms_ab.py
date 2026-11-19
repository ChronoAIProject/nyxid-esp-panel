"""Make physically playable A/B pairs and digital microphone/reference inputs from verified P4 cases."""
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_normalized_marker_gains as test
from package_context_playback import add, formatted

study, grid, ch, prior = test.study, test.grid, test.ch, test.prior
STEM = "c161-marker-rms-mac-ab"
PACK = ch.RESULTS/(STEM+".zip")
REPORT = ch.RESULTS/(STEM+".json")


def main():
    base = [r for r in ch.read(study.OUT)["records"] if r["group"]=="other_music"]
    scored = [r for r in ch.read(test.OUT)["records"] if r["group"]=="other_music"]
    selected = {}
    songs = set()
    for i,(a,b) in enumerate(zip(base,scored,strict=True)):
        if a["routes"]["markers"]["self_opened"]:
            selected[i]="previous_self_open"
        elif not a["routes"]["v5"]["timely"] and b["routes"]["rms"]["timely"] and a["source"]["track"] not in songs and len(songs)<4:
            songs.add(a["source"]["track"])
            selected[i]="gained"
    assert len(selected)==5 and len(songs)==4
    dry = test.tracks()
    commands = grid.room.rooms.source_bank()
    cases = []
    tmp = PACK.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as pack:
        work=Path(folder)
        full=ch.binding(work/"full.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        for i,(s,signals,_) in enumerate(study.more_music(dry)):
            if i not in selected: continue
            assert s==base[i]["source"]
            case=dict(index=i,selection=selected[i],source=s,arms={})
            for name in ("v5","rms"):
                route=(base if name=="v5" else scored)[i]["routes"][name]
                x=signals[name]
                ref=prior.level.guard.refs(s,name,dry)
                assert ch.digest(x)==route["pcm_sha256"] and ch.digest(ref)==route["reference_sha256"]
                command=commands[s["voice"],s["command"]]
                assert ch.digest(command)==s["command_pcm_sha256"]
                physical=np.r_[dry[name],command]
                prefix=f"{i:03}/{name}-"
                assets={key:prior.audio.prior.put_audio(pack,prefix+key+".wav",wave) for key,wave in (
                        ("speaker",physical),("mic",x),("reference",ref))}
                results={}
                for local,rx in ((False,np.zeros_like(x)),(True,ref)):
                    n=4*len(cases)+2*(name=="rms")+int(local)
                    for attempt in range(2):
                        try:
                            obs=prior.p4.phase.probe(work,s,x,rx,full,pair,n)
                            break
                        except (AssertionError,TimeoutError):
                            pack.write(work/"serial.log",f"retry/{n}-{attempt}.txt")
                            if attempt: raise
                    want=route["self_opened"] if local else ([] if route["opened"] is None else [route["opened"]])
                    assert obs["opened"]==want and obs["c161"]==route["c161"] and obs["fixed"]==[e["t"] for e in route["fixed"]]
                    role="self" if local else "external"
                    results[role]={k:obs[k] for k in ("allowed","opened","c161","fixed","reference_events")}
                    add(pack,prefix+role+".txt",("\n".join(obs["uart_events"])+"\n").encode())
                case["arms"][name]=dict(assets=assets,board=results)
            cases.append(case)
            print(len(cases),i,flush=True)
        manifest=dict(rate=16000,format="mono PCM16",physical_command_sample=len(dry["v5"]),cases=cases,
            firmware_sha256=ch.sha(prior.p4.phase.handoff.test.BUILD/"nyx_mac_listener.bin"),
            instructions="Speaker files are dry: play at fixed system volume and position. Each pair has the exact "
            "same command samples after the wake. Digital mic files already include room and music and use the "
            "command_at in each source. Supply zero R for external playback and its paired reference on R for "
            "self playback. The room clock error applies to the digital wake only. Results are from simulated P4.")
        add(pack,"manifest.json",formatted(manifest))
    with ZipFile(tmp) as z:
        assert len(cases)==5 and z.testzip() is None
        for case in json.loads(z.read("manifest.json"))["cases"]:
            suffixes=[]
            for arm in case["arms"].values():
                for role,asset in arm["assets"].items():
                    data=z.read(asset["file"])
                    assert hashlib.sha256(data).hexdigest()==asset["sha256"]
                    wave,rate=sf.read(io.BytesIO(data),dtype="int16")
                    assert rate==16000 and ch.digest(wave)==asset["pcm_sha256"]
                    if role=="speaker": suffixes.append(ch.digest(wave[manifest["physical_command_sample"]:]))
            assert suffixes==[case["source"]["command_pcm_sha256"]]*2
    tmp.replace(PACK)
    ch.save(REPORT,dict(**manifest,archive_sha256=ch.sha(PACK),sources={p.name:ch.sha(p) for p in
            (test.OUT,study.OUT,test.levels.FIT,Path(__file__))}))


if __name__=="__main__":
    main()
