"""Compare the same marked Nyx, plain Nyx and level-matched Willow commands dry and facing forward."""
from pathlib import Path
import sys
import eight_voice_willow as paired
import willow_music_cases as runner

PLAN = paired.PLAN.with_name(paired.STEM+"-channels-plan.json")
OUT = PLAN.with_name(paired.STEM+"-channels.json")
WORK = paired.WORK.with_name(paired.WORK.name+"-channels")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    from io import BytesIO
    import json
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile
    import numpy as np
    from scipy.signal import fftconvolve
    import soundfile as sf
    import check_connected_marker_gains as gains

    ch, study, tx = gains.ch, gains.study, gains.tx
    prior = ch.read(paired.PLAN)
    assert ch.sha(paired.PACK)==prior["files"][str(paired.PACK)]
    linked = {(r["batch"],r["track"],r["target_id"]):r for r in prior["records"] if r["mode"]=="rear"}
    trials = {(r["batch"],r["source"]["index"]):r for r in ch.read(gains.GUEST)["records"] if r.get("paired_back_grid")}
    nyx_front = {(r["batch"],r["source"]["index"],r["source"]["external_db"]):r
                 for r in ch.read(gains.OUT)["records"] if r["profile"]=="front"}
    lengths = {(r["source"]["index"],r["source"]["external_db"]):r["samples"]
               for r in ch.read(study.timing.OUT)["records"] if r["timing"]=="tx80" and r["track"]=="marked"}
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    assert rate==16000
    music,_,music_id = tx.study.grid.room.rooms.read_noise("music")
    v5,_ = tx.study.panel_wave()
    taps,_ = gains.direction.filters()
    native = tx.panel.grid.room.rooms.c161_binding()
    WORK.mkdir(exist_ok=True)
    files = {p:h for p,h in prior["files"].items() if not p.startswith(str(paired.WORK)+"/")}
    for p in (RUNNER,paired.PLAN,gains.OUT):
        files[str(p)] = ch.sha(p)
    rows = []
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(paired.PACK) as pack:
        matcher = tx.single.timed.binding(Path(tmp))
        for t in json.loads(pack.read("manifest.json"))["records"]:
            key = t["batch"],t["track"]
            role = t["role"]
            if role not in ("nyx_marked","nyx_plain","willow_matched"):
                continue
            old = linked[*key,role]
            s = trials[key]["source"]
            data = pack.read(t["file"])
            pcm,rate = sf.read(BytesIO(data),dtype="int16")
            assert rate==16000 and gains.direction.sha256(data).hexdigest()==t["wav_sha256"] and ch.digest(pcm)==t["pcm_sha256"]
            if key[0]=="first":
                _,(hx,_,_) = tx.diag.room(s["geometry"])
                n = lengths[key[1],s["external_db"]]-4800-len(hx)+1
            else:
                n = len(pcm)
            mix = study.mixed.room_scene(s,{"length":(s,range(n))},panel,v5,music,music_id)
            def through(angle):
                signal = fftconvolve(pcm.astype(float),taps[angle])
                assert np.max(np.abs(signal[len(pcm):]))<.5
                return mix(signal[:len(pcm)])
            assert ch.digest(through("back")[0])==old["pcm_sha256"]
            front,clips = through("front")
            for mode,mic,clock,clipped in (("front",front,s["transmit_event"],clips),("dry",pcm,0,0)):
                host = study.playback.host_result(matcher,native,mic,clock)
                if mode=="front" and role=="nyx_marked":
                    assert host==nyx_front[*key,s["external_db"]]["result"]
                path = WORK/f"{len(rows):03}.wav"
                sf.write(path,mic,16000,subtype="PCM_16")
                files[str(path)] = ch.sha(path)
                rows.append(dict(old,mode=mode,path=str(path),sha256=files[str(path)],samples=len(mic),nyx=host,
                                 pcm_sha256=host["pcm_sha256"],transmit_event=clock,clipped=clipped,
                                 external_db=s["external_db"] if mode=="front" else None))
    assert len(rows)==144
    plan = dict(records=rows,runtime=prior["runtime"],threshold_requests=["DEFAULT"],files=files,
        policy="Exploratory follow-up after observing rear returns. Use exactly the same 24 dry complete commands. "
               "Dry is nominal file level with no background. Front keeps the previous room, panel, music and "
               "gain and changes only the external speaker response from 180 to 0 degrees. Measure detections only.")
    if PLAN.exists():
        assert plan==ch.read(PLAN)
    else:
        ch.save(PLAN,plan)


def score():
    from collections import Counter, defaultdict
    from tempfile import TemporaryDirectory
    from zipfile import ZIP_DEFLATED, ZipFile
    import check_connected_marker_gains as gains
    from score_eight_voice_willow import timing
    from score_handoff_willow import willow_events

    ch = gains.ch
    plan = ch.read(PLAN)
    digest = ch.sha(PLAN)
    counts,rows = defaultdict(Counter),[]
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        archive = OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for i,r in enumerate(plan["records"]):
                hits = willow_events(i,r,digest,logs,WORK/"runs")
                w = timing(r,(hits[0]+255)//256*256 if hits else None)
                n = timing(r,next(iter(r["nyx"]["opened"]),None))
                for k,res in (("nyx",n),("willow",w)):
                    counts[".".join((r["mode"],r["target_id"],k))].update(
                        trials=1,timely=int(res["timely"]),within_500ms=int(res["within_500ms"]),opened=int(res["opened"] is not None))
                rows.append(dict(index=i,batch=r["batch"],track=r["track"],mode=r["mode"],role=r["target_id"],
                                 raw_willow=hits,nyx=n,willow=w,pcm_sha256=r["pcm_sha256"]))
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==3*len(rows)
        ch.save(work/OUT.name,dict(records=rows,summary=counts,plan_sha256=digest,source_plan_sha256=ch.sha(paired.PLAN),
            archive_sha256=ch.sha(work/archive.name),willow="wn9_heywillow_tts",mode="DEFAULT",nyx_model=ch.read(paired.PLAN)["nyx_model"]))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    runner.PLAN,runner.WORK,runner.RUNNER = PLAN,WORK,RUNNER
    {"prepare":prepare,"run":runner.run,"case":runner.case,"score":score}[sys.argv.pop(1)]()
