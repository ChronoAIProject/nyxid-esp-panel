"""Replay two measured right-ear orientations with the delivered Nyx receiver and official Willow."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path("research/wakeword_p4")
WORK = Path(".local/wakeword/bella-surrey-orientations")
OUT = ROOT / "results/4090-bella-surrey-orientations.json"
PLAN = OUT.with_name(OUT.stem + "-plan.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = ROOT / "experiments/bella_surrey_orientations_protocol.md"
ORIENTATIONS = {30:"54c0ed04944fb3f5f8774f3728a2fbc74b3649370d8dfd28d5dc6ccd88806b08",
                300:"9fb33ba9535ecf5fa602134070ea761f3b2cd2a08d55f0fba99d0c5723d44c45"}


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf
    import surrey_fixed_cues as surrey
    import bella_delayed_command as trial
    import check_address_gain_wakes as comparison
    P = trial.P
    history = {r["index"]:r for r in P.read(trial.daily.OUT)["records"] if r["role"] == "original"}
    rooms, _ = trial.daily.rooms()
    dry_targets = {r[0]["file"]:r[-1]["absent"].astype(np.int32) for r in rooms[-6]}
    release = P.read(trial.pack.OUT)
    assert P.sha(trial.pack.PACK) == release["archive_sha256"] and P.sha(surrey.MAT) == surrey.MAT_SHA
    native = P.read(comparison.room.PLAN)
    files = {p:native["files"][p] for p in native["runtime"].values()}
    sources = (SCRIPT, PROTOCOL, surrey.MAT, trial.pack.OUT, trial.daily.OUT, trial.pack.PACK,
               comparison.room.official.SOURCE, Path(comparison.room.official.__file__))
    files.update({str(p):P.sha(p) for p in sources})
    scale = np.linalg.norm(resample_poly(loadmat(surrey.MAT)["brir"][:,0,2],1,3))
    rng = np.random.Generator(np.random.PCG64(273961))
    rows, paths = [], []
    WORK.mkdir(exist_ok=True, parents=True)
    with ZipFile(trial.pack.PACK) as pack:
        for angle, checksum in ORIENTATIONS.items():
            mat = WORK / f"angle_{angle}.0.mat"
            url = surrey.DATA_URL.replace("angle_0.0.mat",mat.name)
            if not mat.exists():
                with urlopen(url, timeout=60) as response:
                    data = response.read()
                assert sha256(data).hexdigest() == checksum
                mat.write_bytes(data)
            assert P.sha(mat) == checksum
            files[str(mat)] = checksum
            data = loadmat(mat)["brir"]
            assert data.shape == (65536,2,24)
            response = {k:resample_poly(data[:,1,ch],1,3)/scale for k,ch in (("front",2),("back",8))}
            for direction, h in response.items():
                direct = int(np.argmax(np.abs(h[:800])))
                paths.append(dict(angle=angle, direction=direction, source_url=url, sha256=checksum,
                                  rir_sha256=P.digest(h), arrival=direct, common_scale=float(scale)))
                other = response["front" if direction=="back" else "back"]
                for scene in (77,79,90):
                    src = history[scene]
                    target = dry_targets[src["file"]]
                    end = src["command_at"]
                    dry = {}
                    for role in ("early-nyx","nyx","willow","absent"):
                        member = f"{scene:03}/nominal/{role}.wav"
                        raw = pack.read(member)
                        assert sha256(raw).hexdigest() == release["files"][member]
                        pcm, sr = sf.read(BytesIO(raw),dtype="int32")
                        assert sr == 16000 and np.all(pcm % 16384 == 0)
                        dry[role] = pcm // 16384
                    talk = dry["absent"] - target
                    assert not target[:end].any() and not talk[:src["speech_start"]].any() and not talk[src["speech_end"]:].any()
                    competitor = .25*fftconvolve(talk.astype(float),other)
                    noise = rng.normal(0,12,len(competitor))
                    ref_path = WORK / f"ref-{len(noise)}.wav"
                    zero = np.zeros(len(noise), dtype="<i2")
                    if not ref_path.exists():
                        sf.write(ref_path,zero,16000,subtype="PCM_16")
                    check, sr = sf.read(ref_path,dtype="int16")
                    assert sr == 16000 and np.array_equal(check,zero)
                    files[str(ref_path)] = P.sha(ref_path)
                    for role, combined in dry.items():
                        speaker = combined-talk
                        assert np.array_equal(speaker[end:], target[end:])
                        start = None if role=="absent" else int(np.flatnonzero(speaker[:end])[0])+direct
                        wet = .25*fftconvolve(speaker.astype(float),h)
                        for mode, background in (("quiet",noise),("talk",noise+competitor)):
                            total = np.rint(wet+background)
                            mic = total.clip(-32768,32767).astype("<i2")
                            label = f"{angle}-{direction}-{scene:03}-{role}-{mode}"
                            wav = WORK/(label+".wav")
                            sf.write(wav,mic,16000,subtype="PCM_16")
                            files[str(wav)] = P.sha(wav)
                            rows.append(dict(label=label, role=role, angle=angle, direction=direction, mode=mode, scene=scene,
                                             command=src["command"], start=start,end=end+direct,samples=len(mic), path=str(wav),
                                             sha256=files[str(wav)],pcm_sha256=P.digest(mic), ref_sha256=P.digest(zero),
                                             reference_path=str(ref_path),speaker_sha256=P.digest(speaker),talk_sha256=P.digest(talk),
                                             noise_sha256=P.digest(noise),clipped=int(np.count_nonzero(total!=mic))))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 96
    P.save(PLAN,dict(records=rows,paths=paths,files=files,runtime=native["runtime"],threshold_requests=["DEFAULT"]))


def nyx():
    import soundfile as sf
    import bella_reference_barge as receiver
    P, fixed = receiver.P, receiver.test.fixed
    meta = P.read(receiver.WORK/"build.json")
    assert P.sha(receiver.WORK/"guest.flash")==meta["flash_sha256"] and P.sha(fixed.QEMU)==meta["qemu_sha256"]
    assert P.sha(fixed.MODEL)==meta["model_sha256"] and fixed.MODEL.stat().st_size==60952
    flash = bytearray((receiver.WORK/"guest.flash").read_bytes())
    plan = P.read(PLAN)
    cache = WORK / meta["flash_sha256"]
    cache.mkdir(exist_ok=True)
    rows, files = [], {}
    with TemporaryDirectory(dir=WORK) as folder:
        work=Path(folder)
        archive=WORK/"nyx.zip"
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for r in plan["records"]:
                mic,sr=sf.read(r["path"],dtype="int16")
                ref,rr=sf.read(r["reference_path"],dtype="int16")
                assert sr==rr==16000 and P.digest(mic)==r["pcm_sha256"] and P.digest(ref)==r["ref_sha256"]
                block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                for attempt in range(2):
                    try:
                        result,log,faults=fixed.boot(work,flash,cache,block,1)
                        break
                    except AssertionError:
                        if attempt: raise
                for p in (log,*faults):
                    member=r["label"]+("/wake.log" if p==log else "/"+p.name)
                    logs.write(p,member)
                    files[member]=P.sha(p)
                rows.append(dict(label=r["label"],events=result[0]["events"],input_sha256=fixed.digest(block)))
        P.save(work/"nyx.json",dict(records=rows,files=files,build=meta,archive_sha256=P.sha(work/archive.name),plan_sha256=P.sha(PLAN)))
        (work/archive.name).replace(archive)
        (work/"nyx.json").replace(WORK/"nyx.json")


def report():
    import willow_room_reference as room
    plan, left, right = (room.read(p) for p in (PLAN,WORK/"nyx.json",WORK/"willow.json"))
    assert left["plan_sha256"] == right["plan_sha256"] == room.digest(PLAN.read_bytes())
    rows, counts, files = [], defaultdict(Counter), {}
    trace = WORK / "4090-bella-surrey-orientations-wake-uart.zip"
    with ZipFile(trace,"w",ZIP_DEFLATED) as out:
        for name, doc in (("nyx",left),("willow",right)):
            path=WORK/(name+".zip")
            assert room.digest(path.read_bytes())==doc["archive_sha256"]
            expected=doc["files"] if name=="nyx" else {r["label"]+"/"+k:v for r in doc["records"] for k,v in r["members"].items()}
            with ZipFile(path) as z:
                assert set(expected)==set(z.namelist()) and len(expected)==len(z.namelist())
                for k,v in expected.items():
                    raw=z.read(k)
                    assert room.digest(raw)==v
                    out.writestr(name+"/"+k,raw)
                    files[name+"/"+k]=v
    for r,n,w in zip(plan["records"],left["records"],right["records"],strict=True):
        assert r["label"]==n["label"]==w["label"] and r["pcm_sha256"]==w["pcm_sha256"]
        row=dict(r,nyx=n["events"],willow=w["events"])
        for engine, clocks in (("nyx",n["events"]["O"]),("willow",[e[1] for e in w["events"]])):
            opened=next(iter(clocks),None)
            timely=opened is not None and r["start"] is not None and r["start"] <= opened < r["end"]
            counts["/".join((r["mode"],r["role"],engine))].update(cases=1,opened=int(opened is not None),timely=int(timely))
            row[engine+"_open"]=opened
            row[engine+"_timely"]=timely
        rows.append(row)
    room.save(OUT,dict(records=rows,counts=counts,files=files,build=left["build"],plan_sha256=left["plan_sha256"],
                       archive_sha256=room.digest(trace.read_bytes())))


if __name__ == "__main__":
    stage=sys.argv.pop(1)
    if stage=="case":
        sys.path.insert(0,str(ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN,official.__file__=PLAN,str(SCRIPT)
        official.case()
    elif stage=="willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    else:
        {"prepare":prepare,"nyx":nyx,"report":report}[stage]()
