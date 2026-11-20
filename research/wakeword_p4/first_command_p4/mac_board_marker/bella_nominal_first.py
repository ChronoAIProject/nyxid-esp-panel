"""Put nominal-rate fixed patterns in the early triplet and compare two late triplets."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf
import torch

import bella_240_masking as source

fit, P = source.fit, source.P
WORK = Path(".local/wakeword/bella-nominal-first")
ROOT = fit.study.ROOT
OUT = ROOT / "results/4090-bella-nominal-first-fit.json"
PROTO = ROOT / "experiments/bella_nominal_first_protocol.md"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
NAMES = ("nominal", "multirate")


def wav_path(name):
    return OUT.with_name(f"hye-nyx-bella-nominal-first-{name}.wav")


def build():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    early = fit.early
    seed = P.read(early.playback.OUT)
    base, sr = sf.read(early.WAV.with_name(seed["cue"]), dtype="int16")
    original, rr = sf.read(early.WAV, dtype="int16")
    late, lr = sf.read(early.fitted.WAV, dtype="int16")
    assert sr==rr==lr==16000 and P.digest(base)==seed["pcm_sha256"]
    assert P.sha(early.WAV)==P.read(early.FIT)["cue_sha256"] and P.sha(early.fitted.WAV)==P.read(early.fitted.FIT)["cue_sha256"]
    _, _, _, codes = early.fitted.stems.gains.fit.grid.fit.original()
    delta = original.astype(float)-base
    advance = P.read(early.FIT)["advanced_samples"]
    added = np.zeros_like(delta)
    for stem, code in zip(P.read(early.fitted.FIT)["stems"], codes, strict=True):
        a, b = stem["window"]
        at = stem["start"]-advance
        added[at:at+len(code)] = code * (np.linalg.norm(delta[a-advance:b-advance])/np.linalg.norm(code))
    carrier = torch.tensor(base, device="cuda", dtype=torch.float64)
    extra = torch.tensor(added, device="cuda", dtype=torch.float64)
    trials = []
    for shift in range(-64, 65, 4):
        for sign in (1,-1):
            stem = sign*torch.roll(extra,shift)
            pos, neg = stem>0, stem<0
            gain = min(1., float(((32767-carrier[pos])/stem[pos]).min()), float(((-32768-carrier[neg])/stem[neg]).min()))
            trials.append(dict(shift=shift, sign=sign, gain=gain))
    best = min(trials, key=lambda r:(-r["gain"], abs(r["shift"]), -r["sign"], r["shift"]))
    addition = best["gain"]*best["sign"]*np.roll(added, best["shift"])
    assert not np.any((addition != 0) & (late != base))
    hashes = {}
    for name, background in zip(NAMES, (base, late), strict=True):
        total = np.rint(background.astype(float)+addition)
        assert total.max()<=32767 and total.min()>=-32768
        pcm = total.astype("<i2")
        assert np.array_equal(pcm[addition==0], background[addition==0])
        path = wav_path(name)
        path.write_bytes(early.playback.playback.sources.wav(pcm,"PCM_16"))
        hashes[name] = P.sha(path)
    inputs = (PROTO,SCRIPT,early.WAV,early.fitted.WAV,early.FIT,early.fitted.FIT,early.WAV.with_name(seed["cue"]))
    P.save(OUT,dict(voices=hashes,fit=best,trials=trials,gpu=torch.cuda.get_device_name(),sources={str(p):P.sha(p) for p in inputs}))


def samples(before, after):
    for row in fit.scenes(before, after):
        yield (*row, None)
    plan = P.read(source.angles.PLAN)
    scale = P.read(fit.study.PLAN)["paths"][0]["common_scale"]
    paths = {}
    for angle, checksum in source.angles.ANGLES.items():
        mat = source.angles.WORK/f"angle_{angle}.0.mat"
        assert P.sha(mat)==checksum
        brir = loadmat(mat)["brir"]
        for side, channel in (("front",2),("back",8)):
            paths[angle,side] = resample_poly(brir[:,1,channel],1,3)/scale
    arrivals = {(p["angle"],p["direction"]):p["arrival"] for p in plan["paths"]}
    for r, dry, old, talk, noise in source.stems_for({120,240}, ("early-nyx",)):
        coord = r["angle"],r["direction"]
        end = r["end"]-arrivals[coord]
        assert np.array_equal(dry[end-len(before):end],before)
        dry = dry.copy()
        dry[end-len(before):end] = after
        bg = talk+noise
        assert P.digest(np.rint(old+bg).clip(-32768,32767).astype("<i2"))==r["pcm_sha256"]
        mic = np.rint(.25*fftconvolve(dry.astype(float), paths[coord])+bg).clip(-32768,32767).astype("<i2")
        yield r["label"],"new_angles",mic,np.zeros_like(mic),r["nyx_open"],r["start"],r["end"],r


def replay(name):
    receiver, fixed = source.receiver, source.fixed
    doc = P.read(OUT)
    assert P.sha(wav_path(name))==doc["voices"][name]
    before, sr = sf.read(fit.early.WAV,dtype="int16")
    after, rr = sf.read(wav_path(name),dtype="int16")
    assert sr==rr==16000
    build = P.read(receiver.WORK/"build.json")
    flash = bytearray((receiver.WORK/"guest.flash").read_bytes())
    assert fixed.digest(flash)==build["flash_sha256"] and P.sha(fixed.QEMU)==build["qemu_sha256"]
    folder = WORK/name
    cache = folder/build["flash_sha256"]
    cache.mkdir(exist_ok=True,parents=True)
    report = OUT.with_name(f"4090-bella-nominal-first-{name}.json")
    archive = report.with_suffix(".zip")
    counts, rows, files = defaultdict(Counter), [], {}
    with TemporaryDirectory(dir=folder) as tmp:
        work=Path(tmp)
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for label,group,mic,ref,old,start,end,source_row in samples(before,after):
                block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                native,log,faults=fixed.boot(work,flash,cache,block,1)
                for p in (log,*faults):
                    member=label+("/wake.log" if p==log else "/"+p.name)
                    logs.write(p,member)
                    files[member]=P.sha(p)
                events=native[0]["events"]
                opened=next(iter(events["O"]),None)
                a=old is not None and (start is None or start<=old<end)
                b=opened is not None and (start is None or start<=opened<end)
                counts[group].update(cases=1,before=int(a),after=int(b),gained=int(b and not a),lost=int(a and not b))
                item=dict(label=label,group=group,before=old,events=events,start=start,end=end,
                          mic_sha256=P.digest(mic),ref_sha256=P.digest(ref),input_sha256=fixed.digest(block))
                if source_row:
                    path = folder/(label+".wav")
                    sf.write(path,mic,16000,subtype="PCM_16")
                    item.update(path=str(path),reference_path=source_row["reference_path"],command=source_row["command"])
                rows.append(item)
        assert len(rows)==len({r["label"] for r in rows})==59
        inputs=(SCRIPT,PROTO,OUT,fit.OUT,source.angles.OUT,fit.study.OUT,fit.early.PLAN)
        P.save(work/report.name,dict(records=rows,counts=counts,files=files,build=build,archive_sha256=P.sha(work/archive.name),
                                     sources={str(p):P.sha(p) for p in inputs}))
        (work/archive.name).replace(archive)
        (work/report.name).replace(report)


if __name__ == "__main__":
    if sys.argv[1]=="build":
        build()
    elif sys.argv[1] in NAMES:
        replay(sys.argv[1])
