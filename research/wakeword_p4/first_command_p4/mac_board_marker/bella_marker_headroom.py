"""Fit peak headroom on the isolated early marker, then replay the frozen sound in P4."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve
import soundfile as sf
import torch

import package_bella_surrey_speakers as speakers
import bella_reference_barge as receiver

trial, study, P = speakers.trial, speakers.study, speakers.P
early = trial.weak.early
WORK = Path(".local/wakeword/bella-marker-headroom")
OUT = study.OUT.with_name("4090-bella-marker-headroom.json")
WAV = OUT.with_name("hye-nyx-af-bella-headroom-experiment.wav")
PROTO = study.ROOT/"experiments/bella_marker_headroom_protocol.md"


def fit():
    import scipy
    base_meta = P.read(early.playback.OUT)
    base_path = early.WAV.with_name(base_meta["cue"])
    base, sr = sf.read(base_path,dtype="int16")
    previous, rr = sf.read(early.WAV,dtype="int16")
    fit_meta = P.read(early.FIT)
    assert sr == rr == 16000 and P.sha(early.WAV)==fit_meta["cue_sha256"] and P.sha(base_path)==base_meta["cue_sha256"]
    scale = fit_meta["added_stem_scale"]
    device = torch.cuda.get_device_name()
    assert "4090" in device
    carrier = torch.tensor(base,dtype=torch.float64,device="cuda")
    extra = (torch.tensor(previous,dtype=torch.float64,device="cuda")-carrier)/scale
    assert not torch.any(extra[:64]) and not torch.any(extra[-64:])
    fits = []
    for shift in range(-64,65,4):
        for sign in (1,-1):
            v=sign*torch.roll(extra,shift)
            positive,negative = v>0,v<0
            gain=min(1., float(((32767-carrier[positive])/v[positive]).min()), float(((-32768-carrier[negative])/v[negative]).min()))
            fits.append(dict(shift=shift, sign=sign, gain=gain))
    winner=min(fits,key=lambda r:(-r["gain"],abs(r["shift"]),-r["sign"],r["shift"]))
    audio=torch.round(carrier+winner["gain"]*winner["sign"]*torch.roll(extra,winner["shift"])).cpu().numpy()
    assert audio.max()<=32767 and audio.min()>=-32768
    audio=audio.astype("<i2")
    start=P.read(early.fitted.FIT)["stems"][0]["window"][0]
    assert np.array_equal(audio[start:],previous[start:]) and np.any(audio!=previous)
    blob=early.playback.playback.sources.wav(audio,"PCM_16")
    if WAV.exists():
        assert WAV.read_bytes()==blob
    else:
        WAV.write_bytes(blob)
    return previous, audio, dict(previous_scale=scale, device=device, torch=torch.__version__, scipy=scipy.__version__,
                                  winner=winner, fits=fits, cue_sha256=P.sha(WAV),original_cue_sha256=P.sha(early.WAV))


def scenes(before, after):
    from scipy.signal import resample_poly
    plan=P.read(study.PLAN)
    registered={r["label"]:r for r in P.read(study.OUT)["records"]}
    historic={r["label"]:r for r in P.read(receiver.STEM.with_name(receiver.STEM.name+"-historical.json"))["records"]}
    angles={(r["angle"],r["direction"]):r for r in plan["paths"]}
    src={r["index"]:r for r in P.read(trial.daily.OUT)["records"] if r["role"]=="original"}
    rng=np.random.Generator(np.random.PCG64(273961))
    delta=after.astype(np.int32)-before
    with ZipFile(speakers.PACK) as pack:
        assert P.sha(speakers.PACK)==P.read(speakers.OUT)["archive_sha256"]
        for angle,checksum in study.ORIENTATIONS.items():
            mat=study.WORK/f"angle_{angle}.0.mat"
            assert P.sha(mat)==checksum
            m=loadmat(mat)["brir"]
            hs={name:resample_poly(m[:,1,ch],1,3)/angles[angle,name]["common_scale"] for name,ch in (("front",2),("back",8))}
            for direction,h in hs.items():
                other=hs["back" if direction=="front" else "front"]
                for i in (77,79,90):
                    end=src[i]["command_at"]
                    dry=speakers.unpack(pack.read(f"{i:03}/early-nyx.wav"))
                    assert np.array_equal(dry[end-len(before):end],before)
                    changed=dry.copy()
                    changed[end-len(before):end]+=delta
                    talk=speakers.unpack(pack.read(f"{i:03}/interruption.wav"))
                    competing=.25*fftconvolve(talk.astype(float),other)
                    noise=rng.normal(0,12,len(competing))
                    for mode,bg in (("quiet",noise),("talk",noise+competing)):
                        label=f"{angle}-{direction}-{i:03}-early-nyx-{mode}"
                        row=registered[label]
                        def digitize(wave):
                            return np.rint(.25*fftconvolve(wave.astype(float),h)+bg).clip(-32768,32767).astype("<i2")
                        assert P.digest(digitize(dry))==row["pcm_sha256"]
                        mic=digitize(changed)
                        yield label, mode, mic, np.zeros_like(mic),row["nyx_open"],row["start"],row["end"]
        rooms,taps=trial.daily.rooms()
        room={r[0]["file"]:r for r in rooms[-6]}
        rates={r["label"]:r for r in P.read(early.PLAN)["records"]}
        for i in (77,79,90):
            dry=speakers.unpack(pack.read(f"{i:03}/early-nyx.wav"))+speakers.unpack(pack.read(f"{i:03}/interruption.wav"))
            end=src[i]["command_at"]
            changed=dry.copy()
            changed[end-len(before):end]+=delta
            _,_,render,ref,_,_=room[src[i]["file"]]
            for tag,speed in trial.pack.study.SPEEDS:
                label=f"{i:03}-multirate-{tag}"
                row=rates[label]
                def room_adc(wave):
                    played=trial.rate_convert(wave,speed)
                    rear=trial.daily.board.bella.rear_filter(played,taps)
                    rear=np.pad(rear,(0,max(0,len(dry)-len(rear))))[:len(dry)]
                    sig,clips=render(rear)
                    assert not clips
                    return sig
                assert P.digest(room_adc(dry))==row["pcm_sha256"] and P.digest(ref)==row["ref_sha256"]
                old=historic["rate/"+label]["candidate"]["events"]["O"]
                yield label,"rate",room_adc(changed),ref,next(iter(old),None),row["start"],row["end"]
    panel=early.playback.playback
    original=sf.read(panel.daily.board.bella.CUE,dtype="int16")[0]
    for key,row,mic,ref in panel.panel(original,after):
        old=historic[key]
        yield key,row["role"],mic,ref,next(iter(old["candidate"]["events"]["O"]),None),None,None


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    before,after,identity=fit()
    meta=P.read(receiver.WORK/"build.json")
    flash=bytearray((receiver.WORK/"guest.flash").read_bytes())
    fixed=receiver.test.fixed
    assert fixed.digest(flash)==meta["flash_sha256"] and P.sha(fixed.QEMU)==meta["qemu_sha256"]
    counts, rows, files=defaultdict(Counter),[],{}
    cache=WORK/meta["flash_sha256"]
    cache.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=WORK) as folder:
        work=Path(folder)
        archive=OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for label,group,mic,ref,old,start,end in scenes(before,after):
                block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                result,log,faults=fixed.boot(work,flash,cache,block,1)
                for p in (log,*faults):
                    name=label+("/wake.log" if p==log else "/"+p.name)
                    logs.write(p,name)
                    files[name]=P.sha(p)
                events=result[0]["events"]
                new=next(iter(events["O"]),None)
                a=old is not None and (start is None or start<=old<end)
                b=new is not None and (start is None or start<=new<end)
                counts[group].update(cases=1,before=int(a),after=int(b),gained=int(b and not a),lost=int(a and not b))
                rows.append(dict(label=label,group=group,start=start,end=end,before=old,events=events,
                                 mic_sha256=P.digest(mic),ref_sha256=P.digest(ref),input_sha256=fixed.digest(block)))
        assert len(rows)==len({r["label"] for r in rows})==47
        P.save(work/OUT.name,dict(records=rows,counts=counts,fit=identity,build=meta,files=files,archive_sha256=P.sha(work/archive.name),
               sources={str(p):P.sha(p) for p in (Path(__file__),PROTO,study.OUT,early.PLAN,early.FIT,speakers.OUT)}))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
