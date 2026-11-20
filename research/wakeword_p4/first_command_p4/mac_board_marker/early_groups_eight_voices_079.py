"""Compare the first-two-groups boost across eight frozen synthetic cues on weak 079."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf
import torch

import package_bella_early_groups_3db as mac
import three_triplet_eight_voices as voices

c, shifts, P = mac.c, mac.shifts, mac.P
OUT = c.FIT.with_name("4090-early-groups-eight-voices-079.json")
PACK = OUT.with_name("hye-nyx-eight-early-groups-3db.zip")
PROTOCOL = c.PROTO.with_name("early_groups_eight_voices_079.md")
WORK = c.WORK/"eight-voices"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def build():
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    old_plan = P.read(voices.PLAN)
    transfer = P.read(voices.transfer.PLAN)
    assert P.sha(voices.PACK)==old_plan["files"][str(voices.PACK)]
    gains = voices.transfer.GAINS
    assert P.sha(gains)==transfer["files"][str(gains)]
    WORK.mkdir(parents=True,exist_ok=True)
    with ZipFile(voices.PACK) as previous, ZipFile(gains) as singles, TemporaryDirectory(dir=WORK) as tmp:
        olds = json.loads(previous.read("manifest.json"))["voices"]
        src = [r for r in json.loads(singles.read("manifest.json"))["records"] if "wake_only" in r]
        meta = {}
        with ZipFile(Path(tmp)/PACK.name,"w") as z:
            for r in src:
                name = r["voice"]
                s = r["wake_only"]
                raw = shifts.member_wave(singles,s["file"],s["sha256"],"int16")
                old = shifts.member_wave(previous,olds[name]["file"],olds[name]["sha256"],"int16")
                base = np.zeros(len(old))
                base[-len(raw):] = np.rint(raw*transfer["voices"][name]["scales"]["single"])
                x = torch.tensor(base,dtype=torch.float64,device="cuda")
                delta = torch.tensor(old,device="cuda")-x
                pos, neg = delta>0,delta<0
                scale = min(10**.15,float(((32767-x[pos])/delta[pos]).min()),float(((-32768-x[neg])/delta[neg]).min()))
                assert scale>=1 and not bool(torch.any(delta[15518:]))
                total = torch.round(x+scale*delta).cpu().numpy()
                assert total.max()<=32767 and total.min()>=-32768
                pcm = total.astype("<i2")
                buf = BytesIO()
                sf.write(buf,pcm,16000,format="WAV",subtype="PCM_16")
                blob = buf.getvalue()
                if name=="af_bella":
                    assert blob==c.WAV.read_bytes()
                shifts.handoff.stereo.write_member(z,name+".wav",blob)
                meta[name]=dict(file=name+".wav",sha256=sha256(blob).hexdigest(),scale=scale,gain_db=20*np.log10(scale),
                                changed=int(np.count_nonzero(pcm!=old)),old_sha256=olds[name]["sha256"],single_sha256=s["sha256"])
            assert len(meta)==8
            deps=(SCRIPT,PROTOCOL,voices.PACK,gains,voices.transfer.PLAN,c.FIT)
            manifest=dict(voices=meta,rate=16000,samples=23200,gpu=device,sources={str(p):P.sha(p) for p in deps})
            shifts.handoff.stereo.write_member(z,"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
        (Path(tmp)/PACK.name).replace(PACK)


def replay():
    geometry = (210,"front",79)
    level = shifts.level
    prev = P.read(shifts.OUT)
    expected = {r["role"]:r for r in prev["records"] if
                (r["angle"],r["direction"],r["scene"])==geometry and r["offset_ms"]==0}
    assert expected["new"]["timely"] and not expected["old"]["timely"]
    p = P.read(level.angles.PLAN)
    scale = P.read(level.study.PLAN)["paths"][0]["common_scale"]
    mat = level.angles.WORK/"angle_210.0.mat"
    assert P.sha(mat)==p["files"][str(mat)]
    ir = loadmat(mat)["brir"]
    target,other=(resample_poly(ir[:,1,ch],1,3)/scale for ch in (2,8))
    arrival = int(np.abs(target[:800]).argmax())
    end = expected["new"]["end"]-arrival
    noise_rng = np.random.Generator(np.random.PCG64(631820))
    sizes = {r["scene"]:r["samples"] for r in P.read(level.OUT)["records"]}
    for angle in level.angles.GIT_BLOBS:
        for side in ("front","back"):
            for scene in (77,79,90):
                n = noise_rng.normal(0,12,sizes[scene])
                if (angle,side,scene)==geometry:
                    noise = n
    receipt=P.read(mac.OUT)
    assert P.sha(mac.PACK)==receipt["archive_sha256"]
    with ZipFile(mac.PACK) as archive:
        stereo = shifts.member_wave(archive,"079/erased.wav",receipt["files"]["079/erased.wav"],"float64")
    bg = fftconvolve(32768*stereo[:,1],other)+noise
    build=prev["build"]
    fixed=shifts.fixed
    flash=bytearray((shifts.source.receiver.WORK/"guest.flash").read_bytes())
    assert fixed.digest(flash)==build["flash_sha256"] and P.sha(fixed.QEMU)==build["qemu_sha256"]
    records,files=[],{}
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(voices.PACK) as old, ZipFile(PACK) as new:
        work=Path(tmp)
        before=json.loads(old.read("manifest.json"))["voices"]
        after=json.loads(new.read("manifest.json"))["voices"]
        cases=[("erased","erased",np.zeros(23200))]
        for name in after:
            for role,z,meta in (("old",old,before),("new",new,after)):
                cue=shifts.member_wave(z,meta[name]["file"],meta[name]["sha256"],"int16")
                cases.append((name,role,cue))
        with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as logs:
            for name,role,cue in cases:
                left = 32768*stereo[:,0]
                left[end-len(cue):end] = shifts.GAIN*cue.astype(float)/4
                total=np.rint(fftconvolve(left,target)+bg)
                mic=total.astype("<i2")
                assert np.array_equal(total,mic)
                ref=np.zeros_like(mic)
                assert P.digest(ref)==expected["new"]["ref_sha256"]
                block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                active_cache=c.WORK/build["flash_sha256"]
                if role=="erased" or (name=="af_bella" and role=="old"):
                    active_cache=level.WORK/build["flash_sha256"]
                for attempt in range(2):
                    try:
                        runs,log,faults=fixed.boot(work,flash,active_cache,block,1)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                ev=runs[0]["events"]
                if name in ("af_bella","erased"):
                    assert P.digest(mic)==expected[role]["pcm_sha256"] and ev==expected[role]["events"]
                opens=ev["O"]
                start=int(np.flatnonzero(cue)[0])+end-len(cue)+arrival if role!="erased" else None
                label=name+"/"+role
                for path in (log,*faults):
                    member=label+("/wake.log" if path==log else "/"+path.name)
                    logs.write(path,member)
                    files[member]=P.sha(path)
                records.append(dict(label=label,voice=name,role=role,events=ev,start=start,end=end+arrival,
                                    timely=bool(opens and start is not None and start<=opens[0]<end+arrival),
                                    pcm_sha256=P.digest(mic),input_sha256=fixed.digest(block)))
        assert len(records)==17
        deps=(SCRIPT,PROTOCOL,voices.PACK,PACK,mac.OUT,shifts.OUT,level.angles.PLAN)
        P.save(work/OUT.name,dict(records=records,voices=after,files=files,build=build,archive_sha256=P.sha(work/"uart.zip"),
                                 sources={str(p):P.sha(p) for p in deps}))
        (work/"uart.zip").replace(OUT.with_suffix(".zip"))
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    {"build":build,"replay":replay}[sys.argv[1]]()
