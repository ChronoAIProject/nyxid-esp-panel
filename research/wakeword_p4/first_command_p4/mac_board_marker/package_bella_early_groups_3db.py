"""Export a same-clock 079 board comparison at the tested weak-target player level."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import verify_bella_early_groups_3db as verify

c, shifts, P = verify.c, verify.shifts, verify.P
level, handoff = shifts.level, shifts.handoff
PACK = c.WORK/"hye-nyx-079-early-groups-3db-v1.zip"
OUT = c.FIT.with_name("4090-bella-early-groups-3db-mac.json")


def main():
    assert P.read(verify.OUT)["history_gate"]
    measured = P.read(shifts.OUT)
    clocks = {(r["angle"],r["direction"],r["role"]):r["pcm_sha256"] for r in measured["records"]
              if r["scene"]==79 and r["offset_ms"]==0}
    old = P.read(level.OUT)
    willow = { (r["angle"],r["direction"]):r for r in old["records"] if
              (r["scene"],r["role"],r["mode"])==(79,"willow","target-minus6")}
    parent = P.read(handoff.OUT)
    assert P.sha(handoff.PACK)==parent["archive_sha256"]
    base, sr = sf.read(c.OLD,dtype="int16")
    new, nr = sf.read(c.WAV,dtype="int16")
    assert sr==nr==16000 and P.sha(c.WAV)==P.read(c.FIT)["voices"][c.NAME]
    reference = willow[210,"front"]
    plan = P.read(level.angles.PLAN)
    geometry, = (r for r in plan["paths"] if (r["angle"],r["direction"])==(210,"front"))
    end = reference["end"]-geometry["arrival"]
    address = slice(end-len(base),end)
    hashes = {}
    def add(archive,name,data):
        handoff.stereo.write_member(archive,name,data)
        hashes[name]=sha256(data).hexdigest()
    with TemporaryDirectory(dir=c.WORK) as tmp:
        zip_path = Path(tmp)/PACK.name
        with ZipFile(handoff.PACK) as sources, ZipFile(zip_path,"w") as out:
            add(out,"wake/old.wav",c.OLD.read_bytes())
            add(out,"wake/new.wav",c.WAV.read_bytes())
            for role,src in (("old","three-triplets"),("new","three-triplets"),("erased","absent"),("willow","willow")):
                name = f"079/{src}.wav"
                wave = shifts.member_wave(sources,name,parent["files"][name],"float64")
                if role=="new":
                    np.testing.assert_array_equal(wave[address,0]*131072,base)
                    wave[address,0] = new.astype(float)/131072
                wave[:,0] *= shifts.GAIN
                assert np.max(np.abs(wave))<1
                buf = BytesIO()
                wavfile.write(buf,16000,wave)
                add(out,"079/"+role+".wav",buf.getvalue())
            meta=dict(files=hashes.copy(),sample_rate=16000,scene=79,command=reference["command"],
                      spoken_command_at=end,source_archive_sha256=parent["archive_sha256"],
                      playback="Play one 079 stereo file per trial on one device, left target, right interfering speaker. "
                               "The target is already at -6 dB. Keep master level and positions fixed. "
                               "Use Nyx firmware for old, new and erased, and official Hey Willow for willow. "
                               "wake/ holds standalone full-level mono cues for identification and clock analysis.")
            add(out,"manifest.json",(json.dumps(meta,indent=2)+"\n").encode())
        scale = P.read(level.study.PLAN)["paths"][0]["common_scale"]
        rng = np.random.Generator(np.random.PCG64(631820))
        sizes = {r["scene"]:r["samples"] for r in old["records"]}
        checked = 0
        with ZipFile(zip_path) as z:
            assert len(hashes)==len(z.namelist())==7 and all(sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())
            for angle in level.angles.GIT_BLOBS:
                mat = level.angles.WORK/f"angle_{angle}.0.mat"
                assert P.sha(mat)==plan["files"][str(mat)]
                ir = loadmat(mat)["brir"]
                front, back = (resample_poly(ir[:,1,ch],1,3)/scale for ch in (2,8))
                for side,h,g in (("front",front,back),("back",back,front)):
                    for scene in (77,79,90):
                        noise = rng.normal(0,12,sizes[scene])
                        if scene!=79:
                            continue
                        for role in ("old","new","erased","willow"):
                            name = "079/"+role+".wav"
                            wave = shifts.member_wave(z,name,hashes[name],"float64")
                            mic = fftconvolve(32768*wave[:,0],h)+(fftconvolve(32768*wave[:,1],g)+noise)
                            expected = willow[angle,side]["pcm_sha256"] if role=="willow" else clocks[angle,side,role]
                            assert P.digest(np.rint(mic).clip(-32768,32767).astype("<i2"))==expected
                            checked += 1
        assert checked==16
        deps = (Path(__file__).resolve().relative_to(Path.cwd()),verify.OUT,shifts.OUT,level.OUT,handoff.OUT,c.FIT)
        P.save(Path(tmp)/OUT.name,dict(url=parent["url"].rsplit("/",1)[0]+"/"+PACK.name,files=hashes,
                                       bytes=zip_path.stat().st_size,archive_sha256=P.sha(zip_path),matched_mics=checked,
                                       sources={str(p):P.sha(p) for p in deps}))
        zip_path.replace(PACK)
        (Path(tmp)/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
