"""Export the original-level physical A/candidate comparison on one stereo clock."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import check_third_triplet_collisions as chosen
import package_bella_6db_stereo as stereo

triple, P = chosen.triple, chosen.P
PACK = triple.WORK/"hye-nyx-three-triplet-mac-v1.zip"
OUT = triple.OUT.with_name("4090-bella-third-triplet-mac.json")


def verify_mics(z, expected, sources):
    fit=chosen.fit
    fresh=triple.experiment.source.angles
    scale=P.read(fit.study.PLAN)["paths"][0]["common_scale"]
    checks=0
    for orientations, folder, seed in ((fit.study.ORIENTATIONS,fit.study.WORK,273961),(fresh.ANGLES,fresh.WORK,927463)):
        rng=np.random.Generator(np.random.PCG64(seed))
        for angle,checksum in orientations.items():
            mat=folder/f"angle_{angle}.0.mat"
            assert P.sha(mat)==checksum
            brir=loadmat(mat)["brir"]
            paths={k:resample_poly(brir[:,1,ch],1,3)/scale for k,ch in (("front",2),("back",8))}
            for side,h in paths.items():
                other=paths["back" if side=="front" else "front"]
                for scene in (77,79,90):
                    key=f"{scene:03}/three-triplets.wav"
                    pcm,rate=sf.read(BytesIO(z.read(key)))
                    assert rate==16000
                    talk=fftconvolve(32768*pcm[:,1],other)
                    noise=rng.normal(0,12,len(talk))
                    mic=np.rint(fftconvolve(32768*pcm[:,0],h)+(talk+noise)).clip(-32768,32767).astype("<i2")
                    label=f"{angle}-{side}-{scene:03}-early-nyx"+("-talk" if angle in fit.study.ORIENTATIONS else "")
                    assert P.digest(mic)==expected[label]["mic_sha256"]
                    base,_=sf.read(BytesIO(z.read(f"{scene:03}/delivered.wav")))
                    assert np.array_equal(base[:,1],pcm[:,1])
                    command_at=sources[f"{scene:03}/early-nyx.wav"]["command_at"]
                    assert np.array_equal(base[command_at:],pcm[command_at:])
                    checks+=1
    return checks


def main():
    candidate=triple.wave(chosen.NAME)
    limited,sr=sf.read(candidate,dtype="int16")
    old,rr=sf.read(chosen.fit.early.WAV,dtype="int16")
    assert sr==rr==16000 and limited.shape==old.shape and P.sha(candidate)==P.read(triple.OUT)["voices"][chosen.NAME]
    assert P.sha(stereo.PACK)==P.read(stereo.OUT)["archive_sha256"] and P.sha(stereo.SOURCE.PACK)==P.read(stereo.SOURCE.OUT)["archive_sha256"]
    rows=P.read(chosen.REPLAY)["records"]
    expected={r["label"]:r for r in rows if r["group"] in ("talk","new_angles")}
    files={}
    with TemporaryDirectory(dir=triple.WORK) as folder, ZipFile(stereo.PACK) as previous, ZipFile(stereo.SOURCE.PACK) as sources:
        work=Path(folder)
        meta=json.loads(sources.read("manifest.json"))["files"]
        with ZipFile(work/PACK.name,"w") as z:
            def add(name, raw):
                stereo.write_member(z,name,raw)
                files[name]=dict(bytes=len(raw),sha256=chosen.fit.study.sha256(raw).hexdigest())
            add("wake/three-triplets.wav",candidate.read_bytes())
            add("wake/delivered.wav",chosen.fit.early.WAV.read_bytes())
            for scene in (77,79,90):
                prefix=f"{scene:03}/"
                for role,name in (("early-nyx","delivered"),("willow","willow"),("absent","absent")):
                    key=prefix+role+"-baseline.wav"
                    raw=previous.read(key)
                    add(prefix+name+".wav",raw)
                    if role=="early-nyx":
                        pcm,rate=sf.read(BytesIO(raw))
                        end=meta[prefix+"early-nyx.wav"]["command_at"]
                        assert rate==16000 and np.array_equal(pcm[end-len(old):end,0]*131072,old)
                        pcm[end-len(old):end,0]=limited.astype(float)/131072
                        blob=BytesIO()
                        wavfile.write(blob,16000,pcm)
                        add(prefix+"three-triplets.wav",blob.getvalue())
            manifest=dict(files=files.copy(),candidate_sha256=P.sha(candidate),baseline_sha256=P.sha(chosen.fit.early.WAV),
                          original_pack_sha256=P.sha(stereo.PACK),sample_rate=16000,
                          playback="Root wake/ contains standalone mono PCM16 cues. Each numbered directory contains "
                                   "stereo float64 WAVs at the original player level. Route left to the target speaker "
                                   "and right to the opposing speaker on a single device. Keep master volume and "
                                   "position fixed and disable normalization and downmix. The panel stays silent.")
            add("manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
        with ZipFile(work/PACK.name) as z:
            assert len(files)==len(z.namelist())==15 and z.testzip() is None
            assert all(chosen.fit.study.sha256(z.read(k)).hexdigest()==v["sha256"] for k,v in files.items())
            checked=verify_mics(z,expected,meta)
        assert checked==len(expected)==24
        deps=(triple.PROTO,triple.OUT,chosen.REPLAY,stereo.OUT,stereo.SOURCE.OUT,Path(__file__).resolve().relative_to(Path.cwd()))
        P.save(work/OUT.name,dict(url=stereo.sources.fit.trial.pack.RELEASE+PACK.name,archive_sha256=P.sha(work/PACK.name),
                                  bytes=(work/PACK.name).stat().st_size,files={k:v["sha256"] for k,v in files.items()},
                                  reproduced_target_mics=checked,sources={str(p):P.sha(p) for p in deps}))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
