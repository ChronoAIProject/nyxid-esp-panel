"""Extend the verified two-speaker pack with the frozen headroom WAV and its three paired commands."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_marker_headroom as fit

P = fit.P
PACK = fit.WORK/"hye-nyx-headroom-mac-ab-v1.zip"
OUT = fit.OUT.with_name("4090-bella-headroom-mac-download.json")


def main():
    base = P.read(fit.speakers.OUT)
    meta = P.read(fit.OUT)
    assert P.sha(fit.speakers.PACK)==base["archive_sha256"]
    old,sr=sf.read(fit.early.WAV,dtype="int16")
    candidate,rr=sf.read(fit.WAV,dtype="int16")
    assert sr==rr==16000 and P.sha(fit.WAV)==meta["fit"]["cue_sha256"]
    assert P.sha(fit.early.WAV)==meta["fit"]["original_cue_sha256"]
    fresh=fit.study.OUT.with_name("4090-bella-headroom-new-angles-plan.json")
    rows=P.read(fresh)["records"]
    with TemporaryDirectory(dir=fit.WORK) as temp:
        work=Path(temp)
        with ZipFile(fit.speakers.PACK) as source, ZipFile(work/PACK.name,"w",ZIP_DEFLATED) as dest:
            manifest=json.loads(source.read("manifest.json"))
            for k,r in manifest["files"].items():
                raw=source.read(k)
                assert fit.study.sha256(raw).hexdigest()==r["sha256"]==base["files"][k]
                dest.writestr(k,raw)
            for i in (77,79,90):
                name=f"{i:03}/early-nyx.wav"
                pcm=fit.speakers.unpack(source.read(name))
                info=manifest["files"][name]
                end=info["command_at"]
                assert np.array_equal(pcm[end-len(old):end],old)
                pcm[end-len(old):end]=candidate
                raw=fit.early.playback.playback.sources.wav(pcm*16384,"PCM_24")
                assert np.array_equal(fit.speakers.unpack(raw),pcm)
                key=f"{i:03}/headroom.wav"
                dest.writestr(key,raw)
                manifest["files"][key]=dict(info,sha256=fit.study.sha256(raw).hexdigest(),bytes=len(raw))
            for path in (fit.early.WAV, fit.WAV):
                raw=path.read_bytes()
                dest.writestr(path.name,raw)
                manifest["files"][path.name]=dict(sha256=fit.study.sha256(raw).hexdigest(),bytes=len(raw),
                                                   type="isolated 1.45 s wake at released PCM16 level")
            manifest["instructions"]+=" Within the same scene compare early-nyx (delivered) and headroom (candidate). " \
                "Both use the same board app. The two root WAVs are standalone PCM16 cues."
            manifest["receiver_manifest"]=fit.trial.pack.RELEASE+"hye-nyx-reference-rearm-p4-public-v1.zip"
            raw=(json.dumps(manifest,indent=2)+"\n").encode()
            dest.writestr("manifest.json",raw)
        files={k:r["sha256"] for k,r in manifest["files"].items()}
        files["manifest.json"]=fit.study.sha256(raw).hexdigest()
        with ZipFile(work/PACK.name) as z:
            assert len(z.namelist())==len(files)==21
            assert all(fit.study.sha256(z.read(k)).hexdigest()==v for k,v in files.items())
            # Check the packaged candidate on all 12 independently rendered validation microphones.
            from scipy.io import loadmat
            from scipy.signal import fftconvolve, resample_poly
            import bella_headroom_new_angles as validation
            rng=np.random.Generator(np.random.PCG64(927463))
            common=P.read(fit.study.PLAN)["paths"][0]["common_scale"]
            expected={r["label"]:r for r in rows}
            for angle,hsh in validation.ANGLES.items():
                mat=validation.WORK/f"angle_{angle}.0.mat"
                assert P.sha(mat)==hsh
                mat_data=loadmat(mat)["brir"]
                hs={k:resample_poly(mat_data[:,1,c],1,3)/common for k,c in (("front",2),("back",8))}
                for side,h in hs.items():
                    other=hs["back" if side=="front" else "front"]
                    for i in (77,79,90):
                        talk=fit.speakers.unpack(z.read(f"{i:03}/interruption.wav"))
                        target=fit.speakers.unpack(z.read(f"{i:03}/headroom.wav"))
                        distant=.25*fftconvolve(talk.astype(float),other)
                        bg=distant+rng.normal(0,12,len(distant))
                        pcm=np.rint(.25*fftconvolve(target.astype(float),h)+bg).clip(-32768,32767).astype("<i2")
                        assert P.digest(pcm)==expected[f"{angle}-{side}-{i:03}-headroom"]["pcm_sha256"]
        P.save(work/OUT.name,dict(url=fit.trial.pack.RELEASE+PACK.name,archive_sha256=P.sha(work/PACK.name),
               bytes=(work/PACK.name).stat().st_size,files=files,matched_additional_mics=12,sources={str(p):P.sha(p)
               for p in (Path(__file__),fit.speakers.OUT,fit.OUT,fresh)}))
        (work/OUT.name).replace(OUT)
        (work/PACK.name).replace(PACK)


if __name__=="__main__":
    main()
