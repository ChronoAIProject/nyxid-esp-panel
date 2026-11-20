"""Export synchronized target and interruption tracks for a two-loudspeaker P4 check."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import bella_surrey_orientations as study
import bella_delayed_command as trial

P = trial.P
PACK = study.WORK / "hye-nyx-two-speaker-orientation-tracks-v1.zip"
OUT = study.OUT.with_name("4090-bella-surrey-speaker-download.json")


def unpack(raw):
    samples, sr = sf.read(BytesIO(raw),dtype="int32")
    assert sr == 16000 and samples.ndim == 1 and np.all(samples % 16384 == 0)
    return samples//16384


def main():
    plan = P.read(study.PLAN)
    source = P.read(trial.pack.OUT)
    assert P.sha(trial.pack.PACK) == source["archive_sha256"]
    daily = {r["index"]:r for r in P.read(trial.daily.OUT)["records"] if r["role"]=="original"}
    rooms, _ = trial.daily.rooms()
    clean = {r[0]["file"]:r[-1]["absent"].astype(np.int32) for r in rooms[-6]}
    inputs = {r["label"]:r for r in plan["records"]}
    paths = {(r["angle"],r["direction"]):r for r in plan["paths"]}
    rng = np.random.Generator(np.random.PCG64(273961))
    manifest = dict(sample_rate=16000, encoding="mono PCM24 WAV at the common quarter-scale player gain",
                    files={}, instructions="Choose one target WAV and play the interruption WAV from the opposite speaker. "
                    "Start both files at sample zero. For quiet, mute the interruption. Keep gain, board position and "
                    "sample clocks fixed across Nyx, Willow and address-erased trials. The panel stays silent. "
                    "The released files carry dry playback audio. The simulated microphone also had the measured room "
                    "response and low-level seeded microphone noise.")
    with TemporaryDirectory(dir=study.WORK) as folder, ZipFile(trial.pack.PACK) as base:
        work=Path(folder)
        with ZipFile(work/PACK.name,"w",ZIP_DEFLATED) as pack:
            for scene in (77,79,90):
                prefix=f"{scene:03}/"
                src=daily[scene]
                mixer=prefix+"nominal/"
                erased=mixer+"absent.wav"
                raw=base.read(erased)
                assert study.sha256(raw).hexdigest()==source["files"][erased]
                talk=unpack(raw)-clean[src["file"]]
                waves={"interruption":talk}
                for role in ("early-nyx","nyx","willow","absent"):
                    key=mixer+role+".wav"
                    raw=base.read(key)
                    assert study.sha256(raw).hexdigest()==source["files"][key]
                    waves[role]=unpack(raw)-talk
                for role,pcm in waves.items():
                    key=prefix+role+".wav"
                    assert np.max(np.abs(pcm))<131072
                    raw=trial.pack.early.playback.playback.sources.wav(pcm*16384,"PCM_24")
                    assert np.array_equal(unpack(raw),pcm)
                    pack.writestr(key,raw)
                    manifest["files"][key]=dict(bytes=len(raw),sha256=study.sha256(raw).hexdigest(), samples=len(pcm),
                                               command=src["command"],command_at=src["command_at"])
            meta=(json.dumps(manifest,indent=2)+"\n").encode()
            pack.writestr("manifest.json",meta)
        checked=0
        with ZipFile(work/PACK.name) as pack:
            assert pack.testzip() is None and len(pack.namelist())==16
            for angle,checksum in study.ORIENTATIONS.items():
                mat=study.WORK/f"angle_{angle}.0.mat"
                assert P.sha(mat)==checksum
                m=loadmat(mat)["brir"]
                scale=paths[angle,"front"]["common_scale"]
                h={k:resample_poly(m[:,1,c],1,3)/scale for k,c in (("front",2),("back",8))}
                for direction,response in h.items():
                    other=h["back" if direction=="front" else "front"]
                    for scene in (77,79,90):
                        talk=unpack(pack.read(f"{scene:03}/interruption.wav"))
                        noisy=.25*fftconvolve(talk.astype(float),other)
                        noise=rng.normal(0,12,len(noisy))
                        for role in ("early-nyx","nyx","willow","absent"):
                            wav=unpack(pack.read(f"{scene:03}/{role}.wav"))
                            wet=.25*fftconvolve(wav.astype(float),response)
                            for mode, bg in (("quiet",noise),("talk",noise+noisy)):
                                mic=np.rint(wet+bg).clip(-32768,32767).astype("<i2")
                                label=f"{angle}-{direction}-{scene:03}-{role}-{mode}"
                                assert P.digest(mic)==inputs[label]["pcm_sha256"]
                                checked+=1
        assert checked==len(inputs)==96
        P.save(work/OUT.name,dict(url=trial.pack.RELEASE+PACK.name,archive_sha256=P.sha(work/PACK.name),
               bytes=(work/PACK.name).stat().st_size, matched_mic_count=checked, files={**{k:v["sha256"] for k,v in manifest["files"].items()},
               "manifest.json":study.sha256(meta).hexdigest()}, sources={str(p):P.sha(p) for p in (Path(__file__),study.PLAN,trial.pack.OUT)}))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
