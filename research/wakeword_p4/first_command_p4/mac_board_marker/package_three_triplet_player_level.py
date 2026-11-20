"""Export the paired, quieter stereo programs and reproduce every measured microphone."""
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

import bella_third_triplet_player_level as level
import package_third_triplet_stereo as handoff

PACK = level.WORK/"hye-nyx-three-triplet-player-level-v1.zip"
OUT = level.OUT.with_stem(level.OUT.stem+"-mac")


def main():
    p = handoff.P
    report = p.read(level.OUT)
    assert p.sha(level.PLAN) == report["plan_sha256"]
    old = p.read(handoff.OUT)
    assert p.sha(handoff.PACK) == old["archive_sha256"]
    member_hashes, clocks = {}, {}
    scale = p.read(level.study.PLAN)["paths"][0]["common_scale"]
    rng = np.random.Generator(np.random.PCG64(631820))
    for orientation in level.angles.GIT_BLOBS:
        path = level.angles.WORK/f"angle_{orientation}.0.mat"
        assert p.sha(path) == p.read(level.PLAN)["files"][str(path)]
        ir = loadmat(path)["brir"]
        front, back = (resample_poly(ir[:,1,c],1,3)/scale for c in (2,8))
        for direction, target, other in (("front",front,back),("back",back,front)):
            for scene in (77,79,90):
                row = next(r for r in report["records"] if (r["angle"],r["direction"],r["scene"]) == (orientation,direction,scene))
                clocks[orientation,direction,scene] = target, other, rng.normal(0,12,row["samples"])
    with ZipFile(handoff.PACK) as src, TemporaryDirectory(dir=level.WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/PACK.name,"w") as z:
            for scene in (77,79,90):
                for role in ("delivered","three-triplets","willow","absent"):
                    name = f"{scene:03}/{role}.wav"
                    raw = src.read(name)
                    assert sha256(raw).hexdigest() == old["files"][name]
                    source, rate = sf.read(BytesIO(raw))
                    assert rate == 16000 and source.shape[1] == 2
                    for mode, right_gain in (("target-minus6",1.),("both-minus6",10**(-6/20))):
                        wave = source * (10**(-6/20), right_gain)
                        buf = BytesIO()
                        wavfile.write(buf,16000,wave)
                        member = f"{mode}/{name}"
                        handoff.stereo.write_member(z,member,buf.getvalue())
                        member_hashes[member] = sha256(buf.getvalue()).hexdigest()
            manifest = dict(files=member_hashes.copy(), sample_rate=16000, channels=["target","interruption"],
                            source_pack_sha256=old["archive_sha256"], gains_db={"target-minus6":[-6,0],"both-minus6":[-6,-6]},
                            playback="Keep master level and positions fixed. Play the original pack and these stereo files on one device without normalization or downmix.")
            raw = (json.dumps(manifest,indent=2)+"\n").encode()
            handoff.stereo.write_member(z,"manifest.json",raw)
            member_hashes["manifest.json"] = sha256(raw).hexdigest()
        with ZipFile(work/PACK.name) as z:
            assert len(z.namelist()) == len(member_hashes) == 25
            for r in report["records"]:
                member = f"{r['mode']}/{r['scene']:03}/{r['role']}.wav"
                raw = z.read(member)
                assert sha256(raw).hexdigest() == member_hashes[member]
                wave, rate = sf.read(BytesIO(raw))
                assert rate == 16000
                target, other, noise = clocks[r["angle"],r["direction"],r["scene"]]
                mic = fftconvolve(32768*wave[:,0],target)+(fftconvolve(32768*wave[:,1],other)+noise)
                assert p.digest(np.rint(mic).clip(-32768,32767).astype("<i2")) == r["pcm_sha256"]
        deps = (level.PLAN,level.OUT,handoff.OUT,Path(__file__).resolve().relative_to(Path.cwd()))
        p.save(work/OUT.name,dict(url=old["url"].rsplit("/",1)[0]+"/"+PACK.name,bytes=(work/PACK.name).stat().st_size,
                                  archive_sha256=p.sha(work/PACK.name),files=member_hashes,matched_mics=len(report["records"]),
                                  sources={str(dep):p.sha(dep) for dep in deps}))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
