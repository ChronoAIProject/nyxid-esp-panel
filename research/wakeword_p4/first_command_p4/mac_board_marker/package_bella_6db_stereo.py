"""Export a synchronized two-output player file and check both measured microphones."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import av
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import bella_240_masking as sources
import bella_6db_player as player

P = sources.P
PACK = player.WORK / "hye-nyx-stereo-player-6db-v1.zip"
OUT = player.OUT.with_name("4090-bella-6db-stereo-mac.json")
SOURCE = sources.pack
GAIN = 10**(6/20)


def write_member(z, name, raw):
    z.writestr(ZipInfo(name, (2026, 11, 20, 0, 0, 0)), raw, compress_type=ZIP_DEFLATED)


def main():
    checked = 0
    members = {}
    assert P.sha(SOURCE.PACK) == P.read(SOURCE.OUT)["archive_sha256"]
    base = {r["label"]: r for r in P.read(sources.angles.PLAN)["records"]}
    louder = {r["baseline_label"]: r for r in P.read(player.PLAN)["records"]}
    roles = ("early-nyx", "headroom", "willow", "absent")
    scale = P.read(sources.fit.study.PLAN)["paths"][0]["common_scale"]
    release = sources.fit.trial.pack.RELEASE
    with TemporaryDirectory(dir=player.WORK) as folder, ZipFile(SOURCE.PACK) as src:
        work = Path(folder)
        with ZipFile(work/PACK.name, "w") as z:
            for scene in (77, 79, 90):
                right = sources.fit.speakers.unpack(src.read(f"{scene:03}/interruption.wav")) / 131072
                for role in roles:
                    member = f"{scene:03}/{role}.wav"
                    left = sources.fit.speakers.unpack(src.read(member)) / 131072
                    for level, gain in (("baseline", 1.), ("plus6", GAIN)):
                        name = f"{scene:03}/{role}-{level}.wav"
                        pcm = np.column_stack((gain*left, right))
                        assert np.max(np.abs(pcm)) < 1
                        buf = BytesIO()
                        wavfile.write(buf, 16000, pcm)
                        raw = buf.getvalue()
                        with av.open(BytesIO(raw)) as container:
                            frames = [f.to_ndarray().reshape(-1, 2) for f in container.decode(audio=0)]
                        assert np.array_equal(np.concatenate(frames), pcm)
                        write_member(z, name, raw)
                        members[name] = dict(sha256=player.study.sha256(raw).hexdigest(), bytes=len(raw),
                                             target_source=member, interruption_source=f"{scene:03}/interruption.wav",
                                             target_gain_db=6 if level == "plus6" else 0, samples=len(pcm))
            manifest = dict(sample_rate=16000, encoding="stereo IEEE float64 WAV", left="target speaker",
                            right="opposite speaker, unchanged", source_archive_sha256=P.sha(SOURCE.PACK), files=members,
                            playback="Use one two-output device. Route left and right to separate speakers with the "
                                     "panel silent. Use the same master level throughout. The gain step is already "
                                     "in the left samples. Disable normalization, downmix and spatial processing.")
            raw = (json.dumps(manifest, indent=2) + "\n").encode()
            write_member(z, "manifest.json", raw)
        hashes = {k: v["sha256"] for k, v in members.items()}
        hashes["manifest.json"] = player.study.sha256(raw).hexdigest()
        rng = np.random.Generator(np.random.PCG64(927463))
        with ZipFile(work/PACK.name) as z:
            assert len(z.namelist()) == len(hashes) == 25 and z.testzip() is None
            assert {k: player.study.sha256(z.read(k)).hexdigest() for k in z.namelist()} == hashes
            for angle in (120, 240):
                mat = sources.angles.WORK / f"angle_{angle}.0.mat"
                assert P.sha(mat) == sources.angles.ANGLES[angle]
                brir = loadmat(mat)["brir"]
                paths = {k: resample_poly(brir[:, 1, ch], 1, 3)/scale for k, ch in (("front", 2), ("back", 8))}
                for side, h in paths.items():
                    other = paths["back" if side == "front" else "front"]
                    for scene in (77, 79, 90):
                        first = base[f"{angle}-{side}-{scene:03}-early-nyx"]
                        noise = rng.normal(0, 12, first["samples"])
                        for role in roles:
                            label = f"{angle}-{side}-{scene:03}-{role}"
                            for level, expected in (("baseline", base[label]), ("plus6", louder[label])):
                                wav, rate = sf.read(BytesIO(z.read(f"{scene:03}/{role}-{level}.wav")))
                                assert rate == 16000
                                target = fftconvolve(32768*wav[:, 0], h)
                                talk = fftconvolve(32768*wav[:, 1], other)
                                mic = np.rint(target + (talk + noise)).clip(-32768, 32767).astype("<i2")
                                assert P.digest(mic) == expected["pcm_sha256"], (label, level)
                                checked += 1
        assert checked == 96
        deps = (SOURCE.OUT, player.PLAN, sources.angles.PLAN, Path(__file__).resolve().relative_to(Path.cwd()))
        P.save(work/OUT.name, dict(url=release+PACK.name, bytes=(work/PACK.name).stat().st_size,
                                   archive_sha256=P.sha(work/PACK.name), files=hashes, reproduced_mics=checked,
                                   sources={str(p): P.sha(p) for p in deps}))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
