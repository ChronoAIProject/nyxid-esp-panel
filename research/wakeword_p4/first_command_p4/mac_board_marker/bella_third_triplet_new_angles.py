"""Play the issued Mac stereo programs through two additional measured orientations."""
from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import bella_surrey_orientations as study

WORK = Path(".local/wakeword/bella-third-triplet/new-angles")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
OUT = study.OUT.with_name("4090-bella-third-triplet-new-angles.json")
PLAN = OUT.with_stem(OUT.stem + "-plan")
PROTOCOL = study.PROTOCOL.with_name("bella_third_triplet_new_angles_protocol.md")
GIT_BLOBS = {60: "77a580b5abe8ccd6854ad22fd16232f1604c83be",
             210: "20ce85272b5d24346d6b1d0333a7c7d2183977ac"}


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf

    import package_third_triplet_stereo as handoff
    import surrey_fixed_cues as surrey

    p = handoff.P
    receipt = p.read(handoff.OUT)
    assert p.sha(handoff.PACK) == receipt["archive_sha256"]
    parent = p.read(study.PLAN)
    scenes = {r["scene"]: r for r in parent["records"] if r["role"] == "early-nyx"}
    timing = p.read(handoff.chosen.fit.trial.daily.OUT)
    ends = {row["index"]: row["command_at"] for row in timing["records"] if row["role"] == "original"}
    scale = parent["paths"][0]["common_scale"]
    files = {f: parent["files"][f] for f in parent["runtime"].values()}
    for dep in (SCRIPT, PROTOCOL, handoff.PACK, handoff.OUT, study.PLAN, handoff.chosen.fit.trial.daily.OUT):
        files[str(dep)] = p.sha(dep)
    rows, paths = [], []
    rng = np.random.Generator(np.random.PCG64(631820))
    WORK.mkdir(parents=True, exist_ok=True)
    with ZipFile(handoff.PACK) as pack:
        for angle, git_blob in GIT_BLOBS.items():
            mat = WORK / f"angle_{angle}.0.mat"
            url = surrey.DATA_URL.replace("angle_0.0.mat", mat.name)
            if mat.exists():
                raw = mat.read_bytes()
            else:
                with urlopen(url, timeout=60) as response:
                    raw = response.read()
            assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == git_blob
            if not mat.exists():
                mat.write_bytes(raw)
            files[str(mat)] = sha256(raw).hexdigest()
            brir = loadmat(mat)["brir"]
            assert brir.shape == (65536, 2, 24)
            front, back = (resample_poly(brir[:, 1, ch], 1, 3) / scale for ch in (2, 8))
            for direction, target, other in (("front", front, back), ("back", back, front)):
                arrival = int(np.abs(target[:800]).argmax())
                paths.append(dict(angle=angle, direction=direction, source_url=url, git_blob=git_blob,
                                  sha256=files[str(mat)], arrival=arrival, rir_sha256=p.digest(target)))
                for scene in (77, 79, 90):
                    noise = None
                    for role in ("delivered", "three-triplets", "willow", "absent"):
                        member = f"{scene:03}/{role}.wav"
                        raw = pack.read(member)
                        assert sha256(raw).hexdigest() == receipt["files"][member]
                        wave, rate = sf.read(BytesIO(raw))
                        assert rate == 16000 and wave.shape[1] == 2
                        speech = fftconvolve(32768 * wave[:, 0], target)
                        competing = fftconvolve(32768 * wave[:, 1], other)
                        if noise is None:
                            noise = rng.normal(0, 12, len(speech))
                        total = np.rint(speech + (competing + noise))
                        mic = total.clip(-32768, 32767).astype("<i2")
                        end = ends[scene]
                        active = np.flatnonzero(wave[:end, 0])
                        assert bool(active.size) == (role != "absent")
                        start = int(active[0]) + arrival if active.size else None
                        label = f"{angle}-{direction}-{scene:03}-{role}"
                        wav = WORK / (label + ".wav")
                        sf.write(wav, mic, 16000, subtype="PCM_16")
                        ref = Path(scenes[scene]["reference_path"])
                        assert p.sha(ref) == parent["files"][str(ref)] and sf.info(ref).frames == len(mic)
                        files[str(ref)] = parent["files"][str(ref)]
                        files[str(wav)] = p.sha(wav)
                        rows.append(dict(label=label, role=role, scene=scene, angle=angle, direction=direction,
                                         mode="talk", start=start, end=end+arrival, samples=len(mic),
                                         command=scenes[scene]["command"], path=str(wav), sha256=files[str(wav)],
                                         pcm_sha256=p.digest(mic), reference_path=str(ref), ref_sha256=scenes[scene]["ref_sha256"],
                                         clipped=int(np.count_nonzero(mic != total))))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 48
    p.save(PLAN, dict(records=rows, paths=paths, files=files, runtime=parent["runtime"], threshold_requests=["DEFAULT"]))


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage == "prepare":
        prepare()
    elif stage == "case":
        sys.path.insert(0, str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    else:
        study.WORK, study.PLAN, study.OUT = WORK, PLAN, OUT
        {"nyx": study.nyx, "report": study.report}[stage]()
        if stage == "report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
