"""Replay the frozen stereo handoff with lower target and common output levels."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import bella_third_triplet_new_angles as angles

study = angles.study
WORK = angles.WORK.parent/"player-level"
OUT = study.OUT.with_name("4090-bella-third-triplet-player-level.json")
PLAN = OUT.with_stem(OUT.stem+"-plan")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = angles.PROTOCOL.with_name("bella_third_triplet_player_level_protocol.md")


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf
    import package_third_triplet_stereo as handoff

    p = handoff.P
    base = p.read(angles.PLAN)
    measured = p.read(angles.OUT)
    assert measured["plan_sha256"] == p.sha(angles.PLAN)
    receipt = p.read(handoff.OUT)
    assert p.sha(handoff.PACK) == receipt["archive_sha256"]
    files = {f: base["files"][f] for f in base["runtime"].values()}
    for dep in (SCRIPT, PROTOCOL, angles.PLAN, angles.OUT, handoff.PACK, handoff.OUT, study.PLAN):
        files[str(dep)] = p.sha(dep)
    baseline = {r["label"]: r for r in measured["records"]}
    scale = p.read(study.PLAN)["paths"][0]["common_scale"]
    rng = np.random.Generator(np.random.PCG64(631820))
    reduced = 10**(-6/20)
    rows = []
    WORK.mkdir(exist_ok=True, parents=True)
    with ZipFile(handoff.PACK) as pack:
        for angle in angles.GIT_BLOBS:
            mat = angles.WORK/f"angle_{angle}.0.mat"
            assert p.sha(mat) == base["files"][str(mat)]
            files[str(mat)] = base["files"][str(mat)]
            brir = loadmat(mat)["brir"]
            front, back = (resample_poly(brir[:, 1, ch], 1, 3)/scale for ch in (2, 8))
            for direction, target, other in (("front", front, back), ("back", back, front)):
                for scene in (77, 79, 90):
                    noise = None
                    for role in ("delivered", "three-triplets", "willow", "absent"):
                        old_label = f"{angle}-{direction}-{scene:03}-{role}"
                        old = baseline[old_label]
                        member = f"{scene:03}/{role}.wav"
                        raw = pack.read(member)
                        assert sha256(raw).hexdigest() == receipt["files"][member]
                        stereo, rate = sf.read(BytesIO(raw))
                        assert rate == 16000 and stereo.shape[1] == 2
                        left = fftconvolve(32768*stereo[:, 0], target)
                        right = fftconvolve(32768*stereo[:, 1], other)
                        if noise is None:
                            noise = rng.normal(0, 12, len(left))
                        before = np.rint(left + (right + noise)).clip(-32768,32767).astype("<i2")
                        assert p.digest(before) == old["pcm_sha256"]
                        files[old["reference_path"]] = base["files"][old["reference_path"]]
                        for mode, right_gain in (("target-minus6", 1.), ("both-minus6", reduced)):
                            label = old_label + "-" + mode
                            total = np.rint(reduced*left + (right_gain*right + noise))
                            pcm = total.clip(-32768,32767).astype("<i2")
                            path = WORK/(label+".wav")
                            sf.write(path, pcm, 16000, subtype="PCM_16")
                            files[str(path)] = p.sha(path)
                            row = {k: old[k] for k in ("scene","role","angle","direction","command","start","end",
                                                       "reference_path","ref_sha256","samples")}
                            rows.append(dict(row, label=label, mode=mode, path=str(path), clipped=int(np.count_nonzero(total!=pcm)),
                                             sha256=files[str(path)], pcm_sha256=p.digest(pcm), parent_label=old_label,
                                             parent_pcm_sha256=old["pcm_sha256"]))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 96
    p.save(PLAN, dict(records=rows, files=files, runtime=base["runtime"], threshold_requests=["DEFAULT"]))


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
