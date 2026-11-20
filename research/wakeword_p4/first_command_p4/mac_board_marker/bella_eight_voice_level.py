"""Apply the known target-player step to the 33 registered Surrey voice-transfer inputs."""
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import bella_eight_voice_transfer as transfer

study, air = transfer.study, transfer.air
WORK = transfer.WORK/"plus6"
OUT = transfer.OUT.with_name("4090-bella-eight-voice-level.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
PROTO = transfer.PROTO.with_name("bella_eight_voice_level_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import package_bella_6db_stereo as stereo

    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    P = stereo.P
    previous = P.read(transfer.PLAN)
    loud = {r["baseline_label"]:r for r in P.read(stereo.player.PLAN)["records"]}
    files = {v: previous["files"][v] for v in previous["runtime"].values()}
    deps = (SCRIPT, PROTO, transfer.PLAN, transfer.OUT, transfer.PACK, transfer.GAINS, transfer.PAIRS,
            stereo.PACK, stereo.OUT, stereo.player.PLAN, air.SCRIPT)
    for p in deps:
        files[str(p)] = P.sha(p)
    for p in (transfer.GAINS, transfer.PAIRS, transfer.PACK):
        assert files[str(p)] == previous["files"][str(p)]
    assert P.sha(stereo.PACK) == P.read(stereo.OUT)["archive_sha256"]
    WORK.mkdir(exist_ok=True, parents=True)
    controls = stereo.sources
    mat = controls.angles.WORK/"angle_240.0.mat"
    assert P.sha(mat) == controls.angles.ANGLES[240]
    brir = loadmat(mat)["brir"]
    scale = P.read(controls.fit.study.PLAN)["paths"][0]["common_scale"]
    ir = [resample_poly(brir[:,1,ch],1,3)/scale for ch in (2,8)]
    noise = next(n for r, _, _, _, n in controls.stems_for({240}, ("early-nyx",))
                 if r["direction"] == "front" and r["scene"] == 77)
    rows = []
    with ZipFile(stereo.PACK) as playback, ZipFile(transfer.PACK) as double, ZipFile(transfer.GAINS) as single, ZipFile(transfer.PAIRS) as pairs:
        blank, sr = sf.read(BytesIO(playback.read("077/absent-baseline.wav")))
        assert sr == 16000
        originals = {r["voice"]:r for r in json.loads(single.read("manifest.json"))["records"] if "wake_only" in r}
        paired = {(r["batch"], r["track"], r["role"]):r for r in json.loads(pairs.read("manifest.json"))["records"]}
        for r in previous["records"]:
            if r["room"] != "surrey":
                continue
            source = blank.copy()
            role, voice = r["role"], r["voice"]
            if role != "absent":
                info, base = previous["voices"][voice], originals[voice]
                if role == "double":
                    cue, rate = sf.read(BytesIO(double.read(info["file"])), dtype="int16")
                    assert P.digest(cue) == info["pcm_sha256"]
                else:
                    if role == "single":
                        cue, rate = sf.read(BytesIO(single.read(base["wake_only"]["file"])), dtype="int16")
                    else:
                        paired_role = "nyx_plain" if role=="plain" else "willow_matched"
                        pair = paired[base["batch"], base["index"], paired_role]
                        full, rate = sf.read(BytesIO(pairs.read(pair["file"])), dtype="int16")
                        cue = full[32000:pair["command_at"]]
                    cue = np.rint(cue.astype(float)*info["scales"][role]).astype("<i2")
                assert rate == 16000
                end = r["end"] - int(np.abs(ir[0][:800]).argmax())
                source[end-len(cue):end, 0] = cue.astype(float)/131072
            before, _ = air.receive(32768*source, *ir, noise)
            assert P.digest(before) == r["pcm_sha256"]
            source[:, 0] *= 10**.3
            assert np.max(np.abs(source)) < 1
            mic, clipped = air.receive(32768*source, *ir, noise)
            if role == "absent" or (voice,role) == ("af_bella","double"):
                tag = "absent" if role == "absent" else "early-nyx"
                assert P.digest(mic) == loud["240-front-077-"+tag]["pcm_sha256"]
            label = r["label"]+"-plus6"
            path = WORK/(label+".wav")
            sf.write(path, mic, 16000, subtype="PCM_16")
            files[str(path)] = P.sha(path)
            ref = r["reference_path"]
            files[ref] = previous["files"][ref]
            rows.append(dict(r,label=label,mode="plus6",path=str(path),sha256=files[str(path)],pcm_sha256=P.digest(mic),
                             parent_label=r["label"],parent_pcm_sha256=r["pcm_sha256"],clipped=int(clipped.sum())))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 33
    P.save(PLAN, dict(records=rows, files=files, runtime=previous["runtime"], threshold_requests=["DEFAULT"]))


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
