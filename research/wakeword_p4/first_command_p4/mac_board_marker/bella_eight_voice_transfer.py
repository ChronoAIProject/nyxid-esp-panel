"""Add the delivered early marker to eight existing voice carriers without fitting to rooms."""
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import bella_air_rooms as air

study = air.study
WORK = Path(".local/wakeword/bella-eight-voice-transfer")
OUT = study.OUT.with_name("4090-bella-eight-voice-transfer.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
PACK = OUT.with_name("hye-nyx-eight-voice-transferred-cues.zip")
PROTO = study.PROTOCOL.with_name("bella_eight_voice_transfer_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
GAINS = OUT.with_name("c161-eight-voice-marker-gains.zip")
PAIRS = OUT.with_name("c161-eight-voice-willow-comparison.zip")


def build_cues():
    import numpy as np
    import soundfile as sf
    from scipy.io import wavfile
    import package_bella_6db_stereo as stereo

    fit, P = stereo.sources.fit, stereo.P
    single, sr = sf.read(fit.early.WAV.with_name("hye-nyx-af-bella-plus5db.wav"), dtype="int16")
    double, rr = sf.read(fit.early.WAV, dtype="int16")
    assert sr == rr == 16000 and P.sha(fit.early.WAV) == P.read(fit.early.FIT)["cue_sha256"]
    delta = double.astype(float) - single
    prefix_meta, cues = {}, {}
    with ZipFile(GAINS) as gains, ZipFile(PAIRS) as paired, ZipFile(WORK/PACK.name, "w", ZIP_DEFLATED) as dest:
        rows = [r for r in json.loads(gains.read("manifest.json"))["records"] if "wake_only" in r]
        paired_rows = json.loads(paired.read("manifest.json"))["records"]
        bella, = [r for r in rows if r["voice"] == "af_bella"]
        for r in rows:
            voice = r["voice"]
            key = r["wake_only"]["file"]
            raw = gains.read(key)
            assert study.sha256(raw).hexdigest() == r["wake_only"]["sha256"]
            pcm, rate = sf.read(BytesIO(raw), dtype="int16")
            assert rate == 16000
            inputs = {"single": pcm}
            for role, source_role in (("plain", "nyx_plain"), ("willow", "willow_matched")):
                ref, = [v for v in paired_rows if (v["batch"], v["track"], v["role"]) == (r["batch"], r["index"], source_role)]
                blob = paired.read(ref["file"])
                assert study.sha256(blob).hexdigest() == ref["wav_sha256"]
                wave, rate = sf.read(BytesIO(blob), dtype="int16")
                assert rate == 16000 and not wave[:32000].any()
                inputs[role] = wave[32000:ref["command_at"]]
            scales, waves = {}, {}
            for role, x in inputs.items():
                gain = min(10**.25, 32767/float(x.max()), -32768/float(x.min()))
                scales[role] = gain
                waves[role] = np.rint(x.astype(float)*gain).astype("<i2")
            carrier = waves["single"].astype(float)
            assert len(carrier) <= len(delta) and not delta[:len(delta)-len(carrier)].any()
            added = delta[-len(carrier):] * (r["prefix_scale"]/bella["prefix_scale"]) * (scales["single"]/10**.25)
            pos, neg = added > 0, added < 0
            factor = min(1., float(((32767-carrier[pos])/added[pos]).min()), float(((-32768-carrier[neg])/added[neg]).min()))
            total = np.rint(carrier + factor*added)
            assert total.min() >= -32768 and total.max() <= 32767
            waves["double"] = total.astype("<i2")
            assert np.array_equal(waves["single"][added == 0], waves["double"][added == 0])
            if voice == "af_bella":
                assert np.array_equal(waves["single"], single) and np.array_equal(waves["double"], double)
            buf = BytesIO()
            wavfile.write(buf, 16000, waves["double"])
            name = f"{voice}.wav"
            stereo.write_member(dest, name, buf.getvalue())
            prefix_meta[voice] = dict(file=name, bytes=len(buf.getvalue()), sha256=study.sha256(buf.getvalue()).hexdigest(),
                                      pcm_sha256=P.digest(waves["double"]), scales=scales, added_scale=factor,
                                      changed_samples=int(np.count_nonzero(waves["double"] != waves["single"])))
            cues[voice] = waves
        assert len(cues) == 8
        stereo.write_member(dest, "manifest.json", (json.dumps(dict(sample_rate=16000, voices=prefix_meta,
                            sources={p.name:P.sha(p) for p in (GAINS, PAIRS, fit.early.WAV)}), indent=2)+"\n").encode())
    (WORK/PACK.name).replace(PACK)
    return cues, prefix_meta


def prepare():
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import package_bella_6db_stereo as stereo

    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    WORK.mkdir(exist_ok=True, parents=True)
    P, controls = stereo.P, stereo.sources
    cues, meta = build_cues()
    previous = P.read(air.PLAN)
    files = {v: previous["files"][v] for v in previous["runtime"].values()}
    paths = (SCRIPT, PROTO, GAINS, PAIRS, PACK, air.PLAN, controls.angles.PLAN, stereo.OUT, stereo.PACK, air.SCRIPT,
             controls.fit.early.WAV, controls.fit.early.WAV.with_name("hye-nyx-af-bella-plus5db.wav"))
    files.update({str(p):P.sha(p) for p in paths})
    with ZipFile(stereo.PACK) as z:
        blob = z.read("077/absent-baseline.wav")
        assert study.sha256(blob).hexdigest() == P.read(stereo.OUT)["files"]["077/absent-baseline.wav"]
        blank, sr = sf.read(BytesIO(blob))
    assert sr == 16000
    with ZipFile(stereo.SOURCE.PACK) as z:
        src = json.loads(z.read("manifest.json"))["files"]["077/absent.wav"]
    end = src["command_at"]
    assert not np.any(blank[:end, 0])
    surrey = controls.angles.WORK/"angle_240.0.mat"
    assert P.sha(surrey) == controls.angles.ANGLES[240]
    mat = loadmat(surrey)["brir"]
    scale = P.read(controls.fit.study.PLAN)["paths"][0]["common_scale"]
    surrey_paths = [resample_poly(mat[:, 1, ch], 1, 3)/scale for ch in (2, 8)]
    _, _, _, _, surrey_noise = next((r,x,t,s,n) for r,x,t,s,n in controls.stems_for({240}, ("early-nyx",))
                                    if r["direction"] == "front" and r["scene"] == 77)
    hall_ir = [sf.read(air.response_file(e))[0][:, 1] for e in air.RESPONSES["aula"]]
    hall_paths = [v/np.linalg.norm(hall_ir[0]) for v in hall_ir]
    hall_noise = np.random.Generator(np.random.PCG64(209473)).normal(0, 12, len(blank)+20000)
    cases = [(voice, role, wave) for voice, variants in cues.items() for role, wave in variants.items()]
    cases.append(("shared", "absent", None))
    rows = []
    for room, ir, noise, known, ref_label in (
            ("surrey", surrey_paths, surrey_noise, P.read(controls.angles.PLAN), "240-front-077-"),
            ("aula", hall_paths, hall_noise, previous, "aula-az0-077-")):
        older = {r["label"]:r for r in known["records"]}
        suffix = "-baseline" if room == "aula" else ""
        reference = older[ref_label+"absent"+suffix]
        ref_path = reference["reference_path"]
        files[ref_path] = known["files"][ref_path]
        delay = int(abs(ir[0][:800]).argmax())
        for voice, role, cue in cases:
            player = blank.copy()
            start = None
            if cue is not None:
                player[end-len(cue):end, 0] = cue.astype(float)/131072
                start = int(np.flatnonzero(player[:end, 0])[0])+delay
            mic, clipped = air.receive(32768*player, *ir, noise)
            if role == "absent" or (voice, role) == ("af_bella", "double"):
                tag = "absent" if role == "absent" else "early-nyx"
                assert P.digest(mic) == older[ref_label+tag+suffix]["pcm_sha256"]
            label = f"{room}-{voice}-{role}"
            path = WORK/(label+".wav")
            sf.write(path, mic, 16000, subtype="PCM_16")
            files[str(path)] = P.sha(path)
            rows.append(dict(label=label, room=room, voice=voice, mode=room, role=role, command=src["command"],
                             start=start,end=end+delay,samples=len(mic),clipped=int(clipped.sum()),
                             path=str(path),sha256=files[str(path)],pcm_sha256=P.digest(mic),reference_path=ref_path,
                             ref_sha256=reference["ref_sha256"]))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 66
    P.save(PLAN, dict(records=rows, voices=meta, files=files, runtime=previous["runtime"], threshold_requests=["DEFAULT"]))


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
