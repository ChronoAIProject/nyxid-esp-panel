"""Reduce both playback channels in the Aachen hall after the registered target step."""
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import bella_air_rooms as air

study = air.study
WORK = air.WORK / "master-minus6"
OUT = air.OUT.with_name("4090-bella-air-master.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
PROTO = air.PROTO.with_name("bella_air_master_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import torch
    import package_bella_6db_stereo as stereo

    P = stereo.P
    torch.set_num_threads(1)
    original, native = P.read(air.PLAN), P.read(air.OUT)
    assert P.sha(air.PLAN) == native["plan_sha256"] and P.sha(stereo.PACK) == P.read(stereo.OUT)["archive_sha256"]
    assert "4090" in torch.cuda.get_device_name()
    files = {v: original["files"][v] for v in original["runtime"].values()}
    for p in (SCRIPT, PROTO, air.PLAN, air.OUT, air.SCRIPT, stereo.OUT, stereo.PACK):
        files[str(p)] = P.sha(p)
    responses = []
    for entry in air.RESPONSES["aula"]:
        p = air.response_file(entry)
        files[str(p)] = entry[-1]
        responses.append(sf.read(p)[0][:, 1])
    scale = np.linalg.norm(responses[0])
    rng = np.random.Generator(np.random.PCG64(209473))
    lookup = {r["label"]: r for r in original["records"]}
    rows = []
    WORK.mkdir(parents=True, exist_ok=True)
    with ZipFile(stereo.PACK) as z:
        for i, side in enumerate(("az0", "az180")):
            h, other = responses[i]/scale, responses[1-i]/scale
            for scene in (77, 79, 90):
                first = lookup[f"aula-{side}-{scene:03}-early-nyx-plus6"]
                noise = rng.normal(0, 12, first["samples"])
                for role in ("early-nyx", "headroom", "willow", "absent"):
                    r = lookup[f"aula-{side}-{scene:03}-{role}-plus6"]
                    stereo_pcm, rate = sf.read(BytesIO(z.read(r["source"])))
                    assert rate == 16000
                    wave = 32768*stereo_pcm
                    before, _ = air.receive(wave, h, other, noise)
                    assert P.digest(before) == r["pcm_sha256"]
                    mic, clipped = air.receive(wave*10**(-6/20), h, other, noise)
                    label = r["label"]+"-master-minus6"
                    path = WORK/(label+".wav")
                    sf.write(path, mic, 16000, subtype="PCM_16")
                    files[str(path)] = P.sha(path)
                    ref = r["reference_path"]
                    files[ref] = original["files"][ref]
                    rows.append(dict(r, label=label, mode="plus6-master-minus6", path=str(path), sha256=files[str(path)],
                                     pcm_sha256=P.digest(mic), parent_pcm_sha256=r["pcm_sha256"], parent_label=r["label"],
                                     clipped=int(clipped.sum()), clipped_before_command=int(clipped[:r["end"]].sum())))
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 24
    P.save(PLAN, dict(records=rows, files=files, runtime=original["runtime"], threshold_requests=["DEFAULT"]))


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
