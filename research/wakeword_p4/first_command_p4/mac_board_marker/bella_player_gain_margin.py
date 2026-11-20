"""Apply the registered five player speeds to the three higher address levels."""
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import bella_external_player_rates as rates

ROOT = rates.ROOT
PLAN = ROOT / "results/4090-bella-player-gain-margin-plan.json"
WORK = Path(".local/wakeword/bella-player-gain-margin")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = ROOT / "experiments/bella_player_gain_margin_protocol.md"
SPEEDS = rates.RATES[:5]


def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import bella_address_gain as gain
    import check_address_gain_wakes as wakes
    import check_bella_daily_overlap as daily

    p = daily.P
    sources = p.read(gain.PLAN)
    assert p.sha(gain.PACK) == sources["archive_sha256"]
    parent = p.read(rates.PLAN)
    prior = p.read(rates.OUT)
    assert p.sha(rates.PLAN) == prior["plan_sha256"]
    assert p.sha(rates.OUT.with_suffix(".zip")) == prior["archive_sha256"]
    controls = {r["label"]: r for r in parent["records"] if r["role"] == "erased"}
    files = {path: sources["files"][path] for path in sources["runtime"].values()}
    dependencies = (SCRIPT, PROTOCOL, gain.PLAN, gain.PACK, rates.PLAN, rates.OUT, rates.OUT.with_suffix(".zip"),
                    wakes.OUT, rates.SCRIPT, Path(wakes.__file__), Path(daily.__file__), daily.OUT,
                    wakes.room.official.SOURCE, Path(wakes.room.official.__file__))
    files.update({str(path): p.sha(path) for path in dependencies})
    history = {r["index"]: r["file"] for r in p.read(daily.OUT)["records"] if r["role"] == "original"}
    rooms, taps = daily.rooms()
    renderers = {case["file"]: render for case, _, render, _, _, _ in rooms[-6]}
    WORK.mkdir(exist_ok=True, parents=True)
    records = []
    with ZipFile(gain.PACK) as archive:
        for src in sources["records"]:
            if src["level"] not in ("plus3dB", "plus4dB", "plus5dB"):
                continue
            raw = archive.read(src["member"])
            assert gain.digest(raw) == src["speaker_sha256"]
            packed, sr = sf.read(BytesIO(raw), dtype="int32")
            assert sr == 16000 and np.all(packed % 16384 == 0)
            dry = packed // 16384
            scene = src["scene"]
            render = renderers[history[scene]]
            ref_path = src["reference_path"]
            ref, sr = sf.read(ref_path, dtype="int16")
            assert sr == 16000 and p.digest(ref) == src["reference_sha256"]
            files[ref_path] = sources["files"][ref_path]
            for tag, speed in SPEEDS:
                played = dry if speed == 1 else np.rint(resample_poly(dry.astype(float), speed.denominator, speed.numerator))
                rear = daily.board.bella.rear_filter(played, taps)
                rear = np.pad(rear, (0, max(0, len(dry) - len(rear))))[:len(dry)]
                mic, clipped = render(rear)
                pcm_sha = p.digest(mic)
                if speed == 1:
                    assert pcm_sha == src["pcm_sha256"]
                end = round(src["end"] / speed)
                control = controls[f"{scene}-erased-{tag}"]
                assert (len(mic), src["reference_sha256"], end) == (control["samples"], control["ref_sha256"], control["end"])
                label = f"{scene:03}-{src['role']}-{src['level']}-{tag}"
                wav = WORK / (label + ".wav")
                sf.write(wav, mic, 16000, subtype="PCM_16")
                files[str(wav)] = p.sha(wav)
                records.append(dict(label=label, source=src["label"], scene=scene, role=src["role"],
                                    gain_db=int(src["gain_db"]), rate=tag, ratio=str(speed), end=end,
                                    start=round(src["start"] / speed), command=src["command"],
                                    clipped=clipped, samples=len(mic), path=str(wav), sha256=files[str(wav)],
                                    pcm_sha256=pcm_sha, reference_path=ref_path, ref_sha256=src["reference_sha256"],
                                    erased_control=control["label"], erased_pcm_sha256=control["pcm_sha256"]))
    assert len(records) == len({r["label"] for r in records}) == 90
    doc = dict(records=records, files=files, runtime=sources["runtime"], threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert p.read(PLAN) == doc
    else:
        p.save(PLAN, doc)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        sys.path.insert(0, str(ROOT / "qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif action == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    elif action == "dual":
        rates.PLAN, rates.WORK = PLAN, WORK
        rates.dual()
    elif action == "prepare":
        prepare()
    else:
        raise ValueError(action)
