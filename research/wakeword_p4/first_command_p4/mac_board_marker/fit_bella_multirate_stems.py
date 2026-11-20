"""Fit isolated marker stems against three player rates and evaluate a frozen playback."""
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import bella_player_gain_margin as gain_grid
import bella_external_player_rates as native
import check_bella_marker_only as stems
import replay_bella_gain_candidate as playback

P = playback.P
WORK = Path(".local/wakeword/bella-multirate-stems")
BASE = gain_grid.ROOT / "results/4090-bella-multirate-stems"
FIT = BASE.with_suffix(".json")
WAV = BASE.with_name("hye-nyx-af-bella-multirate-experiment.wav")
PLAN = BASE.with_name(BASE.name + "-plan.json")
REPORT = BASE.with_name(BASE.name + "-wake.json")
PROTOCOL = gain_grid.ROOT / "experiments/bella_multirate_stems_protocol.md"


def fit():
    import torch
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    meta, v5, _, codes = stems.gains.fit.grid.fit.original()
    with ZipFile(stems.gains.PACK) as pack:
        spec = json.loads(pack.read("manifest.json"))
        entry, = [r for r in spec["records"] if r.get("wake_only", {}).get("sha256") == P.sha(stems.overlap.board.bella.CUE)]
    original, rate = sf.read(playback.OUT.with_name(P.read(playback.OUT)["cue"]), dtype="int16")
    assert rate == 16000 and P.digest(original) == P.read(playback.OUT)["pcm_sha256"]
    starts = [len(original) - len(v5) + start for start in meta["code_positions"]]
    center = starts[1] + len(codes[1]) / 2
    out = original.astype(float)
    rows = []
    for start, code, gain in zip(starts, codes, spec["gains"], strict=True):
        code = code * entry["prefix_scale"] * gain * 10 ** .25
        index = np.arange(start-64, start+len(code)+64)
        original_index = np.arange(start, start + len(code))
        versions = np.stack([np.interp(center + (index-center)/speed, original_index, code, left=0, right=0)
                             for speed in (.99, 1., 1.01)])
        spectra = torch.tensor(versions, dtype=torch.float64, device="cuda")
        unit = torch.nn.functional.normalize(spectra, dim=1)
        params = torch.zeros(3, dtype=torch.float64, device="cuda", requires_grad=True)
        optim = torch.optim.Adam([params], lr=.04)
        def blend():
            weights = params.softmax(0)
            mixed = torch.nn.functional.normalize(weights @ unit, dim=0)
            return weights, mixed, unit @ mixed
        for _ in range(160):
            _, _, cosines = blend()
            loss = .04 * torch.logsumexp(-cosines/.04, dim=0)
            optim.zero_grad()
            loss.backward()
            optim.step()
        weights, wave, cosines = blend()
        shaped = wave.detach().cpu().numpy() * np.linalg.norm(code)
        np.testing.assert_allclose(np.linalg.norm(shaped), np.linalg.norm(code), rtol=1e-10)
        out[index] += shaped - versions[1]
        rows.append(dict(start=start, window=[int(index[0]), int(index[-1]+1)], weights=weights.tolist(),
                         before=(unit @ unit[1]).tolist(), after=cosines.tolist(), energy=float(code @ code)))
    rounded = np.rint(out)
    assert np.min(rounded) >= -32768 and np.max(rounded) <= 32767
    cue = rounded.astype("<i2")
    raw = playback.playback.sources.wav(cue, "PCM_16")
    if WAV.exists():
        assert raw == WAV.read_bytes()
    else:
        WAV.write_bytes(raw)
    uses = (Path(__file__), PROTOCOL, playback.OUT, stems.gains.PACK, Path(stems.__file__), Path(stems.gains.fit.grid.fit.__file__))
    doc = dict(cue=WAV.name, cue_sha256=P.sha(WAV), pcm_sha256=P.digest(cue), device=torch.cuda.get_device_name(),
               torch_version=str(torch.__version__), samples=len(cue), rate=rate, seed_wav_sha256=P.read(playback.OUT)["cue_sha256"],
               peak=int(np.abs(cue).max()), changed_samples=int(np.count_nonzero(cue != original)), stems=rows,
               sources={str(p): P.sha(p) for p in uses})
    P.save(FIT, doc)


def prepare():
    gain = playback.playback.gain
    daily = stems.overlap
    candidate, rate = sf.read(WAV, dtype="int16")
    assert rate == 16000 and P.sha(WAV) == P.read(FIT)["cue_sha256"]
    registered = P.read(gain_grid.PLAN)
    known = {r["label"]: r for r in registered["records"]}
    source = P.read(gain.PLAN)
    assert P.sha(gain.PACK) == source["archive_sha256"]
    rooms, taps = daily.rooms()
    history = {r["index"]: r["file"] for r in P.read(daily.OUT)["records"] if r["role"]=="original"}
    render_by_file = {case["file"]: render for case, _, render, _, _, _ in rooms[-6]}
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    with ZipFile(gain.PACK) as z:
        for r in source["records"]:
            if r["role"] != "nyx" or r["level"] != "plus5dB":
                continue
            scene = r["scene"]
            raw = z.read(r["member"])
            assert playback.wakes.digest(raw) == r["speaker_sha256"]
            original = sf.read(BytesIO(raw), dtype="int32")[0] // 16384
            absent = sf.read(BytesIO(z.read(f"{scene:03}/absent/erased.wav")), dtype="int32")[0] // 16384
            remixed = absent.copy()
            end = r["end"]
            remixed[end-len(candidate):end] += candidate.astype(np.int32)
            for tag, speed in gain_grid.SPEEDS:
                label = f"{scene:03}-nyx-plus5dB-{tag}"
                baseline = known[label]
                for variant, dry in (("reference", original), ("candidate", remixed)):
                    played = dry if speed == 1 else np.rint(resample_poly(dry.astype(float), speed.denominator, speed.numerator))
                    rear = daily.board.bella.rear_filter(played, taps)
                    rear = np.pad(rear, (0, max(0, len(dry)-len(rear))))[:len(dry)]
                    mic, clipped = render_by_file[history[scene]](rear)
                    if variant == "reference":
                        assert P.digest(mic) == baseline["pcm_sha256"]
                        continue
                    short = f"{scene:03}-multirate-{tag}"
                    wav = WORK / (short + ".wav")
                    sf.write(wav, mic, 16000, subtype="PCM_16")
                    rows.append(dict(label=short, baseline=label, scene=scene, rate=tag, path=str(wav),
                                     sha256=P.sha(wav), pcm_sha256=P.digest(mic), start=baseline["start"], end=baseline["end"],
                                     clipped=clipped, ref_sha256=baseline["ref_sha256"], reference_path=baseline["reference_path"],
                                     erased_control=baseline["erased_control"], erased_pcm_sha256=baseline["erased_pcm_sha256"]))
    assert len(rows) == len({r["label"] for r in rows}) == 15
    P.save(PLAN, dict(records=rows, cue_sha256=P.sha(WAV), baseline_plan_sha256=P.sha(gain_grid.PLAN)))


def report():
    inputs = P.read(PLAN)
    base = P.read(playback.selection.OUT)
    prev = {r["label"]: r for r in base["records"]}
    trial = P.read(WORK / "dual.json")
    assert trial["plan_sha256"] == P.sha(PLAN)
    assert trial["model_sha256"] == base["model_sha256"] and trial["guest_sha256"] == base["guest_sha256"]
    assert trial["qemu_sha256"] == base["qemu_sha256"] and P.sha(WORK / "dual.zip") == trial["archive_sha256"]
    rows, counts, files = [], {}, {}
    with ZipFile(WORK / "dual.zip") as logs:
        for r, native_row in zip(inputs["records"], trial["records"], strict=True):
            assert r["label"] == native_row["label"]
            before = prev[r["baseline"]]["reads"]["nyx"]
            after = playback.selection.first_read(native_row["events"]["O"], r)
            rows.append(dict(**r, baseline_read=before, candidate_read=after, native=native_row))
            count = counts.setdefault(r["rate"], [0, 0])
            count[0] += int(before["timely"])
            count[1] += int(after["timely"])
            for k, v in native_row["members"].items():
                assert playback.wakes.digest(logs.read(k)) == v
                files[k] = v
    REPORT.with_suffix(".zip").write_bytes((WORK / "dual.zip").read_bytes())
    P.save(REPORT, dict(records=rows, counts=counts, files=files, plan_sha256=P.sha(PLAN), cue_sha256=P.sha(WAV),
                        archive_sha256=trial["archive_sha256"], model_sha256=trial["model_sha256"], guest_sha256=trial["guest_sha256"],
                        qemu_sha256=trial["qemu_sha256"], baseline_report_sha256=P.sha(playback.selection.OUT)))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "dual":
        native.WORK, native.PLAN = WORK, PLAN
        native.dual()
    else:
        {"fit": fit, "prepare": prepare, "report": report}[stage]()
