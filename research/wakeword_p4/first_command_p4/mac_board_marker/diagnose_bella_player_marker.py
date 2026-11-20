"""Locate the three fixed-marker correlations around measured weak-room arrivals."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

import bella_player_gain_margin as trial
import report_bella_player_gain_margin as margin
import check_timed_final_marker as compiled
from replay_c161_pcm_cues_v1 import detect

P = margin.P
COEFF = trial.ROOT / "portable/c161_coded_cue_coefficients_v5.h"
HEADER = trial.ROOT / "portable/c161_timed_final_marker.h"
OUT = trial.PLAN.with_name("4090-bella-player-marker-timing.json")


def coefficients(name):
    text = COEFF.read_text().split(name, 1)[1].split("=", 1)[1].split(";", 1)[0]
    return np.array(list(map(int, re.findall(r"-?\d+", text))), dtype=np.int64)


def correlations(pcm, anchors, fir, patterns):
    blocks = pcm[:len(pcm)//4*4].reshape(-1, 4).sum(axis=1)
    filtered = np.trunc(np.convolve(np.trunc(blocks / 16), fir)[:len(blocks)] / 128).astype(np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(filtered, (319, 0)), 320)
    ticks = np.array([np.arange(center-100, center+101) for center in anchors])
    signal = torch.tensor(windows[ticks-1], device="cuda", dtype=torch.float64)
    dots = torch.bmm(signal, patterns.unsqueeze(-1)).squeeze(-1)
    energy = signal.square().sum(-1) * patterns.square().sum(-1, keepdim=True)
    values = (dots / energy.clamp_min(1).sqrt()).cpu().numpy()
    return [(t, v) for t, v in zip(ticks, values, strict=True)]


def chain(parts, delays, constrained=True):
    found = []
    for polarity in (-1, 1):
        (ta, a), (tb, b), (tc, c) = parts
        first = ta[a * polarity >= .3]
        middle = tb[b * polarity >= .3]
        if not len(first) or not len(middle):
            continue
        if constrained:
            first_matches = np.abs(middle[:, None] - first - delays[0]) <= 6
            middle = middle[first_matches.any(axis=1)]
        if not len(middle):
            continue
        for tick, corr in zip(tc, c * polarity, strict=True):
            if corr < .2:
                continue
            tol = 6 if corr >= .3 else 2
            if not constrained or np.any(np.abs(tick - middle - delays[1]) <= tol):
                found.append(int(tick) * 4)
                break
    return min(found, default=None)


def main(matcher):
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    fir = coefficients("kCueBandFirV5")
    patterns_np = coefficients("kCueBandPatternsV5").reshape(3, 320)
    np.testing.assert_array_equal(np.square(patterns_np).sum(1), coefficients("kCueBandEnergyV5"))
    patterns = torch.tensor(patterns_np, device="cuda", dtype=torch.float64)
    anchors = {}
    for r in P.read(trial.PLAN)["records"]:
        if (r["role"], r["gain_db"], r["rate"]) == ("nyx", 5, "nominal"):
            mic, _ = sf.read(r["path"], dtype="int16")
            hit, = detect(matcher, [mic])
            anchors[r["scene"]] = hit["t"]
    cases, counts = [], {}
    for plan, path, field in ((trial.rates.PLAN, trial.rates.OUT, "nyx"), (trial.PLAN, margin.OUT, "dual")):
        results = {r["label"]: r for r in P.read(path)["records"]}
        for r in P.read(plan)["records"]:
            if r["role"] != "nyx" or r["scene"] == "quiet" or r["rate"] not in dict(trial.SPEEDS):
                continue
            mic, sr = sf.read(r["path"], dtype="int16")
            assert sr == 16000 and margin.sha(mic.tobytes()) == r["pcm_sha256"]
            speed = Fraction(r["ratio"])
            center = anchors[int(r["scene"])]
            positions = [round(Fraction(center - delay, 4) / speed) for delay in (4480, 1920, 0)]
            parts = correlations(mic, positions, fir, patterns)
            native = next(iter(results[r["label"]][field]["events"]["C"]), None)
            fixed = chain(parts, (640, 480))
            raw_hits = [x["t"] for x in detect(matcher, [mic])]
            assert raw_hits == ([] if fixed is None else [fixed]), (r["label"], raw_hits, fixed)
            assert (native is None and fixed is None) or native == fixed + 480
            corrected = chain(parts, (round(640/speed), round(480/speed)))
            unordered_spacing = chain(parts, (0, 0), constrained=False)
            peaks = []
            for ticks, v in parts:
                i = int(np.argmax(np.abs(v)))
                peaks.append(dict(sample=int(ticks[i]) * 4, correlation=float(v[i]),
                                  min_signed=float(v.min()), max_signed=float(v.max())))
            row = dict(label=r["label"], gain_db=r.get("gain_db", 2), scene=int(r["scene"]), rate=r["rate"],
                       pcm_sha256=r["pcm_sha256"], peaks=peaks, native=native, nominal_spacing=fixed,
                       corrected_spacing=corrected, free_spacing=unordered_spacing, anchor_samples=[p*4 for p in positions])
            group = counts.setdefault(str(row["gain_db"]) + "/" + r["rate"], Counter())
            group.update(cases=1, native=native is not None, fixed=fixed is not None, corrected=corrected is not None,
                         free=unordered_spacing is not None, raw_exact=True)
            cases.append(row)
    assert len(cases) == 60
    inputs = (Path(__file__), COEFF, HEADER, trial.PLAN, trial.rates.PLAN, trial.rates.OUT, margin.OUT,
              trial.ROOT / "experiments/bella_player_marker_timing_protocol.md", Path(compiled.host.__file__), Path(compiled.__file__))
    P.save(OUT, dict(records=cases, counts=counts, device=device, torch_version=str(torch.__version__),
           sources={str(p): margin.sha(p.read_bytes()) for p in inputs}))


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        main(compiled.binding(Path(folder)))
