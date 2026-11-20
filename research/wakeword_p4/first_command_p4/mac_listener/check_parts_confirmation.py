"""Calibrate a causal confirmation span on fit positives and replay two frozen heads."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

import score_piper_parts as evaluation

trial, fit, replay = evaluation.trial, evaluation.fit, evaluation.replay
STEM = "4090-c161-parts-confirmation"
CAL = fit.RESULTS / (STEM + "-calibration.json")
PLAN = fit.RESULTS.parent / "experiments/c161_parts_confirmation_protocol.md"


def models():
    old = fit.read(fit.FIT)["models"]["variable"]
    new = fit.read(trial.REPORT)
    return [(fit.RESULTS / old["file"], old["sha256"], fit.read(trial.original.CAL)["cutoffs"]["variable"]["value"]),
            (trial.WEIGHTS, new["weights_sha256"], new["cutoffs"]["expanded"]["value"])]


def calibrate():
    records, features, masks, _, _ = trial.dataset({})
    params = [(evaluation.gpu_weights(p, digest)[0], cutoff) for p, digest, cutoff in models()]
    rows = []
    for r, x, mask in zip(records, features, masks, strict=True):
        if not r["positive"] or r["split"] != "fit":
            continue
        hits = [np.flatnonzero((fit.integer_scores(x.numpy(), w, trial.GAPS) >= cutoff) & mask.numpy())
                for w, cutoff in params]
        distance = int(abs(hits[0][:, None] - hits[1]).min()) if all(map(len, hits)) else None
        rows.append(dict(key=r["key"], pcm_sha256=r["pcm_sha256"], distance=distance))
    distances = [r["distance"] for r in rows if r["distance"] is not None]
    span = int(np.quantile(distances, .9, method="higher"))
    assert len(rows) == 312 and 0 <= span <= 9
    report = dict(records=rows, distribution=dict(sorted(Counter(distances).items())), comparable=len(distances),
                  fitted_frames=span, fixed_frames=5, sources={p.name:fit.sha(p) for p in
                  (PLAN, Path(__file__), trial.REPORT, fit.FIT, trial.original.CAL)})
    with TemporaryDirectory(dir=fit.WORK) as temp:
        path = Path(temp)/CAL.name
        fit.save(path, report)
        path.replace(CAL)


def agreement(a, b, span):
    # A threshold crossing can be credited only after the second detector responds.
    ticks = np.arange(len(a), dtype=np.int32)
    def recent(above):
        last = np.maximum.accumulate(np.where(above, ticks, -span-1))
        return ticks-last <= span
    return (a & recent(b)) | (b & recent(a))


def score(mode):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.cuda.get_device_name()
    assert "4090" in device
    native = fit.frontend.native_frontend()
    params = [(evaluation.gpu_weights(p, d)[1], c) for p, d, c in models()]
    cal = fit.read(CAL)
    spans = {"same": 0, "fitted": cal["fitted_frames"], "150ms": 5}
    old_report = fit.RESULTS/(fit.STEM+"-"+mode+".json")
    new_report = fit.RESULTS/(trial.STEM+"-"+mode+".json")
    sources = {p.name:fit.sha(p) for p in (old_report, new_report, PLAN, CAL, Path(__file__), Path(evaluation.__file__))}
    rows, summary = [], defaultdict(lambda: defaultdict(Counter))
    for old, new, (r, src, c161, pcm) in zip(fit.read(old_report)["records"], fit.read(new_report)["records"],
                                               replay.corpus(mode, sources), strict=True):
        if pcm is None:
            pcm, rate = sf.read(src.get("wav_path") or src["path"], dtype="int16")
            assert rate == 16000
        assert old["pcm_sha256"] == new["pcm_sha256"] == sha256(pcm.tobytes()).hexdigest()
        assert (old["group"], old["key"], old["c161"]) == (r["group"], r["key"], c161)
        bands, clock = replay.native_features(native, pcm)
        above = []
        for (w, cutoff), digest in zip(params, [old["candidates"]["variable"]["score_sha256"], new["score_sha256"]], strict=True):
            values = evaluation.scores(bands, w, trial.GAPS)
            assert sha256(values.tobytes()).hexdigest() == digest
            above.append(values >= cutoff)
        r.update(c161=c161, pcm_sha256=new["pcm_sha256"], events={}, union={})
        for name, span in spans.items():
            yes = agreement(*above, span)
            hits = replay.events(yes, clock, True)
            if name == "same":
                assert hits == new["events"]["expanded_agreement"]
            ts = [e["sample"] for e in replay.choose(c161, hits)]
            r["events"][name], r["union"][name] = hits, ts
            c = summary[r["group"]+"."+r["role"]][name]
            c.update(cases=1, opened=int(bool(ts)), events=len(ts), samples=len(pcm))
            if "command_at" in r:
                first = (ts[0]+255)//256*256 if ts else None
                c.update(timely=int(first is not None and r["start"] <= first < r["command_at"]))
        rows.append(r)
        if len(rows) % (100 if mode == "speech" else 10) == 0:
            print(mode, len(rows), flush=True)
    assert len(rows) == (913 if mode == "speech" else 66)
    out = fit.RESULTS/(STEM+"-"+mode+".json")
    with TemporaryDirectory(dir=fit.WORK) as work:
        p = Path(work)/out.name
        fit.save(p, dict(records=rows, summary=summary, sources=sources, gpu=device, spans=spans,
                 scope="First causal confirmation, original thresholds, 67-call rising-edge cooldown, C161 union."))
        p.replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "speech", "background"])
    mode = parser.parse_args().mode
    if mode == "calibrate":
        calibrate()
    else:
        score(mode)
