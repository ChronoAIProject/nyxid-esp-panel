"""Compare broader synthetic training and simultaneous detectors on full development streams."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

import train_piper_parts as trial

fit = trial.fit
original = trial.original
replay = original.replay


def gpu_weights(path, digest):
    assert fit.sha(path) == digest
    with np.load(path, allow_pickle=False) as archive:
        weights = dict(archive)
    for layer in ("1", "2"):
        bounds = 255 * np.abs(weights["w" + layer].astype(np.int32)).sum(axis=1)
        assert (bounds + np.abs(weights["b" + layer]) < 2**24 - 32).all()
    return weights, {key: torch.as_tensor(value, dtype=torch.float32, device="cuda") for key, value in weights.items()}


def scores(bands, weights, gaps):
    # Dot products on small integer operands fit exactly in full precision FP32.
    x = torch.as_tensor(bands, dtype=torch.float32, device="cuda")
    patches = F.pad(x, (0, 0, 4, 0)).unfold(0, 5, 1).transpose(1, 2).reshape(len(x), 100)
    h = F.linear(patches, weights["w1"], weights["b1"])
    h = ((h + 32) / 64).floor().clamp(0, 255)
    parts = F.linear(h, weights["w2"], weights["b2"]).int().cpu().numpy()
    total = parts[:, 0]
    for part, (low, high) in zip(parts[:, 1:].T, gaps, strict=True):
        past = np.pad(total, (high, 0), constant_values=-100000000)[:-low]
        total = part + np.lib.stride_tricks.sliding_window_view(past, high-low+1).max(axis=1)
    return np.minimum(total, np.r_[-100000000, total[:-1]]).astype(np.int32)


def main(mode):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.cuda.get_device_name()
    assert "4090" in device
    native = fit.frontend.native_frontend()
    report = fit.read(trial.REPORT)
    previous = fit.read(fit.FIT)["models"]["variable"]
    _, old = gpu_weights(fit.RESULTS/previous["file"], previous["sha256"])
    quantized, new = gpu_weights(trial.WEIGHTS, report["weights_sha256"])
    old_cutoff = fit.read(original.CAL)["cutoffs"]["variable"]["value"]
    history = fit.RESULTS/(fit.STEM+"-"+mode+".json")
    records = fit.read(history)["records"]
    training = {r["pcm_sha256"] for r in report["records"]}
    rows, summaries = [], defaultdict(lambda: defaultdict(Counter))
    sources = {str(p): fit.sha(p) for p in (history, trial.REPORT, trial.WEIGHTS, fit.FIT, original.CAL,
                                            Path(__file__), Path(replay.__file__))}
    for prior, (r, src, c161, pcm) in zip(records, replay.corpus(mode, sources), strict=True):
        if pcm is None:
            pcm, rate = sf.read(src.get("wav_path") or src["path"], dtype="int16")
            assert rate == 16000
        digest = sha256(pcm.tobytes()).hexdigest()
        assert digest == src["pcm_sha256"] == prior["pcm_sha256"] and digest not in training
        assert (r["key"], r["group"], c161) == (prior["key"], prior["group"], prior["c161"])
        bands, clock = replay.native_features(native, pcm)
        reference = scores(bands, old, trial.GAPS)
        assert sha256(reference.tobytes()).hexdigest() == prior["candidates"]["variable"]["score_sha256"]
        current = scores(bands, new, trial.GAPS)
        if mode == "speech" or not rows:
            assert np.array_equal(current, fit.integer_scores(bands, quantized, trial.GAPS))
        reference -= old_cutoff
        r.update(pcm_sha256=digest, samples=len(pcm), frames=len(clock), c161=c161,
                 score_sha256=sha256(current.tobytes()).hexdigest(), events={}, union={})
        traces = {"previous": reference}
        for policy, cutoff in report["cutoffs"].items():
            margin = current - cutoff["value"]
            traces[policy] = margin
            traces[policy + "_agreement"] = np.minimum(reference, margin)
        for policy, values in traces.items():
            events = replay.events(values, clock, 0)
            combined = [event["sample"] for event in replay.choose(c161, events)]
            r["events"][policy] = events
            r["union"][policy] = combined
            summary = summaries[r["group"]+"."+r["role"]][policy]
            summary.update(cases=1, samples=len(pcm), opened=bool(combined), events=len(combined))
            if "command_at" in r:
                opened = (combined[0]+255)//256*256 if combined else None
                summary.update(timely=opened is not None and r["start"] <= opened < r["command_at"])
        rows.append(r)
        if len(rows) % (100 if mode == "speech" else 10) == 0:
            print(mode, len(rows), flush=True)
    assert len(rows) == (913 if mode == "speech" else 66)
    out = fit.RESULTS / (trial.STEM + "-" + mode + ".json")
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        path = Path(tmp)/out.name
        fit.save(path, dict(records=rows, summary=summaries, sources=sources, cutoffs=report["cutoffs"],
                 gpu=device, c161=fit.MODEL_SHA, scope="Consumed development PCM. Full streams, one continuous prime. "
                 "Frozen reference score digest matched on every file. Summaries include C161."))
        path.replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["speech", "background"])
    main(parser.parse_args().mode)
