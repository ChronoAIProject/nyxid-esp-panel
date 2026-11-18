"""Cross-Piper-voice-model test of whole-phrase Nicks/Nick/Next separability."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "training"))
from features import features

CLASSES = ("Hey Nicks.", "Hey Nick.", "Hey Next.")
STEM = "4090-c161-phrase-probe"


def sources():
    ref = json.loads((RESULTS / "4090-multispeaker-coverage-native-original-v1.json").read_bytes())
    ledger = Path(ref["records_path"])
    data = ledger.read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["records_sha256"]
    for line in data.splitlines():
        r = json.loads(line)
        if r["text"] in CLASSES:
            assert r["role"] == "training"
            yield (r["voice"], r["speaker_id"], r["id"], r["text"], r["sha256"], ledger.parent / r["path"])
    joe = json.loads((RESULTS / "4090-c161-history-gain-joe-v1.json").read_bytes())
    for record in joe["records"]:
        s = record["source"]
        if s["variant"] == 0 and s["text"] in (*CLASSES, "Hey Nicks!"):
            path = Path(".local/wakeword/joe-development-native-transport-v1") / s["source_path"]
            yield ("joe", 0, s["source_id"], s["text"], s["source_sha256"], path)


def fit(x, y, rows):
    counts = Counter((r["speaker"], r["label"]) for r in rows)
    weight = x.new_tensor([1 / counts[r["speaker"], r["label"]] for r in rows])
    weight = (weight / weight.sum())[:, None]
    mean = (weight * x).sum(0)
    scale = (weight * (x - mean).square()).sum(0).sqrt().clamp_min(0.01)
    normalized = (x - mean) / scale
    prior = (weight * y).sum(0)
    gram = normalized.T @ (weight * normalized)
    gram.diagonal().add_(0.01)
    coef = torch.linalg.solve(gram, normalized.T @ (weight * (y - prior)))
    return mean, scale, prior, coef


def main():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    rows, vectors = [], []
    for group, speaker, source_id, text, sha, path in sources():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        pcm, rate = sf.read(path, dtype="int16")
        assert rate == 16000 and pcm.ndim == 1
        values = np.clip(features(pcm).astype(np.float32) / 25.6, 0, 26) / 26
        pooled = F.adaptive_avg_pool1d(torch.from_numpy(values.T)[None], 24)
        vectors.append(pooled.flatten())
        label = 0 if text == "Hey Nicks!" else CLASSES.index(text)
        rows.append(dict(group=group, speaker=speaker, id=source_id, sha256=sha, text=text, label=label))
    assert len(rows) == 1161 + 24 and len({r["id"] for r in rows}) == len(rows)
    x = torch.stack(vectors).to(device="cuda", dtype=torch.float64)
    labels = torch.tensor([r["label"] for r in rows], device="cuda")
    y = F.one_hot(labels, len(CLASSES)).to(x.dtype)
    fits, arrays = [], {}
    for group in ("en_GB-aru-medium", "en_US-arctic-medium"):
        indices = [i for i, row in enumerate(rows) if row["group"] == group]
        params = fit(x[indices], y[indices], [rows[i] for i in indices])
        mean, scale, prior, coef = params
        decision = (x - mean) / scale @ coef + prior
        assert torch.isfinite(decision).all()
        key = group.split("-")[1]
        for name, value in zip(("mean", "scale", "prior", "coef"), params, strict=True):
            arrays[f"{key}_{name}"] = value.cpu().numpy()
        scores = decision.cpu().numpy()
        arrays[f"{key}_scores"] = scores
        predictions = scores.argmax(1)
        matrices = {}
        for target in sorted({r["group"] for r in rows}):
            matrix = np.zeros((3, 3), dtype=int)
            for i, row in enumerate(rows):
                if row["group"] == target:
                    matrix[row["label"], predictions[i]] += 1
            matrices[target] = matrix.tolist()
        fits.append(dict(training_group=group, training_count=len(indices), confusion=matrices))
    dest = RESULTS / f"{STEM}-weights-and-scores-v1.npz"
    with dest.open("xb") as f:
        np.savez_compressed(f, **arrays)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, classes=CLASSES, fits=fits,
                  archive_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), rows=rows)
    with (RESULTS / f"{STEM}-v1.json").open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
