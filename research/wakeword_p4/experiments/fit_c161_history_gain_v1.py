"""Fit one causal per-band input gain against aligned quiet-reference features.

The detector stays frozen. Use only admitted training waveforms and score the
unchanged and corrected inputs with the same single-thread reference kernel.
"""
import argparse
from collections import defaultdict
from functools import partial
import json
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter, OpResolverType
import numpy as np
from scipy.signal import lfilter
import soundfile as sf

from collect_multispeaker_formal_v1 import predict, prepare
from review_c161_equal_length_history_v1 import MODEL, MODEL_SHA, PRIME, RESULTS, THRESHOLD, features, sha
from uint8_event_policy_v1 import event_steps

STEM = "4090-c161-history-gain"
ADMISSION = "4090-chatterbox-training-admission-v1.json"
BARE = "4090-chatterbox-bare-training-admission-v1.json"
DAILY = "4090-chatterbox-daily-training-admission-v1.json"
PIPER = "4090-binary40-long-history-factorial-plan-v1.json"
REF_KERNEL = partial(Interpreter, num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
FIT = RESULTS / f"{STEM}-fit-v1.json"


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def sources():
    rows = []
    for group, name in [("chatterbox", ADMISSION), ("chatterbox_bare", BARE)]:
        for entry in read(RESULTS / name)["admitted"]:
            req = entry["request"]
            assert req["role"] == "training-only"
            rows.append(dict(source=group, id=entry["index"], text=req["text"], label=req["requested_label"],
                             voice=req["voice"], **entry["wav"]))
    names = set()
    for entry in read(RESULTS / PIPER)["sources"]:
        for source in [entry, entry["context_negative"]]:
            assert source["role"] == "training"
            if source["id"] in names:
                continue
            names.add(source["id"])
            rows.append(dict(source="piper", id=source["id"], text=source["text"], label=source["label"],
                             voice=f"{source['engine']}/{source['speaker_id']}", path=source["native_path"],
                             sha256=source["native_sha256"]))
    pools = defaultdict(list)
    for entry in read(RESULTS / DAILY)["admitted"]:
        req = entry["request"]
        assert req["role"] == "training-only" and req["requested_label"] == 0
        pools[req["group"]].append(entry)
    assert len(rows) == 227
    for i, row in enumerate(rows):
        row["history"] = []
        for j, group in enumerate(["conversation", "ordinary_request", "number_quantity"]):
            pool = pools[group]
            entry = pool[(i + 7 * j) % len(pool)]
            row["history"].append(dict(index=entry["index"], text=entry["request"]["text"], **entry["wav"]))
        yield row


def audio(ref):
    path = Path(ref["path"])
    assert sha(path) == ref["sha256"], path
    data, rate = sf.read(path, dtype="int16")
    assert rate == 16000 and data.ndim == 1
    return data


def frontend(pcm):
    full = np.concatenate((np.zeros(PRIME, np.int16), pcm))
    x, support, _ = prepare(features(full), len(full))
    clock = np.minimum((support + 479) // 480 * 480, len(full)) - PRIME
    return x, clock


def predictors(x):
    # One-second-scale causal memory. Only past frames can affect the gain.
    normalized = x.astype(np.float64) / 26
    past = np.zeros_like(normalized)
    past[1:] = lfilter([0.01], [1.0, -0.99], normalized[:-1], axis=0)
    return np.stack([normalized, normalized * past], axis=-1)


def apply_gain(x, weights):
    correction = (predictors(x) * weights).sum(axis=-1)
    return np.clip(x + 26 * correction, 0, 26).astype(np.float32)


def score(x, clock):
    _, codes, _ = predict(x, str(MODEL), REF_KERNEL)
    return codes, [int(clock[step - 1]) for step in event_steps(codes, THRESHOLD)]


def paired_inputs(row):
    parts = []
    for ref in row["history"]:
        parts.extend((audio(ref), np.zeros(8000, np.int16)))
    speech = np.concatenate(parts)
    target = audio(row)
    start = len(speech)
    end = start + len(target)
    arms = {}
    for name, prefix in [("speech", speech), ("quiet", np.zeros_like(speech))]:
        arms[name], clock = frontend(np.concatenate((prefix, target)))
    frame_end = (np.arange(len(arms["quiet"])) + 3) * 160 - PRIME
    mask = (frame_end - 480 >= start) & (frame_end <= end)
    assert np.any(mask)
    return arms, clock, mask, start, end


def fit():
    import torch

    assert sha(MODEL) == MODEL_SHA and MODEL.stat().st_size == 60952
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    gram = torch.zeros((40, 2, 2), dtype=torch.float64, device="cuda")
    rhs = torch.zeros((40, 2, 1), dtype=torch.float64, device="cuda")
    samples, data, traces = [], [], {}
    for i, row in enumerate(sources()):
        arms, clock, mask, start, end = paired_inputs(row)
        reference = arms["quiet"][mask].astype(np.float64)
        samples.append(dict(**row, start=start, end=end, frames=int(mask.sum()), events={}))
        data.append((arms, clock, mask))
        for name, x in arms.items():
            features_in = torch.from_numpy(predictors(x)[mask].transpose(1, 0, 2).copy()).cuda()
            residual = torch.from_numpy(((reference - x[mask]) / 26).T.copy()).cuda().unsqueeze(-1)
            transposed = features_in.transpose(1, 2)
            gram += (transposed @ features_in) / len(reference)
            rhs += (transposed @ residual) / len(reference)
            codes, events = score(x, clock)
            traces[f"{i}_{name}_c161"] = codes
            samples[-1]["events"][name + "_c161"] = events
    # Each utterance and each of its two views has equal weight.
    regularizer = torch.eye(2, dtype=torch.float64, device="cuda") * (2 * len(samples) * 1e-5)
    weights = torch.linalg.solve(gram + regularizer, rhs).squeeze(-1).cpu().numpy()
    assert weights.shape == (40, 2) and np.isfinite(weights).all()
    for i, (arms, clock, mask) in enumerate(data):
        reference = arms["quiet"][mask]
        for name, x in arms.items():
            corrected = apply_gain(x, weights)
            codes, events = score(corrected, clock)
            traces[f"{i}_{name}_gain"] = codes
            samples[i]["events"][name + "_gain"] = events
            for key, value in [("original", x), ("corrected", corrected)]:
                samples[i][name + "_" + key + "_mse"] = float(np.mean((value[mask] - reference) ** 2, dtype=np.float64))
    archive = RESULTS / f"{STEM}-training-scores-v1.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **traces)
    save(FIT, dict(model_sha256=MODEL_SHA, gpu=torch.cuda.get_device_name(), weights=weights.tolist(),
                   inputs={n:sha(RESULTS/n) for n in [ADMISSION, BARE, DAILY, PIPER]}, score_sha256=sha(archive),
                   code_sha256=sha(Path(__file__)), samples=samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    fit()
