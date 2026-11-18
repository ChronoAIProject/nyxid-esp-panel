"""Fit the C161 output layer against competing daily phrases on matched five-step windows."""
import ctypes
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import webrtcvad
from torch.nn import functional as F

from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, REF_KERNEL, RESULTS, read, save, sha
from generate_c161_vowel_name_pair_v1 import AUDIO, SOURCE
from replay_c161_ambient_history_v1 import BUILD, CPP
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_vowel_name_pair_v1 import instrument, trace

STEM = "4090-c161-paired-event-readout"
DAILY = RESULTS / "4090-chatterbox-daily-training-admission-v1.json"
KINDS = ("nyx", "nick", "nicole", "next", "number_quantity")
SEED = 20261119


def both_histories(pcm, native):
    full = np.r_[np.zeros(PRIME, np.int16), pcm]
    hops = np.pad(full, (0, -len(full) % 160)).reshape(-1, 160)
    vad = webrtcvad.Vad(0)
    active = np.fromiter((vad.is_speech(h.tobytes(), 16000) for h in hops), dtype=np.uint8, count=len(hops))
    for arm in (0, 1):
        raw = np.empty((len(hops) - 2, 40), np.uint16)
        assert native(hops.ctypes.data, active.ctypes.data, len(hops), arm, raw.ctypes.data, raw.size) == 0
        values, support, _ = prepare(raw, len(full))
        clock = np.minimum((support + 479) // 480 * 480, len(full)) - PRIME
        it = instrument()
        inp, = it.get_input_details()
        out, = it.get_output_details()
        scale, zero = inp["quantization"]
        quant = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
        hidden, probabilities = [], []
        for batch in quant.reshape(-1, 3, 40):
            it.set_tensor(inp["index"], batch[None])
            it.invoke()
            hidden.append(it.get_tensor(62)[0])
            probabilities.append(int(it.get_tensor(out["index"])[0, 0]))
        yield clock, np.stack(hidden), np.array(probabilities)


def cache():
    assert sha(MODEL) == MODEL_SHA
    sources = read(SOURCE)
    assert sha(AUDIO) == sources["pcm_sha256"]
    build = read(BUILD / "build.json")
    assert sha(CPP) == build["cpp_sha256"] and sha(BUILD / "frontend.so") == build["library_sha256"]
    native = ctypes.CDLL(str((BUILD / "frontend.so").resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    native.restype = ctypes.c_int
    pairs = defaultdict(dict)
    for row in sources["records"]:
        if row["role"] == "fit" and row["kind"] in KINDS:
            key = row["voice"], row["speed"], row["text"].partition(",")[2]
            pairs[key][row["kind"]] = row
    daily = defaultdict(list)
    for row in read(DAILY)["admitted"]:
        assert row["request"]["role"] == "training-only"
        daily[row["request"]["group"]].append(row)
    assert len(pairs) == 36
    selections, vectors = [], {"silence": [], "speech": []}
    with np.load(AUDIO, allow_pickle=False) as archive:
        for i, (key, group) in enumerate(pairs.items()):
            history = daily["conversation"][i % len(daily["conversation"])]
            number = daily["number_quantity"][i % len(daily["number_quantity"])]
            prefix = load_wav(history["wav"])
            gap = (4000, 8000, 16000)[i % 3]
            onset = len(prefix) + gap
            for condition, samples in (("silence", np.zeros_like(prefix)), ("speech", prefix)):
                windows = []
                for kind in KINDS:
                    if kind == "number_quantity":
                        target = load_wav(number["wav"])
                        target_id = "daily-" + str(number["index"])
                    else:
                        r = group[kind]
                        target = archive[r["key"]]
                        target_id = r["key"]
                        assert hashlib.sha256(target.tobytes()).hexdigest() == r["pcm_sha256"]
                    pcm = np.r_[samples, np.zeros(gap, np.int16), target, np.zeros(8000, np.int16)]
                    ends, codes, features = [], [], []
                    for clock, h, probabilities in both_histories(pcm, native):
                        sums = np.convolve(probabilities, np.ones(5, int), mode="valid")
                        if kind == "nyx":
                            boundary = onset + group[kind]["prefix_end"]
                            allowed = (clock[4:] >= boundary) & (clock[4:] <= boundary + 2880)
                        else:
                            allowed = clock[:-4] > onset
                        eligible = np.flatnonzero(allowed)
                        j = eligible[np.argmax(sums[eligible])]
                        ends.append(int(clock[j+4]))
                        codes.append(int(sums[j]))
                        features.append(h[j:j+5])
                    windows.append(features)
                    selections.append(dict(group=i, voice=key[0], speed=key[1], carrier=key[2], kind=kind,
                                           target_id=target_id, history_id=history["index"], condition=condition,
                                           onset=onset, gap=gap, ends=ends, sums=codes,
                                           pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
                vectors[condition].append(windows)
    return {k: np.stack(v) for k, v in vectors.items()}, selections


def calibration():
    plan = read(RESULTS / "4090-causal-source-balanced-int8-plan-v1.json")
    src = read(RESULTS / "4090-causal-sequence-batch-original-v1.json")["batches"][0]
    path = Path(plan["batch"])
    assert str(path) == src["path"] and sha(path) == src["sha256"]
    with np.load(path) as b:
        return np.concatenate([trace(np.pad(b["x"][i, :n], ((0, -n % 3), (0, 0))))[0]
                               for i, n in enumerate(plan["lengths"])])


def train():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    it = instrument()
    detail = {r["index"]: r for r in it.get_tensor_details()}
    sx, zx = detail[62]["quantization"]
    sw, zw = detail[14]["quantization"]
    assert zx == -128 and zw == 0
    orig = it.get_tensor(14).ravel()
    bias = int(it.get_tensor(13)[0]) * sx * sw
    weights = torch.tensor(orig, device="cuda", dtype=torch.float32) * sw
    vectors, selections = cache()
    ledger = RESULTS / f"{STEM}-windows-v1.json"
    save(ledger, dict(model_sha256=MODEL_SHA, source_sha256=sha(SOURCE), daily_sha256=sha(DAILY), records=selections))
    anchors = (torch.tensor(calibration(), device="cuda", dtype=torch.float32) - zx) * sx
    blob = MODEL.read_bytes()
    assert len(blob) == 60952 and blob.count(orig.tobytes()) == 1
    start = blob.index(orig.tobytes())
    reports = []
    for name, vector in vectors.items():
        assert vector.shape == (36, 5, 2, 5, 3480)
        x = (torch.tensor(vector, device="cuda", dtype=torch.float32) - zx) * sx
        rng = np.random.default_rng(SEED)
        w = weights.clone().requires_grad_()
        opt = torch.optim.Adam([w], lr=8e-5)
        losses = []
        for step in range(400):
            q = (w / sw).round().clamp(-127, 127) * sw
            wq = w + (q - w).detach()
            indices = rng.integers(len(x), size=16)
            logits = x[indices] @ wq + bias
            # Each positive must sustain a response across the chosen five frames.
            pos = logits[:, 0].amin(dim=-1)
            # Penalize the strongest competing phrase for each group and frontend.
            negative = logits[:, 1:].mean(dim=-1).amax(dim=1)
            contrast = F.softplus(4 + negative - pos).mean()
            absolute = F.softplus(2.6 - pos).mean() + F.softplus(negative + 1).mean()
            sample = rng.integers(len(anchors), size=256)
            drift = (anchors[sample] @ (wq - weights)).square().mean()
            loss = contrast + absolute + .25 * drift
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0, 99, 199, 399):
                losses.append(dict(step=step+1, loss=loss.item(), contrast=contrast.item(), drift=drift.item()))
        final = (w.detach() / sw).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
        path = RESULTS / f"{STEM}-{name}-v1.tflite"
        with path.open("xb") as f:
            f.write(blob[:start] + final.tobytes() + blob[start + len(orig):])
        runtime = REF_KERNEL(model_path=str(path))
        runtime.allocate_tensors()
        np.testing.assert_array_equal(runtime.get_tensor(14).ravel(), final)
        reports.append(dict(name=name, path=path.name, sha256=sha(path), bytes=path.stat().st_size,
                            weight_changes=int((final != orig).sum()), losses=losses))
    save(RESULTS / f"{STEM}-training-v1.json", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
         baseline_sha256=MODEL_SHA, windows_sha256=sha(ledger), anchor_steps=len(anchors), seed=SEED, steps=400,
         weight_offset=start, reports=reports))


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    else:
        import replay_c161_vowel_name_pair_v1 as replay
        replay.STEM = STEM
        for cohort in sys.argv[1:]:
            replay.run(cohort)
