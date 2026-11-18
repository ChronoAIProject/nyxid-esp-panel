"""Fit the frozen INT8 C161 readout on paired native histories, preserving its ABI."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, sha
from generate_c161_vowel_name_pair_v1 import AUDIO, SOURCE

STEM = "4090-c161-vowel-name"


def instrument():
    it = REF_KERNEL(model_path=str(MODEL), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    op, = [op for op in it._get_ops_details() if op["op_name"] == "FULLY_CONNECTED"]
    assert list(op["inputs"]) == [62, 14, 13] and list(op["outputs"]) == [63]
    return it


def trace(values):
    it = instrument()
    inp, = it.get_input_details()
    scale, zero = inp["quantization"]
    q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
    h, logits = [], []
    for start in range(0, len(q), 3):
        it.set_tensor(inp["index"], q[None, start:start + 3])
        it.invoke()
        h.append(it.get_tensor(62)[0])
        logits.append(int(it.get_tensor(63)[0, 0]))
    return np.stack(h), np.array(logits)


def data():
    assert sha(MODEL) == MODEL_SHA
    src = json.loads(SOURCE.read_bytes())
    assert sha(AUDIO) == src["pcm_sha256"]
    it = instrument()
    detail = {r["index"]: r for r in it.get_tensor_details()}
    orig = it.get_tensor(14).ravel()
    bias = int(it.get_tensor(13)[0])
    sx, zx = detail[62]["quantization"]
    sw, zw = detail[14]["quantization"]
    so, zo = detail[63]["quantization"]
    assert zw == 0 and zx == -128 and len(orig) == 3480
    groups, vectors, masks = [], [], []
    with np.load(AUDIO, allow_pickle=False) as archive:
        for r in src["records"]:
            if r["role"] != "fit":
                continue
            pcm = archive[r["key"]]
            assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
            values, clock = frontend(np.pad(pcm, (0, 8000)))
            h, codes = trace(values)
            native = (h.astype(np.int32) - zx) @ orig.astype(np.int32) + bias
            rebuilt = np.clip(np.rint(native * (sx * sw / so)) + zo, -128, 127)
            assert np.max(np.abs(rebuilt - codes)) <= 1
            take = clock > 0
            pos = np.zeros(np.count_nonzero(take), dtype=bool)
            if r["kind"] in ("nyx", "lowered"):
                end = r["prefix_end"]
                pos = (clock[take] >= end + 480) & (clock[take] <= end + 4320)
                assert np.count_nonzero(pos) >= 5
            groups.append(r)
            vectors.append(h[take])
            masks.append(pos)
    longest = max(map(len, vectors))
    x = np.full((len(groups), longest, len(orig)), zx, dtype=np.int8)
    pos = np.zeros(x.shape[:2], bool)
    valid = np.zeros_like(pos)
    for i, h in enumerate(vectors):
        x[i, :len(h)] = h
        pos[i, :len(h)] = masks[i]
        valid[i, :len(h)] = True

    # Distill logits on the original training calibration sequences to limit forgetting.
    calibration = json.loads((RESULTS / "4090-causal-source-balanced-int8-plan-v1.json").read_bytes())
    path = Path(calibration["batch"])
    original_batch = json.loads((RESULTS / "4090-causal-sequence-batch-original-v1.json").read_bytes())["batches"][0]
    assert str(path) == original_batch["path"] and sha(path) == original_batch["sha256"]
    anchors = []
    with np.load(path) as batch:
        for i, length in enumerate(calibration["lengths"]):
            v = np.pad(batch["x"][i, :length], ((0, -length % 3), (0, 0)))
            anchors.append(trace(v)[0])
    return groups, x, pos, valid, np.concatenate(anchors), orig, bias, sx, sw


def train():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    groups, x, pos, valid, anchors, orig, bias, sx, sw = data()
    as_gpu = lambda v: torch.as_tensor(v, device="cuda")
    x, pos, valid, anchors = map(as_gpu, (x, pos, valid, anchors))
    original = as_gpu(orig.astype(np.float32) * sw)
    pos_ids = np.array([i for i,r in enumerate(groups) if r["kind"] == "nyx"])
    alt_ids = np.array([i for i,r in enumerate(groups) if r["kind"] == "lowered"])
    assert all(all(groups[a][k] == groups[b][k] for k in ("voice","speed","text"))
               for a,b in zip(pos_ids, alt_ids, strict=True))
    neg_ids = np.array([i for i,r in enumerate(groups) if r["kind"] not in ("nyx", "lowered")])
    blob = MODEL.read_bytes()
    before = orig.tobytes()
    assert blob.count(before) == 1
    start = blob.index(before)
    reports = []
    for name in ("canonical", "tolerant"):
        rng = np.random.default_rng(20261118)
        w = original.clone().requires_grad_()
        optimizer = torch.optim.Adam([w], lr=0.00008)
        history = []
        for step in range(400):
            p = rng.integers(len(pos_ids), size=24)
            choose_alt = rng.permutation(24) < 12
            pi = np.where(choose_alt & (name == "tolerant"), alt_ids[p], pos_ids[p])
            ni = rng.choice(neg_ids, 48)
            ai = rng.integers(len(anchors), size=256)
            quant = (w / sw).round().clamp(-127, 127) * sw
            wq = w + (quant - w).detach()
            def forward(ids):
                return ((x[ids].float() + 128) * sx) @ wq + bias * sx * sw
            target = F.softplus(-forward(pi))[pos[pi]].mean()
            neg = forward(ni).masked_fill(~valid[ni], -torch.inf).topk(5, dim=1).values
            conflict = F.softplus(neg).mean()
            drift = (((anchors[ai].float() + 128) * sx) @ (wq - original)).square().mean()
            loss = target + conflict + 0.1 * drift
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in (0, 99, 199, 399):
                history.append(dict(step=step+1, target=target.item(), negative=conflict.item(), drift=drift.item()))
        final = (w.detach() / sw).round().clamp(-127, 127).cpu().numpy().astype(np.int8)
        candidate = blob[:start] + final.tobytes() + blob[start + len(before):]
        path = RESULTS / f"{STEM}-{name}-v1.tflite"
        with path.open("xb") as f:
            f.write(candidate)
        check = REF_KERNEL(model_path=str(path))
        check.allocate_tensors()
        np.testing.assert_array_equal(check.get_tensor(14).ravel(), final)
        assert len(candidate) == len(blob) == 60952
        reports.append(dict(name=name, path=path.name, sha256=sha(path), bytes=len(candidate),
                            changed=int(np.count_nonzero(final != orig)), history=history))
    with (RESULTS / f"{STEM}-training-v1.json").open("x") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, source_sha256=sha(SOURCE),
                       baseline_sha256=MODEL_SHA, start=start, seed=20261118, steps=400,
                       fit_clips=len(groups), anchor_frames=len(anchors), reports=reports), f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    train()
