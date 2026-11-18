"""Fit per-frame confuser scores and calibrate on previously unseen training recordings."""
import argparse
from unittest.mock import patch

import numpy as np
import torch

from fit_c161_history_gain_v1 import ADMISSION, BARE, MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
import train_c161_bounded_readout_v1 as bounded
import train_c161_confuser_readout_v1 as first

STEM = "4090-c161-protected-confuser"


def windows():
    inputs = bounded.inputs

    def extended():
        yield from inputs()
        for ledger in (ADMISSION, BARE):
            for r in read(RESULTS / ledger)["admitted"]:
                request = r["request"]
                assert request["role"] == "training-only"
                if request["requested_label"] == 1:
                    yield f"{ledger}-{r['index']}", request["voice"], True, load_wav(r["wav"])

    with patch.object(bounded, "inputs", extended):
        records, hidden = first.training_data()
    it = REF_KERNEL(model_path=str(first.PARENT), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    q = {t["index"]: t["quantization"] for t in it.get_tensor_details()}
    sx, zx = q[62]
    sw, _ = q[14]
    sl, zl = q[63]
    w = torch.as_tensor(it.get_tensor(14).ravel(), device="cuda").float() * sw
    b = int(it.get_tensor(13)[0]) * sx * sw
    features, logits, rows = [], [], []
    for meta, h in zip(records, hidden, strict=True):
        x = (h.float() - zx) * sx
        original = ((x @ w + b) / sl).round().clamp(-128-zl, 127-zl) * sl
        peaks = original.sigmoid().unfold(0, 5, 1).mean(1)
        for _ in range(1 if meta["positive"] else 3):
            start = int(peaks.argmax())
            if not meta["positive"] and peaks[start] < .70:
                break
            features.append(x[start:start+5].clone())
            logits.append(original[start:start+5].clone())
            rows.append(dict(**meta, start=start, calibration=int(meta["target_pcm_sha256"][:8], 16) % 5 == 0))
            peaks[max(0, start-4):start+5] = -1
    return torch.stack(features), torch.stack(logits), rows, sx, sl, zl


def train():
    assert "4090" in torch.cuda.get_device_name() and sha(MODEL) == MODEL_SHA
    assert sha(first.PARENT) == read(RESULTS / "4090-c161-signed-confuser-training-v1.json")["parent_sha256"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, parent, rows, sx, sl, zl = windows()
    pos = torch.tensor([r["positive"] for r in rows], device="cuda")
    cal = torch.tensor([r["calibration"] for r in rows], device="cuda")
    assert set(r["target_pcm_sha256"] for r in rows if r["calibration"]).isdisjoint(
        r["target_pcm_sha256"] for r in rows if not r["calibration"])
    # Each positive frame at or above 0.5 protects its own high-scoring direction.
    px = x[(pos & ~cal)[:, None] & (parent >= 0)]
    nx = x[~pos & ~cal].mean(1)
    a = torch.cat((px, nx)).double()
    y = torch.cat((torch.full((len(px),), -3., device="cuda"), (parent[~pos & ~cal].mean(1)+1).clamp(3, 8.5))).double()
    weights = torch.cat((torch.full((len(px),), .5/len(px), device="cuda", dtype=torch.float64),
                         torch.full((len(nx),), .5/len(nx), device="cuda", dtype=torch.float64)))
    mean = weights @ a
    scale = (weights @ (a-mean).square()).sqrt().clamp_min(.25)
    z = (a-mean)/scale
    system = z.T @ (weights[:, None] * z)
    system.diagonal().add_(.10 * a.shape[1] / len(a))
    w = torch.linalg.solve(system, z.T @ (weights * (y-weights @ y))) / scale
    intercept = weights @ y - mean @ w
    sw = float(np.float32(w.abs().max().item()/127))
    quant = (w/sw).round().clamp(-127,127).to(torch.int8)
    response = x.double() @ (quant.double()*sw)
    bias_unit = float(np.float32(sx * sw))
    bias = round(intercept.item()/bias_unit)
    parent_sum = (parent.sigmoid()*256).round().clamp(0,255).sum(1)
    protected = pos & cal & (parent_sum >= 1175)
    assert protected.any() and (~pos & cal).any()
    def scores(offset):
        penalty = ((response + offset*bias_unit)/sl).round().clamp(0,127-zl)*sl
        return (((parent-penalty).clamp((-128-zl)*sl,(127-zl)*sl)).sigmoid()*256).round().clamp(0,255).sum(1)
    # Select the largest intercept up to the fitted value preserving every already-hit calibration window.
    low = min(bias, int(torch.floor(-response[protected].max()/bias_unit).item())-1)
    high = bias
    while low < high:
        mid = (low+high+1)//2
        if (scores(mid)[protected] >= 1175).all():
            low = mid
        else:
            high = mid-1
    result = scores(low)
    assert (result[protected] >= 1175).all()
    model = RESULTS / f"{STEM}-single-v1.tflite"
    with patch.object(first, "FINAL", model):
        first.export(quant.cpu().numpy(), low, sw)
    for r, before, after in zip(rows, parent_sum.tolist(), result.tolist(), strict=True):
        r.update(parent_sum=int(before), calibrated_sum=int(after))
    ledger = RESULTS / f"{STEM}-windows-v1.json"
    save(ledger, dict(records=rows))
    save(RESULTS / f"{STEM}-training-v1.json", dict(parent_sha256=sha(first.PARENT), baseline_sha256=MODEL_SHA,
         gpu=torch.cuda.get_device_name(), positive_frames=len(px), negative_windows=len(nx), protected_windows=int(protected.sum()),
         added_chatterbox_sources=len({r["key"] for r in rows if r["key"].startswith((ADMISSION, BARE))}),
         weight_scale=sw, initial_bias=bias, final_bias=low, windows_sha256=sha(ledger),
         reports=[dict(name="verifier", path=model.name, bytes=model.stat().st_size, sha256=sha(model))]))


def forward_replay():
    from audit_c161_history_paths_v1 import arbitrate, decode, times
    from replay_c161_vowel_name_pair_v1 import binding
    import verify_c161_ambient_stream_v1 as live

    prior = RESULTS / "4090-c161-signed-forward-v1.json"
    report = read(prior)
    source = RESULTS / "4090-c161-signed-forward-scores-v1.npz"
    assert sha(source) == report["scores_sha256"]
    fit = read(RESULTS / f"{STEM}-training-v1.json")["reports"][0]
    model = RESULTS / fit["path"]
    assert sha(model) == fit["sha256"]
    rows, arrays = [], {}
    native = binding()
    with np.load(source) as baseline, patch.object(live, "MODEL", model):
        for i, r in enumerate(report["records"]):
            codes, clock, emitted, suppressed = live.run_stream(native, load_wav(r))
            clock = np.asarray(clock)
            np.testing.assert_array_equal(clock, baseline[f"{i}_clock"])
            arrays[str(i)] = np.asarray(codes, np.uint8)
            std, aux = [decode(c, clock) for c in arrays[str(i)]]
            assert arbitrate(std, aux) == (emitted, suppressed)
            rows.append(dict(index=i, voice=r["voice"], scene=r["scene"], role=r["target_id"],
                             start=r["start"], end=r["end"], first_phone=r["first_phone"],
                             pcm_sha256=r["pcm_sha256"], primary=times(std), pair=times(emitted)))
    dest = RESULTS / f"{STEM}-forward-scores-v1.npz"
    with dest.open("xb") as f:
        np.savez_compressed(f, **arrays)
    save(RESULTS / f"{STEM}-forward-v1.json", dict(model_sha256=sha(model), source_report_sha256=sha(prior),
         scores_sha256=sha(dest), records=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("actions", nargs="+")
    for action in p.parse_args().actions:
        if action == "train":
            train()
        elif action == "forward":
            forward_replay()
        elif action == "isolated":
            import check_c161_signed_legacy_v1 as legacy
            with patch.object(legacy, "STEM", STEM):
                legacy.main()
        elif action in ("audit", "transfer"):
            import review_c161_confuser_readout_v1 as review
            with patch.object(review, "STEM", STEM):
                {"audit":review.audit, "transfer":review.transfer}[action]()
        else:
            import replay_c161_vowel_name_pair_v1 as replay
            with patch.object(replay, "STEM", STEM):
                replay.run(action)
