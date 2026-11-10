"""Train a desktop diagnostic head on frozen, waveform-derived streaming embeddings."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2
import torch
from torch import nn


def head(architecture="original"):
    if architecture == "upstream-dnn":
        # Pinned upstream DNN order, expressed with logits for stable BCE.
        return nn.Sequential(nn.Flatten(), nn.Linear(16*96, 64), nn.LayerNorm(64),
                             nn.ReLU(), nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
    return nn.Sequential(nn.Flatten(), nn.LayerNorm(16*96), nn.Linear(16*96, 64),
                         nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def learning_rate(step, steps, policy):
    if policy == "constant":
        return 1e-3
    # Pinned upstream auto_train first-stage target and phase lengths.
    index, warmup, hold = step-1, steps//5, steps//3
    if index < warmup:
        return 1e-4*index/warmup
    if index <= warmup+hold:
        return 1e-4
    return float(.5e-4*(1+np.cos(np.pi*(index-warmup-hold)/(steps-warmup-hold))))


def event_indices(values, threshold):
    armed, last, result = True, -1000, []
    for n, value in enumerate(values):
        if value < threshold*.5 and n-last >= 25:
            armed = True
        if armed and value >= threshold and n-last >= 25:
            result.append(n); last = n; armed = False
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--negative-candidates", type=int, default=1, help="Training-only max-score mining per confusing-negative row")
    p.add_argument("--lr-policy", choices=["constant", "upstream"], default="constant")
    p.add_argument("--head", choices=["original", "upstream-dnn"], default="original")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--eval-interval", type=int, default=10000)
    a = p.parse_args()
    if a.steps < 10 or a.eval_interval < 1:
        p.error("Require at least 10 steps and a positive evaluation interval")
    if a.negative_candidates < 1:
        p.error("negative candidates must be positive")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.output.exists():
        p.error("Use a fresh output directory")
    cache = json.loads((a.cache/"cache.json").read_text())
    if not cache["complete_development_cache"]:
        raise ValueError("Smoke features cannot be used as a complete training experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("RTX CUDA training required")
    torch.set_num_threads(4)
    torch.manual_seed(726)
    torch.cuda.manual_seed_all(726)
    rng = np.random.default_rng(726)
    rows = json.loads((a.cache/"index.json").read_text())
    if hashlib.sha256((a.cache/"index.json").read_bytes()).hexdigest() != cache["index_sha256"]:
        raise ValueError("Changed embedding index")
    digest = hashlib.sha256()
    with (a.cache/"embeddings.bin").open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != cache["embeddings_sha256"]:
        raise ValueError("Changed embedding binary")
    raw = np.memmap(a.cache/"embeddings.bin", mode="r", dtype="<f4").reshape(-1, 96)
    if len(raw) != cache["frames"]:
        raise ValueError("Truncated embedding data")
    groups, ambient = {0: defaultdict(list), 1: defaultdict(list)}, []
    for row in rows:
        if row["split"] != "training":
            continue
        if row["kind"] == "generated":
            groups[row["label"]][row["engine"]].append(row)
        else:
            ambient.append(row)
    def sequence(row):
        return raw[row["offset"]:row["offset"]+row["frames"]]
    def sample(row, positive=False):
        x = sequence(row)
        if positive:
            # Cache frame 15 is the first real 80 ms embedding, emitted at 80 ms.
            end = 14+int(np.ceil(row["keyword_end_seconds"]/.08))+int(rng.integers(1, 4))
            end = min(len(x)-1, max(15, end))
        else:
            end = int(rng.integers(min(40, len(x)-1), len(x)))
        return x[end-15:end+1]
    def batch():
        examples = []
        mined = []
        for label, count in ((1, 32), (0, 48)):
            engines = list(groups[label])
            for _ in range(count):
                source = groups[label][str(rng.choice(engines))]
                row = source[int(rng.integers(len(source)))]
                if label == 0 and a.negative_candidates > 1:
                    mined.append([sample(row) for _ in range(a.negative_candidates)])
                else:
                    examples.append(sample(row, bool(label)))
        if mined:
            # Select only from training audio; evaluation mode avoids dropout noise.
            model.eval()
            candidates = torch.from_numpy(np.asarray(mined, dtype=np.float32)).cuda()
            with torch.no_grad():
                scores = model(candidates.flatten(0, 1)).reshape(48, a.negative_candidates)
                hardest = candidates[torch.arange(48, device="cuda"), scores.argmax(1)]
            model.train()
            examples.extend(hardest.cpu().numpy())
        for _ in range(48):
            examples.append(sample(ambient[int(rng.integers(len(ambient)))]))
        x = torch.from_numpy(np.asarray(examples, dtype=np.float32)).cuda()
        y = torch.tensor([1.]*32+[0.]*96, device="cuda")[:, None]
        return x, y
    model = head(a.head).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    weights = torch.tensor([1.]*32+[20.]*96, device="cuda")[:, None]
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .97, .98, .985, .99, .992, .994, .996, .999]
    def predict(row):
        x = sequence(row)
        windows = np.lib.stride_tricks.sliding_window_view(x, 16, axis=0).transpose(0, 2, 1)
        scores = []
        with torch.inference_mode():
            for at in range(0, len(windows), 2048):
                data = torch.from_numpy(np.array(windows[at:at+2048], copy=True)).cuda()
                scores.extend(model(data).sigmoid().flatten().cpu().numpy())
        # 160 ms smoothing approximates the MCU path's 150 ms at this cadence.
        values = np.convolve(scores, [.5, .5], mode="valid")
        times = (np.arange(len(values))+2)*.08
        return values, times
    def summarize(clips, threshold):
        positives, negatives, source_recall, source_false = [], [], defaultdict(list), defaultdict(list)
        count, hours, premature = 0, 0., 0
        for row, values, times in clips:
            indices = event_indices(values, threshold)
            if row["kind"] != "generated":
                count += len(indices)
                hours += row.get("source_seconds", row["seconds"])/3600
            elif row["label"]:
                end = row["keyword_end_seconds"]
                hits = [i for i in indices if end-.1 <= times[i] <= end+.8]
                delay = (times[hits[0]]-end)*1000 if hits else None
                positives.append(delay)
                source_recall[row["engine"]].append(bool(hits))
                premature += sum(bool(times[i] < end-.1) for i in indices)
            else:
                negatives.append(bool(indices))
                source_false[row["engine"]].append(bool(indices))
        delays = [d for d in positives if d is not None]
        recalls = {e: float(np.mean(v)) for e,v in source_recall.items()}
        false = {e: float(np.mean(v)) for e,v in source_false.items()}
        return {"threshold": threshold, "positive_variants": len(positives), "detected": len(delays),
                "recall": len(delays)/len(positives), "macro_recall": float(np.mean(list(recalls.values()))),
                "recall_by_engine": recalls, "hard_negative_false_rate": float(np.mean(negatives)),
                "max_source_false_rate": max(false.values()), "hard_negative_false_by_engine": false,
                "premature_positive_events": premature, "ambient_hours": hours, "ambient_false_events": count,
                "ambient_false_per_hour": count/hours,
                "ambient_false_per_hour_upper_95": float(.5*chi2.ppf(.95, 2*(count+1))/hours),
                "latency_ms_median": float(np.median(delays)) if delays else None,
                "latency_ms_p95": float(np.quantile(delays, .95)) if delays else None}
    def validate():
        model.eval()
        clips = [(r, *predict(r)) for r in rows if r["split"] == "validation"]
        grid = [summarize(clips, t) for t in thresholds]
        eligible = [r for r in grid if r["ambient_false_per_hour"] <= .5 and r["max_source_false_rate"] <= .01]
        selected = max(eligible, key=lambda r: (r["macro_recall"], -r["ambient_false_per_hour"])) if eligible else min(
            grid, key=lambda r: (r["ambient_false_per_hour"], r["max_source_false_rate"], -r["macro_recall"]))
        rank = ((1, selected["macro_recall"], -selected["ambient_false_per_hour"]) if eligible else
                (0, -selected["ambient_false_per_hour"], -selected["max_source_false_rate"], selected["macro_recall"]))
        model.train()
        return selected, rank, grid
    a.output.mkdir(parents=True)
    started, best, history, selected_step = time.monotonic(), None, [], None
    model.train()
    for step in range(1, a.steps+1):
        x, y = batch()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, a.steps, a.lr_policy)
        optimizer.zero_grad(set_to_none=True)
        loss = (loss_fn(model(x), y)*weights).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "seconds": round(time.monotonic()-started, 1)}), flush=True)
        if step % a.eval_interval == 0 or step == a.steps:
            selected, rank, grid = validate()
            history.append({"step": step, "selected": selected, "grid": grid})
            print(json.dumps({"step": step, "validation": selected}), flush=True)
            if best is None or rank > best:
                best, selected_step = rank, step
                torch.save(model.state_dict(), a.output/"best.pt")
    selected = next(r["selected"] for r in history if r["step"] == selected_step)
    report = {"role": "Frozen official pretrained-feature desktop diagnostic, custom MLP head",
              "lr_policy": a.lr_policy, "lr_target": 1e-4 if a.lr_policy == "upstream" else 1e-3,
              "steps": a.steps, "selected_step": selected_step, "selected_validation": selected,
              "head_parameters": sum(p.numel() for p in model.parameters()), "negative_weight": 20, "negative_candidates": a.negative_candidates,
              "head": str(model), "head_architecture": a.head,
              "source_sampling": "Equal engines; 32 positive, 48 confusing-negative, 48 ambient per batch",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "model_sha256": hashlib.sha256((a.output/"best.pt").read_bytes()).hexdigest(),
              "cache_index_sha256": cache["index_sha256"], "cache_binary_sha256": cache["embeddings_sha256"],
              "source_sha256": source_sha256, "history": history,
              "seconds": round(time.monotonic()-started, 1),
              "event_policy": "80 ms cadence; two-score average; rising latch; 25-step (2 s) refractory; end window -100..+800 ms",
              "regression_or_final_scores_computed": False,
              "limits": "Float desktop reference only; not quantized MCU model or Hi ESP comparison. Validation calibration is not acceptance. Same input waveforms, different pretrained representation/head/cadence; not an architecture-only ablation."}
    (a.output/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
