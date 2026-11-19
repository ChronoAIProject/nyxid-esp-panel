"""Compare fixed-clock and CTC-aligned Nyx/Nick discrimination using paired training utterances."""
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_early_proposal_readout as previous
from train_name_final_s import integer_model, integer_scores
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

replay = previous.calibration.replay
base = replay.base
STEM = "4090-c161-aligned-nick"
FIT = RESULTS / (STEM + "-fit.json")
MODES = ("fixed", "aligned")
SEED = 5117


def inputs(patch, end_frame):
    # 71 character outputs from 144 ten-ms mel frames. At each offset retain
    # the 20 adjacent-band averages and the fraction of frames actually available.
    features = []
    bands = (patch.reshape(144, 20, 2).astype(np.uint16).sum(2) + 1) // 2
    for offset in range(-24, 25, 4):
        start = end_frame + offset
        ids = np.arange(start - 1, start + 3)
        frames = bands[ids[(ids >= 0) & (ids < len(bands))]]
        values = (frames.sum(0) + 2) // 4
        features.extend([*values, len(frames) * 63])
    return np.array(features, dtype=np.uint8)


class Extractor:
    def __init__(self):
        self.char = replay.prior.model()
        self.vocab = read(replay.prior.WAV2VEC / "vocab.json")

    def __call__(self, patch):
        codes = replay.prior.outputs(self.char, patch)
        paths = [replay.gap.diagnostic.trace(codes, lead + "NICK", self.vocab) for lead in ("HEY", "HAY")]
        best = max(paths, key=lambda p: p["normalized"])
        frame = best["characters"][-1]["last"]
        # The fixed control looks at the same spectrogram at the proposal clock.
        at = 143 - previous.calibration.WAIT // 160
        return dict(fixed=inputs(patch, at), aligned=inputs(patch, 2 * frame + 1)), dict(
            nick=best["word"], k_frame=frame, score=best["normalized"], char_sha256=base.poll.digest(codes))


def collect():
    native = replay.prior.fit.native_frontend()
    extractor = Extractor()
    features = defaultdict(list)
    records = []
    inventory = read(previous.FIT)["records"]
    for (meta, pcm), row in zip(previous.sources(), inventory, strict=True):
        assert meta["key"] == row["key"] and meta["source_sha256"] == row["source_sha256"]
        eligible = [c for c in row["training"] if c["label"] in (0, 1)]
        if not eligible:
            continue
        assert base.poll.digest(pcm) == row["pcm_sha256"]
        raw = base.native_inputs(native, pcm)[0]
        for c in eligible:
            patch = np.rint(previous.room.window(raw, c["end"]).clip(0,666) * (255/666)).astype(np.uint8)
            assert base.poll.digest(patch) == c["patch_sha256"]
            xs, details = extractor(patch)
            for mode, x in xs.items():
                features[mode].append(x)
            records.append(dict(voice=meta["voice"], text=meta["text"], mode=meta["mode"], key=meta["key"],
                                positive=meta["positive"], **c, **details))
    return {k: np.stack(v) for k,v in features.items()}, records


def pair_key(row):
    text = row["text"].lower().replace(",", " ")
    words = re.findall("[a-z]+", text)
    return (row["voice"], row["mode"], words[0], tuple(words[2:]))


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    features, records = collect()
    voices = sorted({r["voice"] for r in records}, key=lambda v: hashlib.sha256(v.encode()).digest())
    calibrators = set(voices[:3])
    fit = np.array([r["voice"] not in calibrators for r in records])
    target = np.array([r["positive"] for r in records])
    paired = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        if fit[i]:
            paired[pair_key(r)][r["positive"]].append(i)
    pairs = np.array([(p,n) for v in paired.values() for p in v[True] for n in v[False]], dtype=int)
    assert len(pairs) and np.all(fit[pairs])
    pools = [np.flatnonzero(fit & (target == label)) for label in (True, False)]
    models = {}
    for mode in MODES:
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        x = torch.from_numpy(features[mode]).cuda().float()/255
        net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.002)
        losses = []
        for step in range(600):
            ids = np.r_[rng.choice(pools[0], 32), rng.choice(pools[1], 32)]
            contrast = pairs[rng.integers(len(pairs), size=32)]
            logits = net(x[ids]).flatten()
            relative = net(x[contrast]).squeeze(2)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(target[ids]).cuda().float())
            loss = loss + F.softplus(1 - relative[:,0] + relative[:,1]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0, 199, 599):
                losses.append(float(loss.detach()))
        params, scale = integer_model(net)
        scores = integer_scores(features[mode], params)
        hold = ~fit & target
        minimum = int(np.quantile(scores[hold], .05, method="lower"))
        with torch.no_grad():
            float_logits = net(x).flatten().cpu().numpy()
        path = RESULTS / f"{STEM}-{mode}.npz"
        with path.open("xb") as out:
            np.savez_compressed(out, **params)
        models[mode] = dict(file=path.name, sha256=sha(path), threshold=minimum, scale=scale, losses=losses,
                            max_abs_error=float(np.max(np.abs(scores * scale - float_logits))),
                            calibration_targets=int(sum(hold)), calibration_targets_pass=int(sum(hold & (scores >= minimum))),
                            calibration_nicks=int(sum(~fit & ~target)), calibration_nicks_pass=int(sum(~fit & ~target & (scores >= minimum))))
        for r, score in zip(records, scores, strict=True):
            r[mode] = int(score)
    save(FIT, dict(records=records, models=models, calibration_voices=sorted(calibrators), pairs=len(pairs),
         paired_groups=sum(bool(v[True] and v[False]) for v in paired.values()), gpu=gpu, steps=600, seed=SEED,
         parent_fit_sha256=sha(previous.FIT), script_sha256=sha(Path(__file__)),
         policy="Compare the fixed proposal clock and the end of the most likely HEYNICK/HAYNICK path. "
                "Same 520 ms mel neighbourhood, architecture, seed and batches. Pairwise ranking matches "
                "voice, dry/room condition, greeting and command text. Three training voices select a "
                "lower fifth-percentile positive cutoff. No development sample in training or calibration."))


if __name__ == "__main__":
    train()
