"""Train a small independent keyword classifier over preceding native frontend bands."""
import hashlib

import numpy as np
import torch
from torch import nn

import generate_c161_rhythm_pair_v1 as pairs
from fit_c161_history_gain_v1 import DAILY, RESULTS, frontend, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_midlevel_v1 import sources as admitted

STEM = "4090-c161-word-window"
SEED = 53883
STEPS = 1100
WIDTH = 32


def sources():
    yield from admitted()
    ledger = read(pairs.META)
    assert sha(pairs.WAVES) == ledger["archive_sha256"]
    with np.load(pairs.WAVES) as waves:
        for row in ledger["records"]:
            used = set()
            for route, view in row["variants"].items():
                if view["pcm_sha256"] in used:
                    continue
                used.add(view["pcm_sha256"])
                pcm = waves[view["key"]]
                assert hashlib.sha256(pcm.tobytes()).hexdigest() == view["pcm_sha256"]
                yield dict(key=pairs.STEM+"_"+row["key"]+"_"+route, voice=row["voice"], positive=row["positive"],
                           text=row["text"], end=view["end"]), pcm


def bands(pcm):
    features, clock = frontend(pcm)
    # Every physical call contributes three 10 ms frames; average each pair of adjacent bands.
    pooled = features.reshape(-1, 3, 20, 2).mean(axis=(1, 3))
    return np.rint(pooled * (255 / 26)).clip(0, 255).astype(np.uint8), clock


def training_data(render=None):
    histories = [r for r in read(RESULTS/DAILY)["admitted"] if r["request"]["group"] == "conversation"]
    rng = np.random.default_rng(SEED)
    rows, arrays, labels, hard = [], [], [], []
    for i, (row, pcm) in enumerate(sources()):
        prefix_id = i % len(histories)
        prefix = load_wav(histories[prefix_id]["wav"])
        if i % 2:
            prefix = np.zeros_like(prefix)
        gap = (4000, 8000, 12000)[i % 3]
        onset = len(prefix) + gap
        full = np.r_[prefix, np.zeros(gap, np.int16), pcm, np.zeros(8000, np.int16)]
        clean_digest = hashlib.sha256(full.tobytes()).hexdigest()
        condition = {}
        if render is not None:
            full, condition = render(row, full, onset)
            condition["clean_stream_sha256"] = clean_digest
        frame, clock = bands(full)
        pool = np.lib.stride_tricks.sliding_window_view(frame, WIDTH, axis=0)
        ends = clock[WIDTH-1:]
        positive = np.zeros(len(ends), bool)
        if row["positive"]:
            tail = onset + row["end"]
            positive = (ends >= tail-480) & (ends <= tail+2880)
            negative = ((ends > 0) & (ends <= onset)) | (ends > tail+16000)
            assert positive.any()
        else:
            negative = ends > 0
        for mask, label in ((positive, 1), (negative, 0)):
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            if not label:
                near = ids[(ends[ids] > onset+3200) & (ends[ids] < onset+32000)]
                ids = np.unique(np.r_[rng.choice(ids, min(len(ids), 20), replace=False),
                                     rng.choice(near, min(len(near), 16), replace=False)])
            arrays.extend(pool[ids])
            labels.extend([label]*len(ids))
            hard.extend([not row["positive"] and row["text"].lower().startswith("hey ")]*len(ids))
        rows.append(dict(**row, **condition, pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), prefix_id=histories[prefix_id]["index"],
                         quiet=bool(i%2), onset=onset, stream_sha256=hashlib.sha256(full.tobytes()).hexdigest()))
    return torch.from_numpy(np.stack(arrays)).cuda().float()/255, np.array(labels), np.array(hard), rows


class WordWindow(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv1d(20, 32, 5), nn.ReLU(), nn.Conv1d(32, 24, 3, stride=2), nn.ReLU(),
                         nn.Flatten(), nn.Linear(24*13, 24), nn.ReLU(), nn.Linear(24, 1))


def train(stem=STEM, render=None):
    torch.set_num_threads(2)
    assert "4090" in torch.cuda.get_device_name()
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(SEED)
    x, y, hard, rows = training_data(render)
    pos, neg, near = (np.flatnonzero(v) for v in (y == 1, y == 0, (y == 0) & hard))
    network = WordWindow().cuda()
    opt = torch.optim.AdamW(network.parameters(), lr=.0012, weight_decay=.001)
    log = []
    for step in range(STEPS):
        ids = np.r_[rng.choice(pos, 64), rng.choice(neg, 192), rng.choice(near, 64)]
        scores = network(x[ids])[:, 0]
        # Mine the 32 strongest confusions from each sampled negative pool.
        loss = nn.functional.softplus(4.5-scores[:64]).mean() + nn.functional.softplus(scores[64:]+2).mean()
        loss += nn.functional.softplus(scores[64:].topk(32).values + 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, 99, 499, STEPS-1):
            log.append(dict(step=step+1, loss=loss.item()))
    assert np.isfinite(log[-1]["loss"])
    path = RESULTS / f"{stem}-weights-v1.npz"
    with path.open("xb") as f:
        np.savez_compressed(f, **{k:v.detach().cpu().numpy() for k,v in network.state_dict().items()})
    ids = np.r_[rng.choice(pos, 96, replace=False), rng.choice(neg, 160, replace=False)]
    calibration = x[ids]
    with torch.no_grad():
        prediction = network(calibration).sigmoid()[:,0].cpu().numpy()
    samples = RESULTS / f"{stem}-calibration-v1.npz"
    with samples.open("xb") as f:
        np.savez_compressed(f, inputs=(calibration.cpu().numpy()*255).round().astype(np.uint8),
                            probabilities=prediction, ids=ids)
    save(RESULTS/f"{stem}-training-v1.json", dict(gpu=torch.cuda.get_device_name(), seed=SEED, steps=STEPS,
         positive_windows=len(pos), negative_windows=len(neg), training=rows, history=log, weights_sha256=sha(path),
         calibration_sha256=sha(samples)))


if __name__ == "__main__":
    train()
