"""Learn whether a compact local acoustic window contains a name-final /s/ using training TTS timing."""
from hashlib import sha256
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_c161_onset_names as onset
import score_spelling_gap as gap
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha

base, prior = gap.study.base, gap.prior
STEM = "4090-c161-name-final-s"
WORK = Path(".local/wakeword/c161-name-final-s")
CACHE = WORK / "frames.npz"
DATA = RESULTS / f"{STEM}-training.json"
FIT = RESULTS / f"{STEM}-fit.json"
SEED = 162817
# Native frontend frames are 10 ms apart. Score only points having this entire context.
OFFSETS = np.array([-24, -16, -12, -8, -4, 0, 4, 8, 12])
CENTERS = np.arange(-OFFSETS.min(), 144 - OFFSETS.max())
MODES = ("any_s", "name_s")


def features(patch):
    # Average adjacent frequency channels and keep unsigned integer features through deployment.
    bands = (patch.reshape(144, 20, 2).astype(np.uint16).sum(-1) + 1) // 2
    return bands[CENTERS[:, None] + OFFSETS].reshape(len(CENTERS), -1).astype(np.uint8)


def collect():
    WORK.mkdir(exist_ok=True, parents=True)
    expected = [r for r in read(onset.SOURCES)["records"] if r["kind"] == "kokoro"]
    model = RESULTS / f"{prior.fit.parent.music.STEM}-v1.tflite"
    assert sha(model) == read(RESULTS / f"{prior.fit.parent.music.STEM}-export-v1.json")["sha256"]
    vm = REF_KERNEL(model_path=str(model))
    vm.allocate_tensors()
    native = prior.fit.native_frontend()
    rows, xs, ys = [], [], []
    for (s, wave), witness in zip(onset.synthesize(), expected, strict=True):
        assert s["text"] == witness["text"] and s["voice"] == witness["voice"] and s["duration"] == witness["duration"]
        phones = s["phones"]
        split = re.search(r"[, ]+[^, ]+[, ]+", phones)
        assert split
        words = list(re.finditer(r"[^ ,.!?]+", phones))
        name, first = words[1:3]
        edges = 32000 + np.cumsum(s["duration"], dtype=np.int32) * 400
        assert len(edges) == len(phones) + 2 and edges[-1] == 32000 + len(wave)
        # For phone with character index i, the leading special token ends at edges[0].
        tail = int(edges[name.end()])
        cmd_start = first.start()
        while phones[cmd_start] in "ˈˌ":
            cmd_start += 1
        original = np.pad(wave, (32000, 16000))
        raw, bands, clock = base.native_inputs(native, original)
        code = base.infer(vm, bands)
        prop = base.scoring.word.times(base.scoring.word.decode(code, clock))
        actual = [t for t in prop if abs(t-tail) <= 3840]
        eligible = np.flatnonzero(abs(clock[4:] - tail) <= 3840)
        summed = np.convolve(code.astype(np.int32), np.ones(5, np.int32), "valid")
        t = actual[0] if actual else int(clock[4:][eligible[np.argmax(summed[eligible])]])
        end = t + base.scoring.word.WAIT
        patch = np.rint(prior.fit.room.window(raw, end).clip(0, 666) * (255. / 666)).astype(np.uint8)
        # The newest 30 ms feature ends at 'end', so its centre is end-15ms.
        at = end - (143-CENTERS)*160 - 240
        labels = np.zeros(len(at), np.uint8)
        if s["label"] == 0:
            i = name.end() - 1
            assert phones[i] == "s"
            labels[(at >= edges[i]) & (at < edges[i+1])] = 1
        if phones[cmd_start] == "s":
            labels[(at >= edges[cmd_start]) & (at < edges[cmd_start+1])] = 2
        xs.append(features(patch))
        ys.append(labels)
        rows.append(dict(voice=s["voice"], text=s["text"], label=s["label"], clock=end, actual=bool(actual),
                         source_pcm_sha256=base.poll.digest(wave), archived_pcm_sha256=witness["pcm_sha256"],
                         patch_sha256=base.poll.digest(patch), labels_sha256=base.poll.digest(labels),
                         target_frames=int(sum(labels==1)), command_frames=int(sum(labels==2))))
    with CACHE.open("xb") as f:
        np.savez_compressed(f, x=np.stack(xs), y=np.stack(ys))
    save(DATA, dict(records=rows, cache_sha256=sha(CACHE), sources_sha256=sha(onset.SOURCES),
                    proposer_sha256=sha(model), script_sha256=sha(Path(__file__)), offsets=OFFSETS.tolist()))


def integer_model(net):
    params = {}
    scale = 1 / 255
    for i, layer in enumerate((net[0], net[2])):
        weight, bias = [x.detach().cpu().numpy().astype(np.float64) for x in (layer.weight, layer.bias)]
        unit = abs(weight).max() / 127
        scale *= unit
        params[f"w{i}"] = np.rint(weight / unit).astype(np.int8)
        params[f"b{i}"] = np.rint(bias / scale).astype(np.int64)
    return params, scale


def integer_scores(x, params):
    h = x.astype(np.int32) @ params["w0"].T.astype(np.int32) + params["b0"]
    logits = h.clip(min=0) @ params["w1"].T.astype(np.int64) + params["b1"]
    return logits[..., 0]


def train():
    assert "4090" in torch.cuda.get_device_name() and sha(CACHE) == read(DATA)["cache_sha256"]
    torch.set_num_threads(1)
    src = read(DATA)["records"]
    voices = sorted({s["voice"] for s in src}, key=lambda v: sha256(v.encode()).digest())
    calibration = set(voices[:3])
    fit = np.array([s["voice"] not in calibration for s in src])
    target = np.array([s["label"] == 0 for s in src])
    with np.load(CACHE) as cache:
        bank, labels = cache["x"], cache["y"]
    x = torch.from_numpy(bank[fit].reshape(-1, bank.shape[-1])).cuda().float() / 255
    y = labels[fit].ravel()
    pools = [np.flatnonzero(y == k) for k in range(3)]
    reports = {}
    for mode in MODES:
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        net = nn.Sequential(nn.Linear(x.shape[-1], 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.002)
        for _ in range(700):
            ids = np.concatenate([rng.choice(pool, n) for pool, n in zip(pools, (128, 64, 64), strict=True)])
            expected = y[ids] > 0 if mode == "any_s" else y[ids] == 1
            z = net(x[ids]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(z, torch.from_numpy(expected).cuda().float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        weights, scale = integer_model(net)
        path = RESULTS / f"{STEM}-{mode}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, **weights)
        scores = integer_scores(bank, weights).max(1)
        limit = int(np.quantile(scores[~fit & target], .05, method="lower"))
        with torch.no_grad():
            z = net(torch.from_numpy(bank[~fit]).cuda().float()/255).squeeze(-1).amax(1).cpu().numpy()
        reports[mode] = dict(file=path.name, sha256=sha(path), threshold=limit, logit_scale=scale,
                             max_abs_quantization_error=float(np.max(abs(z-scores[~fit]*scale))),
                             window_scores=scores.tolist(), calibration_accepts=int(sum(scores[~fit & target]>=limit)),
                             calibration_negatives_accepted=int(sum(scores[~fit & ~target]>=limit)))
    save(FIT, dict(models=reports, calibration_voices=sorted(calibration), training_sha256=sha(DATA),
                   cuda=torch.cuda.get_device_name(), script_sha256=sha(Path(__file__)),
                   calibration_positives=int(sum(~fit & target)), seed=SEED,
                   policy="Same 700 balanced frame batches in both conditions. In the control, target includes /s/ "
                   "at the end of a Nyx name or at the beginning of the command. In the treatment only Nyx's /s/ "
                   "is positive. Hold out three of twelve training voices for a fixed lower fifth percentile of "
                   "their positive integer window scores. Maximum over all complete frame contexts in one proposal."))


if __name__ == "__main__":
    {"collect": collect, "train": train}[sys.argv[1]]()
