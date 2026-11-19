"""Distill class logits and their six time-bin contributions into the native classifier."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import train_c161_onset_letters as base
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-onset-evidence"
FIT = RESULTS / f"{STEM}-fit.json"
MODES = ("posterior", "time_bins")


def weight_file(mode):
    return RESULTS / f"{STEM}-{mode}.npz"


def load(net, path, expected):
    assert sha(path) == expected
    with np.load(path) as data:
        net.load_state_dict({k: torch.from_numpy(data[k]) for k in data})
    return net.cuda().eval()


@torch.no_grad()
def teacher_targets(net, latent):
    # Split the affine map by time. The first-layer bias and final bias are
    # shared evenly; all bins use the same ReLU gates as the actual classifier.
    norm = net[0](latent).reshape(-1, 768, 6)
    hidden, output = net[1], net[3]
    parts = torch.einsum("bct,hct->bth", norm, hidden.weight.reshape(-1, 768, 6))
    parts = parts + hidden.bias / 6
    parts = parts * (parts.sum(1) > 0).unsqueeze(1)
    local = F.linear(parts, output.weight, output.bias/6)
    logits = net(latent)
    torch.testing.assert_close(local.sum(1), logits, atol=2e-4, rtol=2e-4)
    return logits, local


def student_targets(net, x):
    out = net.encoder(x).permute(0, 1, 3, 2).flatten(1, 2)
    pooled = net.names[0](net.temporal(out))
    head = net.names[-1]
    bins = torch.einsum("bct,kct->btk", pooled, head.weight.reshape(5, -1, 6))
    local = bins + head.bias / 6
    return head(pooled.flatten(1)), local


def centered(x):
    return x-x.mean(dim=-1, keepdim=True)


def main():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cache = base.WORK / "features.npz"
    sources = read(base.SOURCES)
    assert sha(cache) == sources["cache_sha256"]
    previous = read(base.FIT)
    teacher = load(base.readout(), base.weight_file("pretrained"), previous["models"]["pretrained"]["weights_sha256"])
    with np.load(cache) as data:
        x = torch.from_numpy(data["x"]).cuda().float().unsqueeze(1)/255
        labels = torch.from_numpy(data["labels"]).cuda()
        latent = torch.from_numpy(data["latent"]).cuda().float()
        alphabet = data["logits"].shape[-1]
    logits, local = teacher_targets(teacher, latent)
    soft = (logits/2).softmax(dim=-1)
    pools = [np.flatnonzero(labels.cpu().numpy() == k) for k in range(5)]
    assert [len(p) for p in pools] == previous["train_counts"]
    models = {}
    for mode in MODES:
        torch.manual_seed(base.SEED)
        rng = np.random.default_rng(base.SEED)
        net = base.Small(alphabet).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.004)
        history = []
        for step in range(base.STEPS):
            ix = np.concatenate([rng.choice(pool, 16) for pool in pools])
            pred, bins = student_targets(net, x[ix])
            loss = F.cross_entropy(pred, labels[ix], label_smoothing=.025)
            loss = loss + 4*F.kl_div((pred/2).log_softmax(dim=-1), soft[ix], reduction="batchmean")
            if mode == "time_bins":
                loss = loss + F.smooth_l1_loss(centered(bins), centered(local[ix]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0, 99, 499, base.STEPS-1):
                history.append(dict(step=step+1, loss=float(loss.detach())))
        net.eval()
        with torch.inference_mode():
            pred, bins = student_targets(net, x[:32])
            torch.testing.assert_close(pred, net(x[:32])[0])
            torch.testing.assert_close(pred, bins.sum(1), atol=5e-5, rtol=5e-5)
        with weight_file(mode).open("xb") as handle:
            np.savez_compressed(handle, **{k:v.cpu().numpy() for k,v in net.state_dict().items()})
        models[mode] = dict(history=history, weights_sha256=sha(weight_file(mode)))
        print(mode, history, flush=True)
    save(FIT, dict(models=models, gpu=gpu, torch=torch.__version__, steps=base.STEPS, seed=base.SEED,
                   classes=base.CLASSES, counts=previous["train_counts"], alphabet=alphabet,
                   files={str(p):sha(p) for p in (Path(__file__), base.SOURCES, base.FIT,
                          base.weight_file("pretrained"), cache)},
                   policy="Two matched native fits, class-balanced batches, 1000 steps. CE plus class KL at T=2. Second fit additionally uses smooth L1 on class-centered contributions of six time bins. No evaluation audio, no threshold sweep; accept Nyx >= 0.5 at onset + 1.32 s."))


if __name__ == "__main__":
    main()
