"""Match a room window to its frozen clean teacher, with a same-batch CE control."""
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

import train_c161_polling_rooms as room
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-paired-rooms"
FIT = RESULTS / (STEM + "-fit.json")
MODES = ("ce", "teacher")
TEMPERATURE = 2.0


def weights(mode):
    return RESULTS / f"{STEM}-{mode}.npz"


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = room.BASE.WORK / "features.npz"
    assert sha(old) == read(room.BASE.SOURCES)["cache_sha256"]
    assert sha(room.CACHE) == read(room.SOURCES)["cache_sha256"]
    with np.load(old) as z:
        original = torch.from_numpy(z["x"]).cuda()
        original_y = z["labels"]
    with np.load(room.CACHE) as z:
        clean, wet = (torch.from_numpy(z[key]).cuda() for key in ("clean", "room"))
        paired_y = z["labels"]
    assert clean.shape == wet.shape and len(wet) == len(paired_y)
    groups = [[np.flatnonzero(y == k) for k in range(5)] for y in (original_y, paired_y)]
    cfg = read(room.base.FIT)
    teacher_path = room.weights("clean")
    teacher = room.base.load(room.BASE.Small(cfg["alphabet"]), teacher_path,
                             read(room.FIT)["models"]["clean"]["weights_sha256"])
    with torch.inference_mode():
        logits = torch.cat([teacher(batch.float().unsqueeze(1)/255)[0] for batch in clean.split(128)])
        soft = (logits/TEMPERATURE).softmax(dim=1)
        teacher_predictions = logits.argmax(1).cpu().numpy()
    parent = room.base.weight_file("posterior")
    reports = {}
    for mode in MODES:
        rng = np.random.default_rng(room.SEED)
        torch.manual_seed(room.SEED)
        net = room.base.load(room.BASE.Small(cfg["alphabet"]), parent,
                             cfg["models"]["posterior"]["weights_sha256"]).train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.0002, weight_decay=.004)
        batches = hashlib.sha256()
        history = []
        for step in range(room.STEPS):
            i, j = [np.concatenate([rng.choice(pool, 8) for pool in classes]) for classes in groups]
            batches.update(np.stack((i, j)).astype("<i4").tobytes())
            x = torch.cat((original[i], clean[j], wet[j])).float().unsqueeze(1)/255
            y = torch.from_numpy(np.r_[original_y[i], paired_y[j], paired_y[j]]).cuda()
            logits = net(x)[0]
            ce = F.cross_entropy(logits, y, label_smoothing=.025)
            wet_logits = logits[-len(j):] / TEMPERATURE
            kl = F.kl_div(wet_logits.log_softmax(dim=1), soft[j], reduction="batchmean")
            loss = ce + TEMPERATURE**2 * kl if mode == "teacher" else ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in (0, 99, room.STEPS-1):
                history.append(dict(step=step+1, ce=ce.item(), kl=kl.item(), loss=loss.item()))
        with weights(mode).open("xb") as f:
            np.savez_compressed(f, **{k:v.cpu().numpy() for k,v in net.state_dict().items()})
        reports[mode] = dict(history=history, batch_sha256=batches.hexdigest(), weights_sha256=sha(weights(mode)))
        print(mode, history[-1], flush=True)
    assert reports["ce"]["batch_sha256"] == reports["teacher"]["batch_sha256"]
    paths = (Path(__file__), parent, room.base.FIT, teacher_path, room.FIT, room.CACHE, room.SOURCES, old, room.BASE.SOURCES)
    save(FIT, dict(models=reports, gpu=gpu, torch=torch.__version__, steps=room.STEPS, seed=room.SEED,
                   temperature=TEMPERATURE, files={str(p):sha(p) for p in paths}, classes=room.BASE.CLASSES,
                   training_counts=[[len(pool) for pool in classes] for classes in groups],
                   teacher_clean_training_confusion=np.bincount(5*paired_y+teacher_predictions, minlength=25).reshape(5,5).tolist(),
                   policy="Shared batches of 40 original, 40 clean and 40 aligned room examples. CE on all 120. Teacher arm adds T^2 KL on 40 room windows against frozen clean-head posteriors on their clean partners. Single-word proposal + 360 ms, Nyx code >=0, union with frozen C161. Report the first notification on the 80 matched scenes and replay all consumed background proposals."))


if __name__ == "__main__":
    train()
