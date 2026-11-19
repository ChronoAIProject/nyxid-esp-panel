"""Adapt the final five-class layer on the actual release clocks of the frozen early proposer."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

import fit_early_spelling as calibration
import train_positive_readout as positive
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha

parent = positive.parent
room = parent.room
base = parent.replay
STEM = "4090-c161-early-proposal-readout"
FIT = RESULTS / (STEM + "-fit.json")
MODES = ("trigger",)
SEED = 658227


def weights(mode):
    return RESULTS / f"{STEM}-{mode}.npz"


def sources():
    yield from calibration.training_streams()
    for meta, wave in parent.training_audio():
        yield dict(key=meta["key"], mode=meta["kind"], positive=False, text=meta.get("text", ""),
                   offset=meta.get("offset"), source_sha256=meta["source_sha256"],
                   start=16000, stop=16000 + len(wave)), np.pad(wave, (16000, 16000))


def collect(net):
    stem = calibration.replay.fit.STEM + "-early"
    front = RESULTS / (stem + "-v1.tflite")
    assert sha(front) == read(RESULTS / (stem + "-export-v1.json"))["sha256"]
    vm = REF_KERNEL(model_path=str(front))
    vm.allocate_tensors()
    native = calibration.prior.fit.native_frontend()
    patches, labels, rows = [], [], []
    for source, pcm in sources():
        raw, bands, clock = base.native_inputs(native, pcm)
        runtime = base.scoring.word
        times = runtime.times(runtime.decode(base.infer(vm, bands), clock))
        accepted = []
        for t in times:
            end = t + runtime.WAIT
            if t <= 0 or end > len(pcm):
                continue
            if source["positive"]:
                # Read only after the onset and where the future check sees at least
                # all but 120 ms of the name. Exclude later proposals on the command.
                if t < source["start"] or not source["tail"] - 1920 <= end <= source["tail"] + runtime.WAIT + 3840:
                    continue
            x = np.rint(room.window(raw, end).clip(0, 666) * (255 / 666)).astype(np.uint8)
            label = room.BASE.category(source["text"], source["positive"])
            labels.append(label)
            patches.append(x)
            accepted.append(dict(t=t, end=end, label=label, patch_sha256=base.poll.digest(x)))
        rows.append(dict(**source, pcm_sha256=base.poll.digest(pcm), proposals=times, training=accepted))
    return parent.embeddings(net, np.stack(patches)), np.array(labels), rows, sha(front)


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    initial = positive.weights("trigger")
    net = room.base.load(room.BASE.Small(read(room.base.FIT)["alphabet"]), initial,
                         read(positive.FIT)["models"]["trigger"]["weights_sha256"])
    trigger, labels, rows, front_hash = collect(net)
    old = room.BASE.WORK / "features.npz"
    assert sha(old) == read(room.BASE.SOURCES)["cache_sha256"] and sha(room.CACHE) == read(room.SOURCES)["cache_sha256"]
    with np.load(old) as dry, np.load(room.CACHE) as wet:
        anchor = torch.cat([parent.embeddings(net, x) for x in (dry["x"], wet["clean"], wet["room"])])
        original = np.r_[dry["labels"], wet["labels"], wet["labels"]]
    anchor_ids = [np.flatnonzero(original == k) for k in range(5)]
    target_ids, negative_ids = np.flatnonzero(labels == 0), np.flatnonzero(labels != 0)
    head = net.names[-1]
    prior = (head(anchor) / 2).softmax(1).detach()
    opt = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.001)
    rng = np.random.default_rng(SEED)
    losses = []
    for step in range(parent.STEPS):
        past = np.concatenate([rng.choice(pool, 12) for pool in anchor_ids])
        new = np.r_[rng.choice(target_ids, 16), rng.choice(negative_ids, 16)]
        logits = head(torch.cat((anchor[past], trigger[new])))
        truth = torch.from_numpy(np.r_[original[past], labels[new]]).cuda()
        loss = F.cross_entropy(logits, truth, label_smoothing=.025)
        loss += 4 * F.kl_div((logits[:len(past)] / 2).log_softmax(1), prior[past], reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, 99, parent.STEPS - 1):
            losses.append(float(loss.detach()))
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    with np.load(initial) as original_weights:
        changed = [k for k, v in state.items() if not np.array_equal(v, original_weights[k])]
    assert set(changed) == {"names.2.weight", "names.2.bias"}
    target = weights("trigger")
    with target.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out, **state)
    target.with_suffix(".tmp").replace(target)
    save(FIT, dict(records=rows, models={"trigger": dict(weights_sha256=sha(target), changed=changed, losses=losses)},
         classes={str(k): int(v) for k, v in Counter(labels).items()}, steps=parent.STEPS, seed=SEED, gpu=gpu,
         parent_sha256=sha(initial), front_sha256=front_hash, anchor_sha256=[sha(old), sha(room.CACHE)],
         script_sha256=sha(Path(__file__)), training_sources_sha256=sha(Path(calibration.__file__)),
         policy="Initialize from the trained trigger head. Freeze all feature layers. Train on full proposal+360 ms "
                "windows of the early INT8 proposer from training-only synthesized dry/room speech, LibriSpeech "
                "train-clean, daily training data and music. Balance 16 targeted positive and 16 negative proposals "
                "with 12 anchors per original class; distill on anchors. Retain the signed Nyx >= 0 threshold."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("train", "export"))
    if parser.parse_args().action == "train":
        train()
    else:
        from export_c161_polling_rooms import main
        main(sys.modules[__name__])
