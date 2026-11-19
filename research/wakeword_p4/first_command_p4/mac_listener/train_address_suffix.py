"""Compare intact and command-muted training windows at identical word-proposal clocks."""
import argparse
import ctypes
from hashlib import sha256
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

import generate_c161_rhythm_pair_v1 as rhythm
import train_positive_readout as previous
from check_c161_word_parts import mute
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha

parent = previous.parent
room = parent.room
replay = parent.replay
STEM = "4090-c161-address-suffix"
FIT = RESULTS / f"{STEM}-fit.json"
MODES = ("intact", "paired")
SEED = 1926413
STEPS = 240


def weights(mode):
    return RESULTS / f"{STEM}-{mode}.npz"


def collect(front):
    ledger = read(rhythm.META)
    assert sha(rhythm.WAVES) == ledger["archive_sha256"]
    training = {r["key"]:r for r in read(RESULTS / f"{parent.music.word.STEM}-training-v1.json")["training"]}
    vm = REF_KERNEL(model_path=str(front))
    vm.allocate_tensors()
    lib = replay.BUILD / "frontend.so"
    assert sha(lib) == read(replay.BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    views = {"intact": [], "muted": []}
    rows = []
    with np.load(rhythm.WAVES) as waves:
        for r in ledger["records"]:
            if r["kind"] == "daily":
                continue
            comma = r["phones"].index(",")
            first = comma + 1 + re.search(r"[^\sˌˈ]", r["phones"][comma + 1:]).start()
            for variant, v in r["variants"].items():
                key = f"{rhythm.STEM}_{r['key']}_{variant}"
                if key not in training:
                    assert variant == "rhythm" and v["pcm_sha256"] == r["variants"]["native"]["pcm_sha256"]
                    continue
                src = training[key]
                wave = waves[v["key"]]
                assert replay.poll.digest(wave) == src["pcm_sha256"] == v["pcm_sha256"]
                assert src["text"] == r["text"] and src["positive"] == r["positive"]
                edges = np.array(r["natural_durations"])
                if variant == "rhythm":
                    edges += r["delta"]
                ticks = np.cumsum(edges) * 400
                assert ticks[comma] == v["end"] and ticks[-1] == len(wave)
                onset = 32000
                pcm = np.pad(wave, (onset, 16000))
                tail, command = onset + v["end"], onset + int(ticks[first])
                raw, bands, clock = replay.native_inputs(native, pcm)
                scores = replay.infer(vm, bands)
                ts = replay.scoring.word.times(replay.scoring.word.decode(scores, clock))
                matches = [t for t in ts if abs(t-tail) <= 3840]
                sums = np.convolve(scores.astype(np.int32), np.ones(5, np.int32), "valid")
                eligible = np.flatnonzero(abs(clock[4:]-tail) <= 3840)
                peak = int(clock[4:][eligible[np.argmax(sums[eligible])]])
                t = matches[0] if matches else peak
                end = t + replay.scoring.word.WAIT
                edited = mute(pcm, command, len(pcm))
                edited_raw, _, _ = replay.native_inputs(native, edited)
                row = dict(source=key, kind=r["kind"], label=room.BASE.category(r["text"], r["positive"]),
                           text=r["text"], voice=r["voice"], variant=variant, pcm_sha256=src["pcm_sha256"],
                           stream_sha256=replay.poll.digest(pcm), muted_sha256=replay.poll.digest(edited),
                           name_end=tail, command_start=command, proposals=ts, used_proposal=bool(matches),
                           peak=peak, end=end)
                for kind, features in (("intact", raw), ("muted", edited_raw)):
                    patch = np.rint(room.window(features, end).clip(0, 666) * (255. / 666)).astype(np.uint8)
                    row[kind+"_sha256"] = replay.poll.digest(patch)
                    views[kind].append(patch)
                rows.append(row)
    return {k:np.stack(v) for k,v in views.items()}, rows


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    front = RESULTS / f"{parent.music.STEM}-v1.tflite"
    assert sha(front) == read(RESULTS / f"{parent.music.STEM}-export-v1.json")["sha256"]
    views, rows = collect(front)
    labels = np.array([r["label"] for r in rows])
    families = [np.flatnonzero([r["kind"] == name for r in rows]) for name, _ in rhythm.LABELS]
    prior_path = previous.weights("trigger")
    cfg = read(room.base.FIT)
    prior_hash = read(previous.FIT)["models"]["trigger"]["weights_sha256"]

    def load_model():
        return room.base.load(room.BASE.Small(cfg["alphabet"]), prior_path, prior_hash)

    teacher = load_model()
    dry_cache = room.BASE.WORK / "features.npz"
    assert sha(dry_cache) == read(room.BASE.SOURCES)["cache_sha256"] and sha(room.CACHE) == read(room.SOURCES)["cache_sha256"]
    with np.load(dry_cache) as z, np.load(room.CACHE) as w:
        anchors = np.concatenate((z["x"], w["clean"], w["room"]))
        anchor_labels = np.r_[z["labels"], w["labels"], w["labels"]]
    pools = [np.flatnonzero(anchor_labels == k) for k in range(5)]
    anchors = torch.from_numpy(anchors).cuda().unsqueeze(1)
    views = {k:torch.from_numpy(v).cuda().unsqueeze(1) for k,v in views.items()}
    with torch.no_grad():
        anchor_targets = torch.cat([(teacher(chunk.float()/255)[0]/2).softmax(1) for chunk in anchors.split(96)])
    reports = {}
    for mode in MODES:
        net = load_model()
        net.requires_grad_(False)
        # Keep batch-normalization statistics and affine parameters frozen, including in the updated temporal block.
        net.temporal[3].requires_grad_(True)
        net.names[-1].requires_grad_(True)
        opt = torch.optim.AdamW([v for v in net.parameters() if v.requires_grad], lr=.00012, weight_decay=.001)
        rng = np.random.default_rng(SEED)
        batches = sha256()
        history = []
        for step in range(STEPS):
            a = np.concatenate([rng.choice(pool, 8) for pool in pools])
            ids = np.concatenate([rng.choice(family, 4) for family in families])
            batches.update(np.r_[a, ids].astype("<i4").tobytes())
            second = views["muted" if mode == "paired" else "intact"][ids]
            data = torch.cat((anchors[a], views["intact"][ids], second)).float()/255
            logits = net(data)[0]
            y = torch.from_numpy(np.r_[anchor_labels[a], labels[ids], labels[ids]]).cuda()
            ce = F.cross_entropy(logits, y, label_smoothing=.025)
            kl = F.kl_div((logits[:len(a)]/2).log_softmax(1), anchor_targets[a], reduction="batchmean")
            loss = ce + 4*kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0, 99, STEPS-1):
                history.append(dict(step=step+1, loss=loss.item(), ce=ce.item(), kl=kl.item()))
        arrays = {k:v.cpu().numpy() for k,v in net.state_dict().items()}
        with np.load(prior_path) as orig:
            changed = [k for k,v in arrays.items() if not np.array_equal(v, orig[k])]
        assert set(changed) == {"names.2.weight", "names.2.bias", "temporal.3.weight", "temporal.3.bias"}
        with weights(mode).open("xb") as f:
            np.savez_compressed(f, **arrays)
        reports[mode] = dict(weights_sha256=sha(weights(mode)), history=history, changed=changed,
                             updated_parameters=sum(arrays[k].size for k in changed), batches_sha256=batches.hexdigest())
    assert reports["intact"]["batches_sha256"] == reports["paired"]["batches_sha256"]
    paths = [rhythm.META, rhythm.WAVES, dry_cache, room.CACHE, previous.FIT, prior_path,
             RESULTS / f"{parent.music.word.STEM}-training-v1.json", replay.BUILD / "frontend.so"]
    save(FIT, dict(models=reports, gpu=gpu, torch=torch.__version__, seed=SEED, steps=STEPS,
                   front_sha256=sha(front), files={str(p):sha(p) for p in paths}, records=rows,
                   script_sha256=sha(Path(__file__)), policy="Both arms inherit the trigger readout. At the first "
                   "training-only native proposal within 240 ms of the marked word tail, else the largest five-step "
                   "sum over that interval, read 360 ms later. Same seven balanced name families and seeds, "
                   "same original anchors and teacher posteriors. Compare two copies of the intact window to "
                   "one intact plus one with speech from the synthesizer's first command phone muted."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("train", "export", "score"))
    p.add_argument("groups", nargs="*")
    args = p.parse_args()
    experiment = sys.modules[__name__]
    if args.action == "train":
        train()
    elif args.action == "export":
        from export_c161_polling_rooms import main
        main(experiment)
    else:
        from score_proposal_readout import GROUPS, main
        for group in args.groups:
            if group not in GROUPS:
                raise ValueError(group)
            main(group, experiment)
