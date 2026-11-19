"""Adapt only the five-class readout to random or high-scoring windows in the same training audio."""
import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

import music_proposal_replay as replay
import train_c161_music_proposals as music
import train_c161_paired_rooms as paired
from fit_c161_history_gain_v1 import DAILY, REF_KERNEL, RESULTS, read, save, sha

room = paired.room
STEM = "4090-c161-proposal-readout"
MODES = ("random", "peak")
FIT = RESULTS/(STEM+"-fit.json")
SEED = 658227
STEPS = 320


def weights(mode):
    return RESULTS/f"{STEM}-{mode}.npz"


def training_audio():
    for r in read(RESULTS/DAILY)["admitted"]:
        req = r["request"]
        assert req["role"] == "training-only" and req["requested_label"] == 0
        if room.BASE.category(req["text"],False) != 4:
            continue
        wave = replay.poll.load_wav(r["wav"])
        yield dict(kind="daily", key=r["index"], text=req["text"], source_sha256=r["wav"]["sha256"]), wave
    history = read(RESULTS/f"{music.STEM}-training-v1.json")
    library = music.MUSIC/"ambient.json"
    assert sha(library) == history["music_index_sha256"]
    admitted = {r["music_path"]: r["music_sha256"] for r in history["records"]}
    inventory = {r["source_path"]: r for r in read(library)}
    assert len(admitted) == 32
    rng = np.random.default_rng(SEED)
    for key, digest in sorted(admitted.items()):
        r = inventory[key]
        path = music.MUSIC/r["audio_path"]
        assert r["split"] == "training" and r["source_sha256"] == digest == sha(path)
        with sf.SoundFile(path) as f:
            assert f.samplerate == 16000 and f.channels == 1
            for offset in rng.choice(len(f)-8*16000+1, 6, replace=False):
                f.seek(int(offset))
                yield dict(kind="music", key=key, offset=int(offset), source_sha256=digest), f.read(8*16000,dtype="int16")


def collect():
    vm = REF_KERNEL(model_path=str(RESULTS/f"{music.STEM}-v1.tflite"))
    vm.allocate_tensors()
    path = replay.BUILD/"frontend.so"
    assert sha(path) == read(replay.BUILD/"build.json")["library_sha256"]
    native = ctypes.CDLL(str(path.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    rng = np.random.default_rng(SEED)
    rows, data = [], {mode:[] for mode in MODES}
    for source, wave in training_audio():
        pcm = np.pad(wave, (16000,16000))
        raw, bands, clock = replay.native_inputs(native,pcm)
        trace = replay.infer(vm,bands)
        sums = np.convolve(trace.astype(np.int32),np.ones(5,np.int32),mode="valid")
        ends = clock[4:]
        eligible = np.flatnonzero((ends>=16000+4*480) & (ends<=16000+len(wave)))
        row = dict(**source, pcm_sha256=replay.poll.digest(wave))
        random = rng.choice(eligible)
        peak = eligible[np.argmax(sums[eligible])]
        for mode,i in zip(MODES,(random,peak),strict=True):
            end = int(ends[i])+replay.scoring.word.WAIT
            patch = room.window(raw,end)
            quantized = np.rint(patch.clip(0,666)*(255./666)).astype(np.uint8)
            data[mode].append(quantized)
            row[mode] = dict(proposer_end=int(ends[i]), window_end=end, five_sum=int(sums[i]),
                             window_sha256=replay.poll.digest(quantized))
        rows.append(row)
    return {k:np.stack(v) for k,v in data.items()},rows


def embeddings(model, patches):
    output = []
    with torch.no_grad():
        for chunk in torch.from_numpy(patches).split(96):
            encoded = model.encoder(chunk.cuda().float().unsqueeze(1)/255).permute(0,1,3,2).flatten(1,2)
            output.append(model.names[:2](model.temporal(encoded)))
    return torch.cat(output)


def train(experiment=None, collect_positives=None):
    experiment = experiment or sys.modules[__name__]
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    front = RESULTS/f"{music.STEM}-v1.tflite"
    assert sha(front) == read(RESULTS/f"{music.STEM}-export-v1.json")["sha256"]
    mined, rows = collect()
    positive_views, positive_records = collect_positives() if collect_positives else ({}, [])
    parent = paired.weights("teacher")
    old = room.BASE.WORK/"features.npz"
    assert sha(old) == read(room.BASE.SOURCES)["cache_sha256"]
    assert sha(room.CACHE) == read(room.SOURCES)["cache_sha256"]
    cfg = read(room.base.FIT)
    net = room.base.load(room.BASE.Small(cfg["alphabet"]), parent, read(paired.FIT)["models"]["teacher"]["weights_sha256"])
    with np.load(old) as z, np.load(room.CACHE) as wet:
        anchors = torch.cat([embeddings(net,w) for w in (z["x"],wet["clean"],wet["room"])])
        labels = np.r_[z["labels"],wet["labels"],wet["labels"]]
    pools = [np.flatnonzero(labels==k) for k in range(5)]
    head = net.names[-1]
    initial = {k:v.detach().clone() for k,v in head.state_dict().items()}
    with torch.no_grad():
        prior = (head(anchors)/2).softmax(dim=-1)
    reports = {}
    for mode in experiment.MODES:
        head.load_state_dict(initial)
        extra = embeddings(net, mined["peak" if positive_views else mode])
        positives = embeddings(net, positive_views[mode]) if positive_views else None
        rng = np.random.default_rng(SEED)
        positive_rng = np.random.default_rng(SEED+1)
        opt = torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.001)
        losses = []
        for step in range(STEPS):
            ids = np.concatenate([rng.choice(g,12) for g in pools])
            x = torch.cat((anchors[ids], extra[rng.choice(len(extra), 16)]))
            y = torch.from_numpy(np.r_[labels[ids], np.full(16,4,np.int64)]).cuda()
            if positives is not None:
                x = torch.cat((x, positives[positive_rng.choice(len(positives),16)]))
                y = torch.cat((y, y.new_zeros(16)))
            logits = head(x)
            loss = F.cross_entropy(logits,y,label_smoothing=.025)
            loss = loss + 4*F.kl_div((logits[:len(ids)]/2).log_softmax(dim=-1),prior[ids],reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0,99,STEPS-1):
                losses.append(float(loss.detach()))
        state = {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
        with np.load(parent) as inherited:
            changed = [k for k,v in state.items() if not np.array_equal(v,inherited[k])]
            assert set(changed) == {"names.2.weight","names.2.bias"}
        with experiment.weights(mode).open("xb") as f:
            np.savez_compressed(f,**state)
        reports[mode] = dict(weights_sha256=sha(experiment.weights(mode)),losses=losses,changed=changed)
    save(experiment.FIT, dict(models=reports,records=rows,positive_records=positive_records,
        positive_policy=getattr(experiment,"POSITIVE_POLICY",None), experiment_sha256=sha(Path(experiment.__file__)),seed=SEED,steps=STEPS,gpu=gpu,torch=torch.__version__,
        parent_weights_sha256=sha(parent),front_sha256=sha(front),script_sha256=sha(Path(__file__)),
        input_files={str(p):sha(p) for p in (paired.FIT,room.SOURCES,room.BASE.SOURCES,old,room.CACHE,RESULTS/DAILY,
                                               RESULTS/f"{music.STEM}-training-v1.json")},
        policy="Same seed, negative clips, anchors and frozen feature encoder. One arm selects a random "
               "eligible endpoint; the other the maximum trailing five-code sum of the frozen INT8 proposer. "
               "Read teacher window 360 ms later. Replay original balanced classes and distill parent on anchors. "
               "Keep C161, proposal threshold, cooldown and teacher decision threshold unchanged."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=("train","export"))
    if p.parse_args().action == "train":
        train()
    else:
        from export_c161_polling_rooms import main
        main(sys.modules[__name__])
