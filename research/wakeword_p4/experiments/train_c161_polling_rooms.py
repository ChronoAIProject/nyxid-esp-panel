"""Adapt the five-way onset readout to room audio and uncertainty in window placement."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

import train_c161_onset_evidence as base
from fit_c161_history_gain_v1 import PRIME, RESULTS, audio, features, read, save, sha
from train_c161_room_word import room_render
from check_c161_v5_command_ns import digest

STEM = "4090-c161-polling-rooms"
WORK = Path(".local/wakeword/c161-polling-rooms")
CACHE = WORK/"windows.npz"
SOURCES = RESULTS/(STEM+"-sources.json")
FIT = RESULTS/(STEM+"-fit.json")
SEED = 935841
STEPS = 450
CADENCE = 1920
MODES = ("clean", "room")
BASE = base.base


def raw_features(pcm):
    raw = features(np.pad(pcm,(PRIME,0)))
    return np.rint(raw.clip(0,666)*(255./666)).astype(np.uint8)


def window(raw,end):
    assert end%160 == 0
    last = (end+PRIME)//160-2
    result = raw[last-144:last]
    assert result.shape == (144,40)
    return result


def materials():
    rng = np.random.default_rng(SEED)
    pools = [[] for _ in range(5)]
    for r in read(BASE.INVENTORY)["inventory"]:
        s = r["source"]
        assert s["role"] == "training"
        lab = BASE.category(s["text"],s["label"])
        pools[lab].append(s)
    for lab,limit in enumerate((320,160,160,0,320)):
        for i in rng.choice(len(pools[lab]),limit,replace=False):
            s = pools[lab][i]
            yield s["id"],s["text"],lab,audio(dict(path=s["native_path"],sha256=s["native_sha256"]))
    for r in read(RESULTS/BASE.onset.DAILY)["admitted"]:
        req = r["request"]
        assert req["role"]=="training-only" and req["requested_label"]==0
        yield f"daily-{r['index']}",req["text"],BASE.category(req["text"],False),audio(r["wav"])
    onset = BASE.onset
    with patch.multiple(onset,VOICES=("af_alloy","am_eric","bf_lily","am_fenrir"),
                        ADDRESSES=onset.ADDRESSES+("Hey Next","Hi Next"),LABELS=onset.LABELS+(4,4)):
        for i,(r,pcm) in enumerate(onset.synthesize()):
            yield f"kokoro-{i}",r["text"],BASE.category(r["text"],r["label"]==0),pcm


def collect():
    WORK.mkdir(exist_ok=True)
    render = room_render()
    rows, views, labels = [], {m:[] for m in MODES}, []
    for key,text,lab,pcm in materials():
        start = 32000
        dry = np.pad(pcm,(start,32000))
        wet, source = render(dict(key="five-class-"+key),dry,start)
        # Both treatments see the same decision clocks, command and label. First speech
        # onset in dry audio supplies training alignment only. Runtime polls autonomously.
        onsets = BASE.onset.windows(dry)
        first = next(onsets,None)
        if first is None:
            rows.append(dict(key=key,text=text,label=lab,pcm_sha256=digest(pcm),skipped="dry VAD found no speech"))
            continue
        anchor = first[0]+BASE.onset.WAIT
        ends = [anchor-1920,anchor,anchor+1920]
        assert ends[-1]<=len(dry)
        # Use the same pre-speech background and post-utterance silence for both labels.
        late = (len(dry)//160)*160
        ends.extend([start-480,late])
        for mode,stream in zip(MODES,(dry,wet),strict=True):
            sequence = raw_features(stream)
            views[mode].extend(window(sequence,t) for t in ends)
        labels.extend([lab,lab,lab,4,4])
        rows.append(dict(key=key,text=text,label=lab,pcm_sha256=digest(pcm),room_pcm_sha256=digest(wet),
                         source=source,ends=ends))
    with CACHE.open("xb") as f:
        np.savez_compressed(f,**{k:np.stack(v) for k,v in views.items()},labels=np.array(labels))
    save(SOURCES,dict(records=rows,cache_sha256=sha(CACHE),script=sha(Path(__file__)),
         classes=BASE.CLASSES,files={str(p):sha(p) for p in (BASE.INVENTORY,RESULTS/BASE.onset.DAILY)},
         policy="Matched 120 ms early, anchored, 120 ms late. Wake-free prefix and last 1.44 s of 2 s padded tail. Training room and noise partitions only; kernel and sampling code from train_c161_room_word."))


def weights(mode):
    return RESULTS/(STEM+"-"+mode+".npz")


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    cfg = read(base.FIT)
    old = BASE.WORK/"features.npz"
    assert sha(CACHE)==read(SOURCES)["cache_sha256"] and sha(old)==read(BASE.SOURCES)["cache_sha256"]
    with np.load(old) as z:
        original = torch.from_numpy(z["x"]).cuda()
        y0 = z["labels"]
    with np.load(CACHE) as z:
        y1 = z["labels"]
        tensors = {m:torch.from_numpy(z[m]).cuda() for m in MODES}
    groups = [[np.flatnonzero(y==k) for k in range(5)] for y in (y0,y1)]
    reports = {}
    for m in MODES:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        net = base.load(BASE.Small(cfg["alphabet"]),base.weight_file("posterior"),
                        cfg["models"]["posterior"]["weights_sha256"]).train()
        opt = torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.004)
        history = []
        for step in range(STEPS):
            idx0,idx1 = [np.concatenate([rng.choice(ids,8) for ids in g]) for g in groups]
            x = torch.cat((original[idx0],tensors[m][idx1])).float().unsqueeze(1)/255
            y = torch.from_numpy(np.r_[y0[idx0],y1[idx1]]).cuda()
            logits,_ = net(x)
            loss = F.cross_entropy(logits,y,label_smoothing=.025)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0,99,STEPS-1):
                history.append(float(loss.detach()))
        net.eval()
        with weights(m).open("xb") as f:
            np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in net.state_dict().items()})
        reports[m]=dict(loss=history,weights_sha256=sha(weights(m)))
        print(m,history,flush=True)
    save(FIT,dict(models=reports,seed=SEED,steps=STEPS,gpu=gpu,torch=torch.__version__,parent=sha(base.FIT),
         parent_weight=sha(base.weight_file("posterior")),sources_sha256=sha(SOURCES),code_sha256=sha(Path(__file__)),
         original_class_counts=dict(Counter(map(int,y0))),new_class_counts=dict(Counter(map(int,y1))),
         decisions="Same initial weights, batches and spacing for both fits. Original VAD and fixed 120 ms cadence with two consecutive Nyx >=0.5 decisions. Union with C161 at original cooldown."))


if __name__=="__main__":
    import sys
    {"collect":collect,"train":train}[sys.argv[1]]()
