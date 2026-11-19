"""Fine-tune the word proposer on five-frame events before two fixed training deadlines."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

import train_proposal_letters as letters
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base, music = letters.base, letters.parent.music
STEM = "4090-c161-proposal-deadline"
MODES = {"tail": 0, "early": 2880}
SEED = 41951
STEPS = 300


def collect():
    parent = RESULTS / f"{music.STEM}-training-v1.json"
    sources = {r["key"]:r for r in read(parent)["records"]}
    native, render = letters.native_frontend(), music.mining.room_render()
    rows, pos, neg = [], [], []
    for src, wave in music.word.sources():
        original = sources.pop(src["key"])
        assert base.poll.digest(wave) == original["pcm_sha256"] and original["positive"] == src["positive"]
        onset = 32000
        dry = np.pad(wave,(onset,16000))
        wet, setting = render(src,dry,onset)
        for condition, pcm in (("dry",dry),("room",wet)):
            if condition == "room" and np.array_equal(wet,dry):
                continue
            _, bands, clock = base.native_inputs(native,pcm)
            width = music.word.WIDTH
            patches = np.lib.stride_tricks.sliding_window_view(bands,width,axis=0)
            ends = clock[width-1:]
            row = dict(key=src["key"], positive=src["positive"], voice=src["voice"], text=src["text"],
                       pcm_sha256=base.poll.digest(pcm), source_pcm_sha256=original["pcm_sha256"],
                       condition=condition, acoustic=setting if condition == "room" else None)
            if src["positive"]:
                tail = int(np.searchsorted(ends,onset+src["end"],side="right")-1)
                assert ends[tail] <= onset+src["end"] < ends[tail+1] and ends[tail-17]>=onset
                pos.append(patches[tail-17:tail+1].copy())
                row.update(tail=int(onset+src["end"]), first=int(ends[tail-17]), last=int(ends[tail]))
            else:
                eligible = (ends >= onset) & (ends <= onset+len(wave))
                assert eligible.sum() >= 5
                # Keep all spans of this named or ordinary negative until choosing the
                # strongest five-frame parent event, avoiding isolated easy frames.
                neg.append(patches[eligible].copy())
                row.update(first=int(ends[eligible][0]), last=int(ends[eligible][-1]))
            rows.append(row)
    assert not sources
    return pos, neg, rows


def parent_model():
    file = RESULTS / f"{music.STEM}-weights-v1.npz"
    assert sha(file) == read(RESULTS / f"{music.STEM}-training-v1.json")["weights_sha256"]
    net = music.word.WordWindow().cuda()
    with np.load(file) as weights:
        net.load_state_dict({k:torch.from_numpy(weights[k]) for k in weights})
    return net


def train():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    pos, neg, rows = collect()
    teacher = parent_model().eval()
    negatives = []
    with torch.no_grad():
        for xs, row in zip(neg,(r for r in rows if not r["positive"]),strict=True):
            prob = teacher(torch.from_numpy(xs).cuda().float()/255).sigmoid().flatten()
            i = int(prob.unfold(0,5,1).mean(1).argmax())
            negatives.append(xs[i:i+5])
            row["parent_peak_start"] = row["first"]+480*i
    mining = music.mining.WORK / "negative-banks.npz"
    assert sha(mining) == read(RESULTS/f"{music.mining.PREFIX}-mining-v1.json")["cache_sha256"]
    with np.load(mining) as bank:
        speech = torch.from_numpy(bank["hard"]).cuda().float()/255
    p = torch.from_numpy(np.stack(pos)).cuda().float()/255
    n = torch.from_numpy(np.stack(negatives)).cuda().float()/255
    inventory = RESULTS/(STEM+"-sources.json")
    save(inventory,dict(records=rows, parent_sha256=sha(RESULTS/f"{music.STEM}-weights-v1.npz"),
         background_bank_sha256=sha(mining),script_sha256=sha(Path(__file__))))
    for mode, advance in MODES.items():
        name = STEM+"-"+mode
        model = parent_model()
        rng = np.random.default_rng(SEED)
        opt = torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.001)
        history = []
        for step in range(STEPS):
            # Each positive bag has twelve successive windows, hence eight possible
            # five-tick releases. The two arms differ only by six 30 ms ticks.
            start = 6-advance//480
            targets = p[rng.choice(len(p),32),start:start+12]
            negatives = torch.cat((n[rng.choice(len(n),48)],speech[rng.choice(len(speech),24)]))
            batch = torch.cat((targets.flatten(0,1),negatives.flatten(0,1)))
            lp, ln = model(batch).flatten().split([32*12,len(negatives)*5])
            best = lp.reshape(32,12).sigmoid().unfold(1,5,1).mean(2).amax(1)
            loss = -torch.log(best.clamp_min(1.e-7)).mean() + F.relu(.97-best).mean()*4
            loss = loss + F.softplus(ln+2).mean() + F.softplus(ln.reshape(-1,5).mean(1).topk(12).values+2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0,99,STEPS-1):
                history.append(dict(step=step+1,loss=float(loss.detach()),positive_mean=float(best.detach().mean())))
        weights = RESULTS/f"{name}-weights-v1.npz"
        with weights.open("xb") as f:
            np.savez_compressed(f,**{k:v.cpu().numpy() for k,v in model.state_dict().items()})
        samples = torch.cat((p[rng.choice(len(p),96,replace=False),rng.integers(0,18,96)],
                             n[rng.choice(len(n),96,replace=False),rng.integers(0,5,96)],
                             speech[rng.choice(len(speech),64,replace=False),rng.integers(0,5,64)]))
        cal = RESULTS/f"{name}-calibration-v1.npz"
        with torch.no_grad(), cal.open("xb") as f:
            np.savez_compressed(f,inputs=(samples.cpu().numpy()*255).round().astype(np.uint8),
                                probabilities=model(samples).sigmoid().flatten().cpu().numpy())
        save(RESULTS/f"{name}-training-v1.json",dict(seed=SEED,steps=STEPS,advance_samples=advance,
             history=history,gpu=torch.cuda.get_device_name(),sources_sha256=sha(inventory),weights_sha256=sha(weights),
             calibration_sha256=sha(cal),script_sha256=sha(Path(__file__)),positive_views=len(p),
             negative_views=len(n),speech_bags=len(speech),
             policy="Same original music proposer, optimizer, seed, fixed 300 updates and training samples in both arms. "
                    "Maximize the best five adjacent sigmoid scores averaged within a 12-window positive bag, "
                    "ending at the annotated keyword tail or 180 ms earlier. Reject original parent peak negative "
                    "spans plus frozen mined training speech. All parameters train; identical calibration windows."))


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    elif sys.argv[1:] == ["export"]:
        import export_c161_word_window_v1 as exporter
        for mode in MODES:
            exporter.STEM = STEM+"-"+mode
            exporter.main()
    else:
        raise SystemExit("train or export")
