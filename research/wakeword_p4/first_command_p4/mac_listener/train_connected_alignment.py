"""Compare fixed and strongest deliverable five-tick training spans for the same connected calls."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve
import torch
from torch.nn import functional as F

import calibrate_unmarked_dwell as calibration

ch, fit = calibration.ch, calibration.fit
STEM = "4090-c161-connected-alignment"
ARMS = ("fixed", "latent")
PARENT = calibration.replay.STEM + "-command"
SEED = 331497


def file(stem, kind):
    return ch.RESULTS / f"{stem}-{kind}-v1.npz"


def microphones():
    doc = ch.read(calibration.OUT)
    assert ch.sha(calibration.WAVES) == doc["waves_sha256"]
    sources = {s["key"]: s for s in doc["sources"]}
    beds, songs, rooms, demos = fit.acoustics()
    impulse = dict(rooms)
    panel = {r[0]: demo for r, demo in zip(rooms, demos, strict=True)}
    music = {s["sha256"]: bed for s, bed in zip(songs, beds, strict=True)}
    with np.load(calibration.WAVES, allow_pickle=False) as waves:
        for r in doc["records"]:
            s = sources[r["source_key"]]
            pcm = waves[s["key"]]
            assert ch.digest(pcm) == s["pcm_sha256"]
            boundary = calibration.BOUNDARY
            dry = np.pad(pcm, (boundary-s["command_at"], 16000))
            if r["kind"] == "no_address":
                dry[:boundary] = 0
            mic = dry
            if r["condition"] == "mixed":
                scene = s["scene"]
                wet = fftconvolve(dry.astype(float), impulse[scene["external_room"]])
                demo = panel[scene["panel_room"]]
                at = scene["panel_at"]
                length = max(len(wet), at+len(demo))
                bed = music[scene["music_sha256"]]
                background = bed[(scene["offset"]+np.arange(length)) % len(bed)] * scene["music_gain"]
                background[at:at+len(demo)] += demo * scene["panel_gain"]
                mic = np.rint((background + np.pad(wet, (0,length-len(wet)))) * r["gain"]).astype("<i2")
            assert ch.digest(mic) == r["mic_sha256"]
            yield r, mic


def initialized():
    net = fit.prior.music.word.WordWindow().cuda()
    path = file(PARENT, "weights")
    assert ch.sha(path) == ch.read(ch.RESULTS / (PARENT+"-training-v1.json"))["weights_sha256"]
    with np.load(path) as z:
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z})
    return net


@torch.no_grad()
def collect():
    native = fit.prior.letters.native_frontend()
    teacher = initialized().eval()
    pos, neg, rows = [], [], []
    for r, mic in microphones():
        _, bands, clock = fit.prior.base.native_inputs(native, mic)
        windows = np.lib.stride_tricks.sliding_window_view(bands, 32, axis=0)
        delivery = (clock[31:]+255)//256*256
        if r["kind"] == "target":
            end = int(np.searchsorted(delivery, calibration.BOUNDARY, side="left"))
            pick = slice(end-11, end)
            assert delivery[end-5] >= r["wake_start"] and delivery[end-1] < calibration.BOUNDARY
            pos.append(windows[pick].copy())
        else:
            first = int(np.searchsorted(delivery, 0, side="right"))
            x = torch.from_numpy(windows[first:].copy()).cuda().float()/255
            scores = teacher(x).flatten().sigmoid().unfold(0, 5, 1).mean(-1)
            start = first + int(scores.argmax())
            pick = slice(start, start+5)
            neg.append(windows[pick].copy())
        rows.append(dict(key=r["key"], voice=r["voice"], condition=r["condition"], kind=r["kind"],
                         pcm_sha256=r["mic_sha256"], ticks=delivery[pick].tolist(), bands_sha256=ch.digest(windows[pick].copy())))
    assert len(pos) == 96 and len(neg) == 384
    return torch.tensor(np.stack(pos), device="cuda").float()/255, torch.tensor(np.stack(neg), device="cuda").float()/255, rows


@torch.no_grad()
def negative_streams(rows):
    native = fit.prior.letters.native_frontend()
    streams = []
    for r, pcm in microphones():
        if r["kind"] == "target":
            continue
        _, bands, clock = fit.prior.base.native_inputs(native, pcm)
        patches = np.lib.stride_tricks.sliding_window_view(bands, 32, axis=0)
        delivery = (clock[31:]+255)//256*256
        first = int(np.searchsorted(delivery, 0, side="right"))
        assert r["key"] == rows[len(streams)]["key"]
        streams.append((torch.from_numpy(patches[first:].copy()).cuda().float()/255, delivery[first:]))
    assert len(streams) == len(rows)
    return streams


@torch.no_grad()
def refresh_negatives(model, streams):
    groups, ends = [], []
    for windows, delivery in streams:
        score = model(windows).flatten().sigmoid().unfold(0,5,1).mean(-1)
        start = int(score.argmax())
        groups.append(windows[start:start+5])
        ends.append(int(delivery[start+4]))
    return torch.stack(groups), ends


def train(refresh=False):
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pos, neg, rows = collect()
    source = ch.RESULTS / (STEM+"-sources.json")
    mining = fit.prior.music.mining.WORK / "negative-banks.npz"
    assert ch.sha(mining) == ch.read(ch.RESULTS/(fit.prior.music.mining.PREFIX+"-mining-v1.json"))["cache_sha256"]
    representative = file(PARENT, "calibration")
    assert ch.sha(representative) == ch.read(ch.RESULTS/(PARENT+"-training-v1.json"))["calibration_sha256"]
    with np.load(mining) as z, np.load(representative) as cal:
        hard = torch.tensor(z["hard"], device="cuda").float()/255
        rehearsal = torch.tensor(cal["inputs"], device="cuda").float()/255
    assert len(rehearsal) == 256
    source_data = dict(records=rows, calibration_sha256=ch.sha(calibration.OUT), waves_sha256=ch.sha(calibration.WAVES),
        parent_weights_sha256=ch.sha(file(PARENT,"weights")), hard_sha256=ch.sha(mining), rehearsal_sha256=ch.sha(representative),
        script_sha256=ch.sha(Path(__file__)), seed=SEED)
    if refresh:
        original = ch.read(source)
        assert all(source_data[k] == original[k] for k in source_data if k != "script_sha256")
        negative_rows = [r for r in rows if r["kind"] != "target"]
        streams = negative_streams(negative_rows)
    else:
        ch.save(source, source_data)
    for arm in (("refreshed",) if refresh else ARMS):
        model = initialized()
        opt = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.001)
        rng = np.random.default_rng(SEED)
        history, refreshes = [], []
        for step in range(1, 301):
            if refresh and step in (51, 101, 151, 201, 251):
                neg, ends = refresh_negatives(model, streams)
                refreshes.append(dict(step=step, last_delivery=ends))
            target = pos[rng.choice(len(pos), 32)]
            other = torch.cat((neg[rng.choice(len(neg), 64)], hard[rng.choice(len(hard), 24)]))
            positive = model(target.flatten(0,1)).view(32,11).unfold(1,5,1)
            best = positive.detach().sigmoid().mean(-1).argmax(-1)
            aligned = positive[:, -1] if arm == "fixed" else positive[torch.arange(32,device="cuda"),best]
            negative = model(other.flatten(0,1)).view(-1,5)
            anchor = model(rehearsal[np.r_[rng.choice(64,16),64+rng.choice(192,48)]]).flatten()
            loss = (F.softplus(4.5-aligned).mean() + F.softplus(negative+2).mean()
                    + F.softplus(negative.mean(-1).topk(16).values+2).mean()
                    + .5*(F.softplus(4.5-anchor[:16]).mean()+F.softplus(anchor[16:]+2).mean()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (1, 100, 200, 300):
                history.append(dict(step=step, loss=loss.item(), strongest_end_offsets=(best-6).tolist()))
        stem = STEM+"-"+arm
        with TemporaryDirectory(dir=ch.RESULTS) as tmp:
            work = Path(tmp)
            weights = work / file(stem,"weights").name
            cal_path = work / file(stem,"calibration").name
            np.savez_compressed(weights, **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            with torch.no_grad():
                np.savez_compressed(cal_path, inputs=(rehearsal*255).round().byte().cpu().numpy(),
                                    probabilities=model(rehearsal).sigmoid().flatten().cpu().numpy())
            ch.save(work/(stem+"-training-v1.json"),dict(arm=arm, steps=300, seed=SEED, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, history=history, sources_sha256=ch.sha(source), weights_sha256=ch.sha(weights),
                calibration_sha256=ch.sha(cal_path), positive_bags=len(pos), negative_bags=len(neg), hard_bags=len(hard),
                **(dict(refreshes=refreshes, trainer_sha256=ch.sha(Path(__file__))) if refresh else {})))
            for path in work.iterdir():
                path.replace(ch.RESULTS/path.name)
        print(arm, history[-1]["loss"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train","export"))
    parser.add_argument("--refresh", action="store_true", help="Fine-tune the latent arm with five negative-window refreshes")
    args = parser.parse_args()
    if args.stage == "train":
        train(args.refresh)
    else:
        import export_c161_word_window_v1 as exporter
        for arm in (("refreshed",) if args.refresh else ARMS):
            exporter.STEM = STEM+"-"+arm
            exporter.main()
