"""Train the 960 ms word proposer on matched music with and without the synthetic target."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

import train_c161_mined_room_word as mining
import train_c161_word_window_v1 as word
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-music-proposal"
PARENT = mining.PREFIX + "-hard"
MUSIC = Path(".local/wakeword/musan-music-features")
SEED = 931014
STEPS = 420


def collect():
    index = MUSIC / "ambient.json"
    prep = read(RESULTS / "4090-musan-music-preparation.json")
    assert prep["all_training_only"] and sha(index) == prep["index_sha256"]
    inventory = read(index)
    rng = np.random.default_rng(SEED)
    tracks = []
    for i in rng.choice(len(inventory), 32, replace=False):
        r = inventory[i]
        path = MUSIC / r["audio_path"]
        assert r["split"] == "training" and sha(path) == r["source_sha256"]
        tracks.append(r)
    paired_negatives = []
    def render(s, dry, onset):
        track = tracks[rng.integers(len(tracks))]
        frames = round(track["seconds"]*16000)
        with sf.SoundFile(MUSIC / track["audio_path"]) as f:
            assert f.samplerate == 16000 and f.channels == 1 and len(f) == frames
            start = int(rng.integers(frames-len(dry)+1))
            f.seek(start)
            music = f.read(len(dry), dtype="float64")
        end = onset+s["end"] if s["positive"] else len(dry)-8000
        speech = dry[onset:end].astype(float)
        mask = np.abs(speech) >= max(1., np.abs(speech).max()*.035)
        assert mask.any()
        snr = int(rng.choice((0, 6, 12)))
        noise_rms = np.sqrt(np.mean(music[onset:end][mask]**2))
        gain = np.sqrt(np.mean(speech[mask]**2))*10**(-snr/20)/max(noise_rms, 1e-6)
        noise = music*gain
        mixture = dry+noise
        scale = min(1., 30000/max(1., np.abs(mixture).max(), np.abs(noise).max()))
        q = np.rint(scale*mixture).astype("<i2")
        if s["positive"]:
            bg = np.rint(scale*noise).astype("<i2")
            raw, clock = word.bands(bg)
            windows = np.lib.stride_tricks.sliding_window_view(raw, word.WIDTH, axis=0)
            ticks = clock[word.WIDTH-1:]
            paired_negatives.extend(windows[(ticks >= end-480) & (ticks <= end+2880)])
        return q, dict(music_path=track["source_path"], music_sha256=track["source_sha256"],
                       offset=start, snr_db=snr, gain=gain, scale=scale)
    x, labels, near, rows = word.training_data(render)
    matched = np.stack(paired_negatives)
    assert len(matched) == np.count_nonzero(labels)
    return x, labels, near, rows, torch.from_numpy(matched).cuda().float()/255


def train():
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    x, y, near, rows, music = collect()
    parent = RESULTS / f"{PARENT}-weights-v1.npz"
    assert sha(parent) == read(RESULTS / f"{PARENT}-training-v1.json")["weights_sha256"]
    model = word.WordWindow().cuda()
    with np.load(parent) as z:
        model.load_state_dict({k:torch.from_numpy(z[k]) for k in z})
    cache = mining.WORK / "negative-banks.npz"
    assert sha(cache) == read(RESULTS / f"{mining.PREFIX}-mining-v1.json")["cache_sha256"]
    with np.load(cache) as z:
        hard = torch.from_numpy(z["hard"]).cuda().float()/255
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(y == 0)
    confusers = np.flatnonzero((y == 0) & near)
    history = []
    opt = torch.optim.AdamW(model.parameters(), lr=.00025, weight_decay=.001)
    for step in range(STEPS):
        ix = rng.choice(len(pos), 64)
        j = np.r_[rng.choice(neg,96), rng.choice(confusers,48)]
        difficult = hard[rng.choice(len(hard),32)].reshape(-1,20,word.WIDTH)
        batch = torch.cat((x[pos[ix]], x[j], music[ix], difficult))
        a, b, music_logits, talk = model(batch)[:,0].split((64,len(j),64,len(difficult)))
        loss = F.softplus(4.5-a).mean()+F.softplus(b+2).mean()+F.softplus(b.topk(24).values+2).mean()
        loss = loss+F.softplus(talk+2).mean()+F.softplus(music_logits+2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,99,STEPS-1):
            history.append(dict(step=step+1, loss=loss.item()))
    weights = RESULTS / f"{STEM}-weights-v1.npz"
    with weights.open("xb") as f:
        np.savez_compressed(f, **{k:v.cpu().numpy() for k,v in model.state_dict().items()})
    cal = RESULTS / f"{STEM}-calibration-v1.npz"
    samples = torch.cat((x[rng.choice(pos,96,replace=False)], x[rng.choice(neg,96,replace=False)],
                         music[rng.choice(len(music),32,replace=False)], hard[rng.choice(len(hard),32,replace=False),2]))
    with torch.no_grad(), cal.open("xb") as f:
        np.savez_compressed(f, inputs=(samples.cpu().numpy()*255).round().astype(np.uint8),
                            probabilities=model(samples).sigmoid()[:,0].cpu().numpy())
    save(RESULTS / f"{STEM}-training-v1.json", dict(gpu=gpu, seed=SEED, steps=STEPS, history=history,
         weights_sha256=sha(weights), calibration_sha256=sha(cal), parent_sha256=sha(parent),
         negative_bank_sha256=sha(cache), music_index_sha256=sha(MUSIC/"ambient.json"), music_windows=len(music),
         script_sha256=sha(Path(__file__)), records=rows, policy="Warm-start the hard-negative word proposer. Mix each existing synthetic training utterance with a training-only MUSAN music crop at 0, 6 or 12 dB. Every positive training tick is paired with the same isolated music at the same gain and time. Reuse five-frame speech hard negatives. Keep thresholds, 360 ms wait and teacher verifier frozen."))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["export"]:
        import export_c161_word_window_v1 as exporter
        exporter.STEM = STEM
        exporter.main()
    else:
        train()
