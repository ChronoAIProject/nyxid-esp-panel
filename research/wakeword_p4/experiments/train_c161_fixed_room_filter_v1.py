"""Fit one causal integer spectrotemporal filter to playback of the frozen v3 WAV."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt
import torch

from fit_c161_history_gain_v1 import DAILY, PRIME, RESULTS, read, save, sha
from review_c161_equal_length_history_v1 import features
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_midlevel_v1 import sources

STEM = "4090-c161-fixed-room-filter"
WIDTH = 32
SEED = 67801


def frames(pcm):
    full = np.r_[np.zeros(PRIME, np.int16), pcm]
    raw = features(full)
    raw = np.pad(raw, ((0, -len(raw)%3), (0,0)))
    # Six frontend values per cell; division by 16 fits a byte.
    data = (np.minimum(raw, 665).reshape(-1,3,20,2).sum((1,3))//16).astype(np.uint8)
    clock = np.minimum((np.arange(len(data))+2)*480, len(full))-PRIME
    return data,clock


def windows(data):
    return np.lib.stride_tricks.sliding_window_view(data, WIDTH, axis=0).transpose(0,2,1).reshape(-1,WIDTH*20)


def cue():
    ref = read(RESULTS / "4090-c161-suffix-cue-v3-source.json")["wav_sha256"]
    return load_wav(dict(path=RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav",sha256=ref)),ref


def recordings():
    wave,_ = cue()
    entries = read(RESULTS/DAILY)["admitted"]
    rng = np.random.default_rng(SEED)
    for i in range(320):
        row = entries[i%len(entries)]
        assert row["request"]["role"] == "training-only"
        prefix = load_wav(row["wav"])
        prefix = prefix[-24000:] if i%3 else np.zeros(0,np.int16)
        gap = int(rng.integers(0,12001))
        onset = len(prefix)+gap
        ratio = rng.uniform(.984,1.016)
        source = np.interp(np.arange(round(len(wave)/ratio))*ratio,np.arange(len(wave)),wave)
        ir = np.zeros(3201)
        ir[0] = rng.uniform(.35,1)
        lags = rng.choice(np.arange(48,len(ir)),24,replace=False)
        ir[lags] = rng.normal(0,.29,len(lags))*np.exp(-lags/rng.uniform(700,2200))
        level = rng.uniform(.22,1.1)
        wet = fftconvolve(source,ir)*level
        cut = (rng.uniform(55,380),rng.uniform(3100,7300))
        wet = sosfilt(butter(2,cut,fs=16000,btype="bandpass",output="sos"),wet)
        full = np.r_[prefix,np.zeros(gap),wet,np.zeros(8000)]
        noise = lfilter([1],[1,-rng.uniform(.3,.95)],rng.normal(size=len(full)))
        snr = rng.uniform(3,30)
        full += noise * np.sqrt(np.mean(wet**2)/np.mean(noise**2)) * 10**(-snr/20)
        full *= min(1, 31000/max(1,np.abs(full).max()))
        pcm = np.rint(full).astype("<i2")
        yield dict(index=i,history=row["index"],onset=onset,speed=ratio,snr_db=snr), pcm
    for row,pcm in sources():
        if not row["positive"]:
            yield dict(key=row["key"],voice=row["voice"],text=row["text"]), np.r_[pcm,np.zeros(8000,np.int16)]


def train():
    torch.set_num_threads(2)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    rng = np.random.default_rng(SEED)
    pos,neg,rows = [],[],[]
    for meta,pcm in recordings():
        data,clock = frames(pcm)
        x = windows(data)
        end = clock[WIDTH-1:]
        if "onset" in meta:
            rel = end-meta["onset"]
            pos.extend(x[(rel>=12480)&(rel<=16800)])
            negative = (end>0)&((rel<4800)|(rel>26880))
        else:
            negative = end>0
        ids = np.flatnonzero(negative)
        neg.extend(x[rng.choice(ids,min(48,len(ids)),replace=False)])
        rows.append(dict(**meta,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
    positive = torch.from_numpy(np.stack(pos)).cuda().float()/128
    negative = torch.from_numpy(np.stack(neg)).cuda().float()/128
    torch.manual_seed(SEED)
    weight = torch.nn.Parameter(torch.zeros(WIDTH*20,device="cuda"))
    bias = torch.nn.Parameter(torch.tensor(-3.,device="cuda"))
    optimizer = torch.optim.Adam([weight,bias],lr=.015)
    for step in range(1400):
        qw = (weight*64).round().clamp(-127,127)/64
        w = weight+(qw-weight).detach()
        positive_scores = positive[rng.choice(len(pos),64)]@w+bias
        negative_scores = negative[rng.choice(len(neg),512)]@w+bias
        loss = torch.nn.functional.softplus(6-positive_scores).mean()
        loss += torch.nn.functional.softplus(negative_scores.topk(32).values+4).mean() + .015 * weight.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    coeff = np.rint(weight.detach().cpu().numpy()*64).clip(-127,127).astype(np.int8)
    b = round(bias.item()*8192)
    lut = np.rint(256/(1+np.exp(-np.arange(-128,128)/16))).clip(0,255).astype(np.uint8)
    path = RESULTS/f"{STEM}-weights-v1.npz"
    with path.open("xb") as f:
        np.savez_compressed(f,weight=coeff,bias=np.int32(b),probability=lut)
    save(RESULTS/f"{STEM}-training-v1.json",dict(gpu=gpu,source_sha256=cue()[1],seed=SEED,steps=1400,
         weights_sha256=sha(path),positive_windows=len(pos),negative_windows=len(neg),records=rows))


def scores(pcm,weights):
    data,clock = frames(pcm)
    x = windows(np.pad(data,((WIDTH-1,0),(0,0))))
    # Round signed logit to the nearest 1/16; exact half steps go toward +infinity.
    acc = x.astype(np.int32)@weights["weight"].astype(np.int32) + int(weights["bias"])
    return weights["probability"][np.clip((acc+256)//512,-128,127)+128],clock


if __name__ == "__main__":
    train()
