"""Optimize three emitted markers for the frozen v5 correlator through measured training RIRs."""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

import coded_fixed_advance as fixed
from diagnose_c161_v5_room_misses import coefficients, COEFF
from select_room_spoken_cues import noise_training

ch = fixed.ch
STEM = "4090-c161-room-trained-cue"
WAV = ch.RESULTS / "hye-nyx-v5-room-trained.wav"
FIT = ch.RESULTS / (STEM + "-fit.json")
AUDIT = ch.RESULTS / "4090-mit-rir-audit.json"
EVAL = (1,3,4,5,7,9,10,11,12,13,18,19,24,25)
SEED = 191126
N = 1280


def original():
    meta = ch.read(fixed.SOURCE_INFO)
    assert ch.sha(fixed.CUE) == meta["wav_sha256"]
    source, rate = sf.read(fixed.SOURCE, dtype="int16")
    pcm, other = sf.read(fixed.CUE, dtype="int16")
    assert rate == other == 16000 and len(pcm) == meta["samples"] and meta["code_length"] == N
    starts = meta["code_positions"]
    base = source[:len(pcm)].astype("int32")
    delta = pcm.astype("int32") - base
    codes = np.stack([delta[i:i+N] for i in starts])
    for i in starts:
        delta[i:i+N] = 0
    assert not delta.any()
    return meta, pcm, base, codes


def train():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    meta, pcm, base, codes = original()
    lens = len(pcm)
    starts = meta["code_positions"]
    regions = torch.tensor(starts, device="cuda")[:,None] + torch.arange(N,device="cuda")
    def tensor(x):
        return torch.as_tensor(x,dtype=torch.float32,device="cuda")
    voiced = tensor(base/32768)
    frozen = tensor(codes/32768)
    rooms = [r for r in ch.read(AUDIT)["records"] if r["split"]=="training"
             and int(Path(r["path"]).name[1:4]) not in EVAL]
    subset = [rooms[i] for i in rng.choice(len(rooms),48,replace=False)]
    hs, peaks = [], []
    for r in subset:
        name = Path(r["path"]).name
        h, path, digest = ch.read_room(name[:5])
        assert (path,digest) == (r["path"],r["sha256"])
        hs.append(h)
        peaks.append(int(np.abs(h).argmax()))
    fft = 1 << (lens+max(map(len,hs))-1).bit_length()
    bank = torch.stack([torch.fft.rfft(tensor(h),fft) for h in hs])
    noise, noises = [], []
    for n in ("AirConditioner_4.wav","VacuumCleaner_3.wav"):
        bg, digest = noise_training(n)
        noises.append(dict(file=n,sha256=digest))
        for _ in range(16):
            at = int(rng.integers(len(bg)-fft+1))
            z = bg[at:at+fft]
            noise.append(z / np.sqrt(np.mean(z*z)))
            noises[-1].setdefault("offsets",[]).append(at)
    noise = tensor(np.stack(noise))
    fir = tensor(coefficients("kCueBandFirV5")[::-1].copy())[None,None]/128
    pats = tensor(coefficients("kCueBandPatternsV5").reshape(3,320))
    pats = pats / pats.norm(dim=1,keepdim=True)
    peak = torch.tensor(peaks,device="cuda") // 4
    centers = torch.tensor(starts,device="cuda") // 4
    # Allow one coherent arrival within 32 ms either side of the strongest RIR tap.
    first = centers[None,:,None] + peak[:,None,None] + torch.arange(-128,129,device="cuda")
    first = first.clamp_min(0)
    delta = torch.nn.Parameter(torch.zeros_like(frozen))
    optimizer = torch.optim.Adam([delta],lr=.02)
    hz = torch.fft.rfftfreq(N,1/16000,device="cuda")
    band = (hz>=700) & (hz<=1580)
    taper = torch.hann_window(160,periodic=False,device="cuda")
    envelope = torch.ones(N,device="cuda")
    envelope[:80],envelope[-80:] = taper[:80],taper[-80:]
    def synth():
        perturb = torch.fft.irfft(torch.fft.rfft(delta)*band,n=N) * envelope
        ratio = perturb.norm(dim=1,keepdim=True)/frozen.norm(dim=1,keepdim=True)
        shifted = frozen + perturb / (ratio/.50).clamp_min(1)
        shifted = shifted * (frozen.norm(dim=1,keepdim=True)/shifted.norm(dim=1,keepdim=True))
        out = voiced.clone()
        out[regions] += shifted
        return out
    logs=[]
    for step in range(301):
        idx = torch.randint(len(hs),(10,),device="cuda")
        clean = torch.fft.irfft(torch.fft.rfft(synth(),fft)*bank[idx],n=fft)
        power = clean[:,:lens].square().mean(1,keepdim=True).sqrt().detach()
        gain = torch.rand(len(idx),1,device="cuda")*.9
        mic = clean + noise[torch.randint(len(noise),(len(idx),),device="cuda")]*power*gain
        phase = step % 4
        down = mic[:,phase:phase + (fft//4-1)*4].reshape(len(idx),-1,4).mean(-1)
        filtered = F.conv1d(F.pad(down[:,None],(16,0)),fir)[:,0]
        frames = filtered.unfold(-1,320,1)
        endpoints = first[idx]
        windows = frames[torch.arange(len(idx),device="cuda")[:,None,None],endpoints,:]
        scores = (windows * pats[None,:,None,:]).sum(-1)/windows.norm(dim=-1).clamp_min(1e-7)
        positive = scores.amin(1).amax(-1)
        negative = (-scores).amin(1).amax(-1)
        quality = torch.maximum(positive,negative)
        loss = F.softplus((.48-quality)*10).mean()/10
        if step%50==0:
            logs.append(dict(step=step,mean_correlation=quality.mean().item(),loss=loss.item()))
        if step<300:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    candidate = (32768*synth()).detach().cpu().numpy().round().astype("int32")
    assert abs(candidate).max() <= 32767
    candidate = candidate.astype("<i2")
    outside = np.ones(lens,bool)
    diagnostics=[]
    for i,start in enumerate(starts):
        outside[start:start+N]=False
        c = candidate[start:start+N]-base[start:start+N]
        diagnostics.append(dict(start=start,relative_code_rms=float(np.linalg.norm(c)/np.linalg.norm(codes[i])),
                                delta_rms_ratio=float(np.linalg.norm(c-codes[i])/np.linalg.norm(codes[i]))))
    np.testing.assert_array_equal(candidate[outside],pcm[outside])
    assert not WAV.exists() and not FIT.exists()
    sf.write(WAV,candidate,16000,subtype="PCM_16")
    ch.save(FIT,dict(training_rooms=subset,development_room_ids=EVAL,noise=noises,seed=SEED,step=300,
        code_diagnostics=diagnostics,training=logs,gpu=torch.cuda.get_device_name(),pcm_sha256=ch.digest(candidate),
        wav_sha256=ch.sha(WAV),files={p.name:ch.sha(p) for p in (COEFF,fixed.CUE,fixed.SOURCE,AUDIT,Path(__file__))},
        policy="Only the three emitted 80 ms regions may change. Match each original marker energy, band-limit "
        "and taper the additive correction. Frozen P4 coefficients, no detector retraining. Same-polarity "
        "minimum of three normalized correlations at one delay, through 48 training RIRs with additive training "
        "background noise and four downsampling phases. Final iterate, no development selection."))


if __name__ == "__main__":
    train()
