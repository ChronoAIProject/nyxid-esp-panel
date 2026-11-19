"""Fit a causal spectral mask to training-only synthetic speech, music and room responses."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import torch
from torch import nn
from torch.nn import functional as F

import select_room_spoken_cues as selection
from train_c161_music_proposals import MUSIC

ch = selection.ch
STEM = "4090-c161-causal-speech-mask"
FIT = ch.RESULTS / (STEM + "-fit.json")
WEIGHTS = ch.RESULTS / (STEM + "-weights.npz")
SEED = 19112026
HOP = 160
STEPS = 1200
BATCH = 32
SEGMENT = 64 * HOP


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Linear(161, 48)
        self.memory = nn.GRU(48, 48, batch_first=True)
        self.out = nn.Linear(48, 161)

    def forward(self, spectrum, state=None):
        features = torch.log1p(4 * spectrum.abs())
        hidden, state = self.memory(F.relu(self.encode(features)), state)
        return torch.sigmoid(self.out(hidden)), state


def spectrum(x, window):
    return torch.fft.rfft(x.unfold(-1, 2*HOP, HOP) * window)


def window():
    return torch.hann_window(2*HOP, device="cuda").sqrt()


def training_bank(rng):
    clean, noise = [], []
    ledger = dict(speech=[], music=[], rooms=[], air=[])
    responses = [np.ones(1)]
    for prefix in ("h001_", "h007_", "h025_"):
        h, path, digest = selection.rooms.read_room(prefix)
        responses.append(h)
        ledger["rooms"].append(dict(path=path, sha256=digest))
    for group in ("training", "bare-training", "daily-training"):
        path = ch.RESULTS / f"4090-chatterbox-{group}-admission-v1.json"
        for r in ch.read(path)["admitted"]:
            wav = r["wav"]
            assert r["request"]["role"] == "training-only" and ch.sha(Path(wav["path"])) == wav["sha256"]
            x, sr = sf.read(wav["path"])
            assert x.ndim == 1 and sr == 16000
            h = int(rng.integers(len(responses)))
            wet = fftconvolve(x, responses[h])
            clean.append(np.pad(wet, (2400, 2400)))
            ledger["speech"].append(dict(**wav, room=h, voice=r["request"]["voice"]))
    index = MUSIC/"ambient.json"
    prep = ch.read(ch.RESULTS/"4090-musan-music-preparation.json")
    excluded = ch.read(ch.RESULTS/"4090-c161-music-development-sources-v1.json")["records"]
    excluded_names = {r["source_path"] for r in excluded}
    excluded_hashes = {r["sha256"] for r in excluded}
    assert prep["all_training_only"] and ch.sha(index) == prep["index_sha256"]
    available = [r for r in ch.read(index) if r["source_path"] not in excluded_names and r["source_sha256"] not in excluded_hashes]
    for i in rng.choice(len(available), 64, replace=False):
        r = available[i]
        path = MUSIC/r["audio_path"]
        assert r["split"] == "training" and ch.sha(path) == r["source_sha256"]
        with sf.SoundFile(path) as f:
            assert f.samplerate == 16000 and f.channels == 1
            length = min(10*16000, len(f))
            start = int(rng.integers(len(f)-length+1))
            f.seek(start)
            noise.append(f.read(length))
        ledger["music"].append(dict(source_path=r["source_path"], sha256=r["source_sha256"], start=start, length=length))
    for basename in ("AirConditioner_4.wav", "VacuumCleaner_3.wav"):
        x, h = selection.noise_training(basename)
        noise.append(x)
        ledger["air"].append(dict(name=basename, sha256=h))
    def bank(rows):
        lens = torch.tensor([len(x) for x in rows], device="cuda")
        offsets = lens.cumsum(0)-lens
        return torch.from_numpy(np.concatenate(rows).astype("float32")).cuda(), lens, offsets
    return bank(clean), bank(noise), ledger


def draw(bank, ids, time):
    samples, lens, offsets = bank
    starts = (torch.rand(len(ids),device="cuda") * (lens[ids]-SEGMENT+1).clamp_min(1)).long()
    where = starts[:,None] + time
    return samples[offsets[ids,None]+torch.minimum(where,lens[ids,None]-1)] * (where < lens[ids,None])


def train():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    clean, noise, ledger = training_bank(rng)
    model = Mask().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    time = torch.arange(SEGMENT,device="cuda")
    win = window()
    logs = []
    for step in range(1, STEPS+1):
        ids = torch.randint(len(clean[1]), (BATCH,), device="cuda")
        target = draw(clean, ids, time)
        idx = torch.randint(len(noise[1])-2, (BATCH,),device="cuda")
        air = torch.randint(2, (BATCH,), device="cuda") + len(noise[1])-2
        idx = torch.where(torch.rand(BATCH,device="cuda") < .2, air, idx)
        masker = draw(noise, idx, time)
        amp = 10 ** ((torch.rand(BATCH,1,device="cuda") * 22 - 36)/20)
        target = target * amp / target.square().mean(-1,keepdim=True).sqrt().clamp_min(.005)
        snr = torch.rand(BATCH,1,device="cuda") * 27 - 9
        masker = masker * amp * (10**(-snr/20)) / masker.square().mean(-1,keepdim=True).sqrt().clamp_min(.0001)
        target *= (torch.rand(BATCH,1,device="cuda") >= .1)
        masker *= (torch.rand(BATCH,1,device="cuda") >= .1)
        c, n = spectrum(target,win), spectrum(masker,win)
        mixed = c+n
        gain, _ = model(mixed)
        ideal = c.abs() / (c.abs()+n.abs()).clamp_min(1e-5)
        # Magnitude compression keeps low-energy speech bands in the objective.
        error = ((gain*mixed.abs()+1e-8).pow(.5) - (c.abs()+1e-8).pow(.5)).abs()
        loss = error.mean() + .15 * (gain - ideal).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            logs.append(dict(step=step, loss=loss.item()))
            print(step, loss.item(), flush=True)
    assert not WEIGHTS.exists() and not FIT.exists()
    np.savez_compressed(WEIGHTS, **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    ch.save(FIT, dict(seed=SEED, steps=STEPS, batch=BATCH, segment=SEGMENT, sources=ledger,
        parameters=sum(p.numel() for p in model.parameters()), logs=logs, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, weights_sha256=ch.sha(WEIGHTS), script_sha256=ch.sha(Path(__file__)),
        policy="Final training step only, synthetic speech with unity or one of three training RIRs, "
        "training music excludes development track identities. Dry/reverberant synthetic speech is the "
        "reconstruction target. C161 frozen. Adam lr .001, 20 ms sqrt-Hann windows, 10 ms hop, single-direction GRU. "
        "Evaluation uses full causal output including one-hop overlap-add delay."))


def load_model():
    report = ch.read(FIT)
    assert ch.sha(WEIGHTS) == report["weights_sha256"]
    model = Mask().cuda().eval()
    with np.load(WEIGHTS, allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
    return model


@torch.inference_mode()
def enhance(model, pcm, chunk_frames=64):
    x = torch.tensor(pcm.astype("float32")/32768, device="cuda")
    win = window()
    spec = spectrum(F.pad(x, (HOP, (-len(x)) % HOP)), win)
    state = None
    overlap = torch.zeros(HOP,device="cuda")
    output = []
    for block in spec.split(chunk_frames):
        gain, state = model(block[None], state)
        frames = torch.fft.irfft(block * gain[0]) * win
        emitted = frames[:,:HOP] + torch.cat((overlap[None], frames[:-1,HOP:]))
        output.append(emitted.flatten())
        overlap = frames[-1,HOP:]
    y = torch.cat(output)[:len(x)]
    return (32768*y).round().clamp(-32768,32767).short().cpu().numpy()


if __name__ == "__main__":
    train()
