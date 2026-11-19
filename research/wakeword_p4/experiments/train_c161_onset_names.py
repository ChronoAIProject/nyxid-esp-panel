"""Train a competing-name classifier on connected commands at real VAD onsets."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from check_c161_v5_command_ns import digest
from check_c161_vad_pronunciations import windows
from fit_c161_history_gain_v1 import DAILY, PRIME, RESULTS, features, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_vad_word_check import VOICES

STEM = "4090-c161-onset-names"
WORK = Path(".local/wakeword/c161-onset-names")
WEIGHTS = RESULTS / f"{STEM}-weights.npz"
FIT = RESULTS / f"{STEM}-fit.json"
SOURCES = RESULTS / f"{STEM}-sources.json"
CACHE = WORK / "inputs.npz"
SEED = 285973
ADDRESSES = ("Hey Nicks", "Hi Nicks", "Hey Nick", "Hi Nick", "Hey Mick", "Hi Mick",
             "Hey Max", "Hi Chris", "Hey Willow", "Hi Nikki")
LABELS = (0, 0, 1, 1, 2, 2, 3, 3, 3, 3)
CARRIERS = ("send the note.", "start the music.", "add an alarm.", "write a reminder.")
CLASSES = ("Nyx", "Nick", "Mick", "other")
FRAMES = 144
WAIT = 21120  # 44 blocks after VAD onset, 1.32 s
STEPS = 650


def observations(pcm):
    raw = features(np.pad(pcm, (PRIME, 0)))
    # Each native 30 ms feature window advances by 10 ms. Read only fully available frames.
    entries = []
    for onset, _, _ in windows(pcm):
        end = onset + WAIT
        if end <= len(pcm):
            last = (end + PRIME) // 160 - 2
            patch = raw[last-FRAMES:last]
            assert patch.shape == (FRAMES, 40)
            entries.append((onset, end, np.rint(patch.clip(0, 666) * (255./666)).astype(np.uint8)))
    return entries


def synthesize():
    from check_c161_wake_phone_duration_v1 import session
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly

    inventory = RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json"
    records = read(inventory)
    allowed = {r["request"]["conditioning_id"] for r in records["requests"] if r["request"]["split"] == "training"}
    assert set(VOICES) <= allowed
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p) == records["files"][str(p)] for p in (model, styles))
    engine = session(str(model))
    synth = Kokoro.from_session(engine, str(styles))
    for vi, voice in enumerate(VOICES):
        for ci, carrier in enumerate(CARRIERS):
            # Each speaker exposes every name to both command-initial /s/ and other initials.
            # Speed and punctuation are identical across all addresses in each cell.
            sep = ", " if (vi + ci) % 3 == 0 else " "
            speed = (.88, 1., 1.12)[(vi//3 + ci) % 3]
            for address, label in zip(ADDRESSES, LABELS, strict=True):
                text = address + sep + carrier
                phones = synth.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
                ids = synth.tokenizer.tokenize(phones)
                assert len(ids) == len(phones)
                raw, durations = engine.run(None, dict(input_ids=np.array([[0,*ids,0]], np.int64),
                    style=np.asarray(synth.get_voice_style(voice)[len(ids)], np.float32), speed=np.array([speed],np.float32)))
                wave = resample_poly(raw.ravel(), 2, 3)
                assert np.isfinite(wave).all() and len(wave) == int(durations.sum()) * 400
                pcm = np.rint(wave * 32768).clip(-32768,32767).astype("<i2")
                yield dict(kind="kokoro", voice=voice, text=text, address=address, label=label, carrier=carrier,
                           separator=sep, speed=speed, phones=phones, duration=durations.ravel().tolist()), pcm
        print("synthesized", voice, flush=True)


def training_sources():
    yield from synthesize()
    for r in read(RESULTS / DAILY)["admitted"]:
        q = r["request"]
        assert q["role"] == "training-only" and q["requested_label"] == 0
        yield dict(kind="chatterbox", voice=q["voice"], text=q["text"], index=r["index"], label=3), load_wav(r["wav"])


def inputs():
    WORK.mkdir(exist_ok=True)
    rows, inputs, labels = [], [], []
    for row, pcm in training_sources():
        views = []
        for gain in (1., .25):
            padded = np.pad(np.rint(pcm*gain).astype("<i2"), (12000,24000))
            matches = observations(padded)
            if matches:
                onset, end, patch = matches[0]
                inputs.append(patch)
                labels.append(row["label"])
                views.append(dict(gain=gain, onset=onset, end=end))
        rows.append(dict(**row, pcm_sha256=digest(pcm), samples=len(pcm), views=views))
    with CACHE.open("xb") as handle:
        np.savez_compressed(handle, x=np.stack(inputs), y=np.array(labels))
    inventory = RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json"
    save(SOURCES, dict(records=rows, features_sha256=sha(CACHE), script_sha256=sha(Path(__file__)),
         inventory_sha256=sha(inventory), daily_sha256=sha(RESULTS/DAILY), kokoro_provider="CUDAExecutionProvider"))


class Classifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 24, (7,5), stride=2, padding=(3,2)), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 32, (5,3), stride=2, padding=(2,1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,3)), nn.Flatten(), nn.Linear(48*4*3, 48), nn.ReLU(), nn.Linear(48,4))


def fit():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    assert sha(CACHE) == read(SOURCES)["features_sha256"]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with np.load(CACHE) as data:
        x = torch.from_numpy(data["x"]).cuda().float().unsqueeze(1)/255
        y = torch.from_numpy(data["y"]).cuda()
    pools = [np.flatnonzero(y.cpu().numpy() == i) for i in range(4)]
    model = Classifier().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.003)
    history = []
    for i in range(STEPS):
        ids = np.concatenate([rng.choice(pool, 16) for pool in pools])
        batch = x[ids].clone()
        # Slight onset jitter, with silence beyond the recorded window.
        shift = int(rng.integers(-6, 7))
        batch = batch.roll(shift, dims=2)
        if shift > 0:
            batch[:,:,:shift] = 0
        elif shift < 0:
            batch[:,:,shift:] = 0
        t, f = rng.integers(FRAMES-6), rng.integers(38)
        batch[:,:,t:t+6] = 0
        batch[:,:,:,f:f+2] = 0
        loss = nn.functional.cross_entropy(model(batch), y[ids], label_smoothing=.03)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i in (0, 99, 399, STEPS-1):
            history.append(dict(step=i+1, loss=float(loss.detach())))
    model.eval()
    with torch.inference_mode():
        prob = torch.cat([model(v).softmax(dim=1) for v in x.split(64)]).cpu().numpy()
    assert np.isfinite(prob).all()
    with WEIGHTS.open("xb") as f:
        np.savez_compressed(f, **{k:v.cpu().numpy() for k,v in model.state_dict().items()})
    save(FIT, dict(sources_sha256=sha(SOURCES), weights_sha256=sha(WEIGHTS), script_sha256=sha(Path(__file__)),
         gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, seed=SEED, classes=CLASSES,
         counts=[len(pool) for pool in pools], train_nyx_accepted=[int((prob[pool,0] >= .5).sum()) for pool in pools],
         parameters=sum(v.numel() for v in model.parameters()), steps=STEPS, history=history,
         policy="At a real speech onset following 450 ms of silence, wait 1.32 s; use the preceding 144 complete native feature frames. Accept P(Nyx) >= 0.5. No development threshold selection."))


if __name__ == "__main__":
    import sys
    {"inputs":inputs, "fit":fit}[sys.argv[1]]()
