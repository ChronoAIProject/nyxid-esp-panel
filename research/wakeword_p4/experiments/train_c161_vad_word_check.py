"""Train one small classifier on generated wake-word segments and name/ordinary alternatives."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_word_window_v1 import WordWindow, bands

STEM = "4090-c161-vad-word-check"
WORK = Path(".local/wakeword/c161-vad-word-check")
CACHE = RESULTS / f"{STEM}-features.npz"
WEIGHTS = RESULTS / f"{STEM}-weights.npz"
FIT = RESULTS / f"{STEM}-training.json"
PLAN = RESULTS / f"{STEM}-plan.json"
SEED = 103507
STEPS = 500
VOICES = ("af_alloy", "af_aoede", "af_heart", "af_nicole", "af_river", "am_adam", "am_echo",
          "am_eric", "am_fenrir", "am_michael", "bf_isabella", "bf_lily")
ADDRESSES = ("Hey Nicks", "Hi Nicks", "Hey Nick", "Hi Nick", "Hey Max", "Hi Max", "Hey Mick", "Hi Mick",
             "Hey Chris", "Hi Chris", "Hi Nikki", "Nick", "Nice mix", "Please mix", "Hey Nicole", "I think")
CARRIERS = ("read the last item.", "bring the calendar back.")


def segment(frame, clock, start, end, available):
    # Native features are only read once their containing block has arrived.
    latest = clock.searchsorted(available, side="right")
    assert latest > 0 and start < end
    points = np.linspace(start-1440, min(end+1440, available), 32)
    indices = np.searchsorted(clock[:latest], points).clip(0, latest-1)
    return frame[indices].T.copy()


def inputs():
    from check_c161_wake_phone_duration_v1 import session
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly
    inventory = RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json"
    bindings = read(inventory)
    admitted = {r["request"]["conditioning_id"] for r in bindings["requests"] if r["request"]["split"] == "training"}
    assert set(VOICES) <= admitted
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p) == bindings["files"][str(p)] for p in (model, styles))
    rule = dict(seed=SEED, steps=STEPS, voices=VOICES, addresses=ADDRESSES, carriers=CARRIERS,
                training_positive=list(ADDRESSES[:2]), inventory_sha256=sha(inventory),
                model_sha256=sha(model), styles_sha256=sha(styles), script_sha256=sha(Path(__file__)),
                policy="One fixed class-balanced fit on labelled Kokoro training segments. Threshold logit >= 0. Keep the original name-contrast route and check only extra VAD matches. No development audio in fitting.")
    if PLAN.exists():
        assert read(PLAN) == dict(rule,voices=list(VOICES),addresses=list(ADDRESSES),carriers=list(CARRIERS))
    else:
        save(PLAN, rule)
    ort = session(str(model))
    speaker = Kokoro.from_session(ort, str(styles))
    rng = np.random.default_rng(SEED)
    records, arrays, labels = [], [], []
    for vi, voice in enumerate(VOICES):
        for ci, carrier in enumerate(CARRIERS):
            # The same voice, speed and sentence tail occur in both classes.
            speed = (.9, 1., 1.1)[(vi + ci) % 3]
            for address in ADDRESSES:
                text = f"{address}, {carrier}"
                phones = speaker.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
                ids = speaker.tokenizer.tokenize(phones)
                assert len(ids) == len(phones)
                wave, duration = ort.run(None, dict(input_ids=np.array([[0,*ids,0]], np.int64),
                    style=np.asarray(speaker.get_voice_style(voice)[len(ids)],np.float32), speed=np.array([speed], np.float32)))
                signal = resample_poly(wave.ravel(), 2, 3)
                pcm = np.rint(signal*32768).clip(-32768,32767).astype("<i2")
                d = duration.ravel()
                assert len(pcm) == sum(d)*400 and len(d) == len(phones)+2
                start, end = int(d[0])*400, int(sum(d[:1+phones.index(",")]))*400
                pad = 8000
                # Generate two level conditions per utterance; do not use validation RMS.
                source = dict(voice=voice, address=address, text=text, phones=phones, speed=speed,
                              duration=d.tolist(), pcm_sha256=digest(pcm), samples=len(pcm), start=start, end=end)
                records.append(source)
                for gain in (1., .25):
                    view = np.pad(np.rint(pcm*gain).astype("<i2"), (pad, 4800))
                    frame, clock = bands(view)
                    for _ in range(12):
                        left = pad+start+int(rng.uniform(-.08,.16)*16000)
                        right = max(left+1600,pad+end+int(rng.uniform(-.18,.10)*16000))
                        arrays.append(segment(frame, clock, left, right, min(len(view), right+1920)))
                        labels.append(address in ADDRESSES[:2])
        print("synthesized", voice, flush=True)
    with CACHE.open("xb") as f:
        np.savez_compressed(f, x=np.stack(arrays), y=np.array(labels))
    save(RESULTS / f"{STEM}-sources.json", dict(plan_sha256=sha(PLAN), provider=ort.get_providers(),
                                              features_sha256=sha(CACHE), records=records))


def fit():
    WORK.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(SEED)
    source = RESULTS / f"{STEM}-sources.json"
    assert sha(CACHE) == read(source)["features_sha256"] and sha(PLAN) == read(source)["plan_sha256"]
    with np.load(CACHE) as archive:
        x = torch.from_numpy(archive["x"]).cuda().float()/255
        y = archive["y"]
    assert x.shape == (len(VOICES)*len(ADDRESSES)*len(CARRIERS)*24, 20, 32)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    net = WordWindow().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.002)
    history = []
    for step in range(STEPS):
        ids = np.r_[rng.choice(pos,64), rng.choice(neg,128)]
        logits = net(x[ids])[:,0]
        loss = F.softplus(-logits[:64]).mean() + F.softplus(logits[64:]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (0, 99, 249, STEPS-1):
            history.append(dict(step=step+1, loss=loss.item()))
    with WEIGHTS.open("xb") as f:
        np.savez_compressed(f, **{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    with torch.no_grad():
        logits = torch.cat([net(part)[:,0] for part in x.split(512)]).cpu().numpy()
    assert np.isfinite(logits).all()
    save(FIT, dict(sources_sha256=sha(source), weights_sha256=sha(WEIGHTS), plan_sha256=sha(PLAN),
                   gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, history=history,
                   train_positive=len(pos), train_negative=len(neg), train_positive_accepted=int((logits[pos]>=0).sum()),
                   train_negative_accepted=int((logits[neg]>=0).sum()), parameters=sum(p.numel() for p in net.parameters())))


if __name__ == "__main__":
    import sys
    {"inputs": inputs, "fit": fit}[sys.argv[1]]()
