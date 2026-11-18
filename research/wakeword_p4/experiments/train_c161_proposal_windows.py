"""Fine-tune a compact acoustic verifier at the actual midlevel proposal clocks on training voices."""
import argparse

import numpy as np
import torch
from torch.nn import functional as F

from audit_c161_history_paths_v1 import decode, times
from check_c161_stream_stretch_v1 import MID, run_quantized
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, frontend, read, save, sha
from train_c161_mined_room_word import PREFIX, WORK
from train_c161_word_window_v1 import WIDTH, WordWindow, sources

STEM = "4090-c161-proposal-windows"
PARENT = PREFIX + "-hard"
PLAN = RESULTS / "4090-c161-word-window-training-v1.json"
SEED = 83811
STEPS = 300


def training_windows():
    inventory = {r["key"]: r for r in read(PLAN)["training"]}
    fit = read(RESULTS / f"{MID}-training-v1.json")["reports"][0]
    proposer = RESULTS / fit["path"]
    assert sha(proposer) == fit["sha256"]
    scale, zero = REF_KERNEL(model_path=str(proposer)).get_input_details()[0]["quantization"]
    windows, labels, records = [], [], []
    for src, pcm in sources():
        entry = inventory.pop(src["key"])
        assert digest(pcm) == entry["pcm_sha256"] and src["positive"] == entry["positive"]
        padded = np.pad(pcm, (8000, 8000))
        values, clock = frontend(padded)
        pooled = values.reshape(-1, 3, 20, 2).mean(axis=(1, 3))
        bands = np.rint(pooled * (255 / 26)).clip(0, 255).astype(np.uint8)
        q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
        events = set()
        for expand in (False, True):
            codes, _ = run_quantized(q, proposer, expand)
            events.update(times(decode(codes, clock)))
        training_ticks = []
        for tick in sorted(events):
            relative = tick - 8000
            if not 0 < relative <= len(pcm) + 4800:
                continue
            if src["positive"] and not src["end"] - 1600 <= relative <= src["end"] + 4800:
                continue
            index = int(np.searchsorted(clock, tick))
            assert clock[index] == tick and index >= WIDTH - 1
            windows.append(bands[index + 1 - WIDTH:index + 1].T.copy())
            labels.append(bool(src["positive"]))
            training_ticks.append(relative)
        records.append(dict(key=src["key"], voice=src["voice"], pcm_sha256=entry["pcm_sha256"],
                            positive=src["positive"], word_end=src["end"], ticks=training_ticks))
    assert not inventory and len(records) == 918
    return torch.from_numpy(np.stack(windows)).cuda().float() / 255, np.asarray(labels), records, fit


def train():
    torch.set_num_threads(1)
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(SEED)
    x, labels, records, proposer = training_windows()
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    print("training proposal windows", len(pos), len(neg), flush=True)
    parent = read(RESULTS / f"{PARENT}-training-v1.json")
    weights = RESULTS / f"{PARENT}-weights-v1.npz"
    calibration = RESULTS / f"{PARENT}-calibration-v1.npz"
    assert sha(weights) == parent["weights_sha256"] and sha(calibration) == parent["calibration_sha256"]
    bank = WORK / "negative-banks.npz"
    assert sha(bank) == read(RESULTS / f"{PREFIX}-mining-v1.json")["cache_sha256"]
    model = WordWindow().cuda()
    with np.load(weights) as data:
        model.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files})
    with np.load(bank) as data:
        hard = torch.from_numpy(data["hard"]).cuda().float() / 255
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00012, weight_decay=0.001)
    history = []
    for step in range(STEPS):
        positive = x[rng.choice(pos, 64)]
        negative = x[rng.choice(neg, 64)]
        background = hard[rng.choice(len(hard), 32)].reshape(-1, 20, WIDTH)
        logits = model(torch.cat((positive, negative, background)))[:, 0]
        p, n, b = logits.split([len(positive), len(negative), len(background)])
        loss = F.softplus(4.5 - p).mean() + F.softplus(n + 2).mean() + F.softplus(b + 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in (0, 99, STEPS - 1):
            history.append(dict(step=step + 1, loss=loss.item()))
    wfile = RESULTS / f"{STEM}-weights-v1.npz"
    with wfile.open("xb") as f:
        np.savez_compressed(f, **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
    with np.load(calibration) as data:
        inputs = data["inputs"]
    with torch.inference_mode():
        expected = model(torch.from_numpy(inputs).cuda().float() / 255).sigmoid()[:, 0].cpu().numpy()
    cfile = RESULTS / f"{STEM}-calibration-v1.npz"
    with cfile.open("xb") as f:
        np.savez_compressed(f, inputs=inputs, probabilities=expected)
    save(RESULTS / f"{STEM}-training-v1.json", dict(gpu=gpu, seed=SEED, steps=STEPS, parent=parent["weights_sha256"],
         proposer=proposer, inputs_sha256=sha(PLAN), hard_bank_sha256=sha(bank), positive_windows=len(pos),
         negative_windows=len(neg), training=records, history=history, weights_sha256=sha(wfile), calibration_sha256=sha(cfile)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "export"))
    if parser.parse_args().action == "train":
        train()
    else:
        import export_c161_word_window_v1 as exporter
        exporter.STEM = STEM
        exporter.main()
