"""Continue a frozen word classifier with matched same-voice command-suffix controls."""
import argparse
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from check_c161_v5_command_ns import digest
from check_c161_word_parts import training_marks
from fit_c161_history_gain_v1 import DAILY, RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_mined_room_word import PREFIX as MINING, WORK
from train_c161_room_word import STEM as ROOM, room_render
from train_c161_word_window_v1 import WIDTH, WordWindow, bands, sources, training_data

STEM = "4090-c161-command-swap"
PARENT = f"{MINING}-hard"
SEED = 78109
STEPS = 300


def make_pairs():
    marks = training_marks()
    originals = {r["key"]: (r, pcm) for r, pcm in sources() if r["key"] in marks and marks[r["key"]]["command_start"] is not None}
    histories = {r["index"]: r["wav"] for r in read(RESULTS / DAILY)["admitted"]}
    fit = read(RESULTS / f"{ROOM}-training-v1.json")
    sets = {"own": [], "swapped": []}
    rows = []
    for source in fit["training"]:
        key = source["key"]
        if key not in originals:
            continue
        row, pcm = originals[key]
        def first_word(meta):
            return meta["text"].split(",", 1)[1].strip().lower().split()[0]
        choices = [k for k, (r, _) in originals.items() if r["voice"] == row["voice"] and first_word(r) != first_word(row)]
        if not choices:
            continue
        cut = marks[key]["command_start"]
        donor = choices[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(choices)]
        other = originals[donor][1]
        other_start = marks[donor]["command_start"]
        prefix = load_wav(histories[source["prefix_id"]])
        prefix = np.r_[np.zeros_like(prefix) if source["quiet"] else prefix, np.zeros(source["onset"]-len(prefix), np.int16)]
        original = np.r_[prefix, pcm, np.zeros(8000, np.int16)]
        assert digest(original) == source["clean_stream_sha256"]
        transition = len(prefix) + cut
        end = source["onset"] + row["end"]
        kept = {}
        for label, tail in (("own", pcm[cut:]), ("swapped", other[other_start:])):
            # First 4 ms crossfade preserves the original before the command boundary.
            tail = tail.copy()
            fade = np.linspace(0, 1, 64)
            tail[:64] = np.rint((1-fade)*pcm[cut:cut+64] + fade*tail[:64])
            full = np.r_[original[:transition], tail, np.zeros(8000, np.int16)]
            if label == "own":
                np.testing.assert_array_equal(full, original)
            frame, clock = bands(full)
            windows = np.lib.stride_tricks.sliding_window_view(frame, WIDTH, axis=0)
            ticks = clock[WIDTH-1:]
            mask = (ticks > transition) & (ticks >= end-480) & (ticks <= end+2880)
            if label == "own":
                selected = ticks[mask]
            else:
                np.testing.assert_array_equal(ticks[mask], selected)
            sets[label].extend(windows[mask])
            kept[label] = digest(full)
        if len(selected):
            rows.append(dict(key=key, donor=donor, text=row["text"], donor_text=originals[donor][0]["text"],
                             voice=row["voice"], boundary=transition, ticks=selected.tolist(), pcm_sha256=kept))
    assert len(sets["own"]) == len(sets["swapped"]) == sum(len(r["ticks"]) for r in rows)
    return {k: torch.from_numpy(np.stack(v)).cuda().float()/255 for k, v in sets.items()}, rows


def train():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gpu = torch.cuda.get_device_name()
    assert "4090" in gpu
    x, y, hard, rows = training_data(room_render())
    assert [r["stream_sha256"] for r in rows] == [r["stream_sha256"] for r in read(RESULTS/f"{ROOM}-training-v1.json")["training"]]
    pairs, sources_used = make_pairs()
    ledger = RESULTS/f"{STEM}-pairs.json"
    save(ledger, dict(seed=SEED, records=sources_used, positive_windows=len(pairs["own"])))
    bank = WORK/"negative-banks.npz"
    assert sha(bank) == read(RESULTS/f"{MINING}-mining-v1.json")["cache_sha256"]
    with np.load(bank) as data:
        negative_bank = torch.from_numpy(data["hard"]).cuda().float()/255
    pos, neg, near = (np.flatnonzero(v) for v in (y==1, y==0, (y==0)&hard))
    training = read(RESULTS/f"{PARENT}-training-v1.json")
    parent = RESULTS/f"{PARENT}-weights-v1.npz"
    assert sha(parent) == training["weights_sha256"]
    for mode, extra in pairs.items():
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = WordWindow().cuda()
        with np.load(parent) as weights:
            model.load_state_dict({k: torch.from_numpy(weights[k]) for k in weights.files})
        opt = torch.optim.AdamW(model.parameters(), lr=.00018, weight_decay=.001)
        history = []
        for step in range(STEPS):
            positive = torch.cat((x[rng.choice(pos, 64)], extra[rng.choice(len(extra), 32)]))
            negative = x[np.r_[rng.choice(neg, 96), rng.choice(near, 48)]]
            mined = negative_bank[rng.choice(len(negative_bank), 32)].reshape(-1,20,WIDTH)
            logits = model(torch.cat((positive, negative, mined)))[:,0]
            p, n, h = logits.split([len(positive), len(negative), len(mined)])
            loss = F.softplus(4.5-p).mean() + F.softplus(n+2).mean() + F.softplus(n.topk(24).values+2).mean() + F.softplus(h+2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (0, 99, STEPS-1):
                history.append(dict(step=step+1, loss=loss.item()))
        stem = f"{STEM}-{mode}"
        wfile = RESULTS/f"{stem}-weights-v1.npz"
        with wfile.open("xb") as out:
            np.savez_compressed(out, **{k: v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        # Calibration inputs are identical across arms, from the shared training partition.
        selected = np.r_[rng.choice(pos,96,replace=False),rng.choice(neg,128,replace=False)]
        batch = torch.cat((x[selected], negative_bank[rng.choice(len(negative_bank),32,replace=False),2]))
        with torch.no_grad():
            probabilities = model(batch).sigmoid()[:,0].cpu().numpy()
        cfile = RESULTS/f"{stem}-calibration-v1.npz"
        with cfile.open("xb") as out:
            np.savez_compressed(out, inputs=(batch.cpu().numpy()*255).round().astype(np.uint8), probabilities=probabilities)
        save(RESULTS/f"{stem}-training-v1.json", dict(gpu=gpu, seed=SEED, steps=STEPS, mode=mode,
             pairs_sha256=sha(ledger), parent_weights_sha256=sha(parent), hard_bank_sha256=sha(bank),
             history=history, weights_sha256=sha(wfile), calibration_sha256=sha(cfile)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train","export","replay","compare"))
    action = parser.parse_args().action
    if action == "train":
        train()
    else:
        for mode in ("own", "swapped"):
            candidate = f"{STEM}-{mode}"
            if action == "compare":
                from check_c161_room_word import main
                main(candidate)
            else:
                if action == "export":
                    import export_c161_word_window_v1 as runner
                else:
                    import replay_c161_word_window_v1 as runner
                runner.STEM = candidate
                runner.main()
