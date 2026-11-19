"""Train matched keyword readouts at the last five deliverable ticks before the keyword boundary."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
import torch
from torch.nn import functional as F

import check_marked_connected_panel as mixed
import train_proposal_deadlines as prior

ch = mixed.ch
STEM = "4090-c161-connected-overlap"
SEED = 463102
ARMS = ("room", "overlap")
STEPS = 400
SRC = ch.RESULTS / (STEM + "-sources.json")


def acoustics():
    root = prior.music.MUSIC
    fit = ch.read(ch.RESULTS / "4090-c161-causal-speech-mask-fit.json")
    index = {r["source_path"]: r for r in ch.read(root/"ambient.json")}
    excluded = {r["sha256"] for r in ch.read(ch.RESULTS/"4090-c161-music-development-sources-v1.json")["records"]}
    beds = []
    for r in fit["sources"]["music"]:
        item = index[r["source_path"]]
        path = root / item["audio_path"]
        assert item["split"] == "training" and ch.sha(path) == r["sha256"] and r["sha256"] not in excluded
        with sf.SoundFile(path) as f:
            assert f.samplerate == 16000 and f.channels == 1
            f.seek(r["start"])
            beds.append(f.read(r["length"]))
    room_bank = []
    eval_rooms = {r["source"][k] for r in ch.read(mixed.study.OUT)["records"]
                  for k in ("external_rir_sha256", "panel_rir_sha256")}
    for r in ch.read(ch.RESULTS/"4090-mit-rir-audit.json")["records"]:
        if r["split"] != "training" or r["sha256"] in eval_rooms:
            continue
        if int(Path(r["path"]).name[1:4]) % 7 != 3:
            continue
        p = Path(".local/wakeword/mit-rirs") / r["path"]
        assert ch.sha(p) == r["sha256"]
        h, rate = sf.read(p)
        assert rate == 16000
        room_bank.append((r["sha256"], h / np.linalg.norm(h)))
    assert len(room_bank) >= 2
    panel, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000
    local = [fftconvolve(panel.astype(float), h) for _, h in room_bank]
    return beds, fit["sources"]["music"], room_bank, local


def collect(*, suffix_test=False):
    native = prior.letters.native_frontend()
    teacher = prior.parent_model().eval()
    rng = np.random.default_rng(SEED)
    beds, music, rooms, panel = acoustics()
    arms = ("background", "command") if suffix_test else ARMS
    rows, bank = [], {arm:dict(positive=[], absent=[], negative=[]) for arm in arms}
    original = {r["key"]:r for r in ch.read(SRC)["records"]} if suffix_test else {}
    admitted = {r["key"]:r for r in ch.read(prior.base.RESULTS/(prior.STEM+"-sources.json"))["records"]
                if r["condition"]=="dry"}
    external_eval = {r["original_source_pcm_sha256"] for r in ch.read(mixed.study.connected.PLAN)["records"]}
    for src, wave in prior.music.word.sources():
        digest = ch.digest(wave)
        assert digest == admitted.pop(src["key"])["source_pcm_sha256"] and digest not in external_eval
        hx, hp = rng.choice(len(rooms), 2, replace=False)
        onset = 32000
        wet = fftconvolve(wave.astype(float), rooms[hx][1])
        ext = np.pad(wet, (onset, 16000))
        k = int(rng.integers(len(beds)))
        offset = int(rng.integers(len(beds[k])))
        sound = beds[k][(offset+np.arange(len(ext))) % len(beds[k])]
        stop = src["end"] if src["positive"] else len(wave)
        part = wet[:stop]
        active = np.abs(part) > max(1, np.max(np.abs(part))*.035)
        assert active.any()
        rms = np.sqrt(np.mean(part[active]**2))
        snr = int(rng.choice([-6,0,6,12]))
        sound = sound * (rms * 10**(-snr/20) / max(1.e-8, np.sqrt(np.mean(sound**2))))
        panel_at = onset + int(rng.integers(-8000, 8001))
        local = panel[hp][:len(ext)-panel_at]
        panel_db = int(rng.choice([-6,0,6]))
        active_local = np.abs(local) > max(1., np.abs(local).max()*.035)
        sound[panel_at:panel_at+len(local)] += local * (rms * 10**(panel_db/20) / np.sqrt(np.mean(local[active_local]**2)))
        gain = min(1., 30000 / max(1., np.abs(ext+sound).max(), np.abs(sound).max(), np.abs(ext).max()))
        row = dict(**src, pcm_sha256=digest, room_sha256=rooms[hx][0], panel_rir_sha256=rooms[hp][0],
                   music_sha256=music[k]["sha256"], music_offset=offset, snr_db=snr, panel_rms_db=panel_db,
                   panel_at=panel_at, gain=gain, inputs={})
        # Delete the address at the source, before room convolution, to preserve the command.
        suffix = None
        if suffix_test and src["positive"]:
            left = int(src["end"])
            clean = wave.copy()
            clean[:left] = 0
            assert np.array_equal(clean[left:], wave[left:])
            suffix = np.pad(fftconvolve(clean.astype(float), rooms[hx][1]), (onset, 16000))
            row["retained_suffix_samples"] = len(clean[left:])
            row["retained_suffix_sha256"] = ch.digest(clean[left:])
        old = original.pop(src["key"]) if suffix_test else None
        for arm in arms:
            bg = np.zeros_like(sound) if arm == "room" else sound
            inputs = [("positive" if src["positive"] else "negative", ext+bg)]
            if src["positive"]:
                inputs.append(("absent", bg + suffix if arm == "command" else bg))
            for label, signal in inputs:
                rounded = np.rint(signal*gain)
                mic = rounded.clip(-32768, 32767).astype("<i2")
                if old is not None and (arm != "command" or label != "absent"):
                    assert ch.digest(mic) == old["inputs"]["overlap."+label]["pcm_sha256"]
                _, bands, clock = prior.base.native_inputs(native, mic)
                patches = np.lib.stride_tricks.sliding_window_view(bands, 32, axis=0)
                clocks = clock[31:]
                blocks = (clocks+255)//256*256
                if src["positive"]:
                    end = int(np.searchsorted(blocks, onset+src["end"]+(1920 if suffix_test else 0), side="left"))
                else:
                    candidates = np.flatnonzero((clocks >= onset) & (clocks <= onset+len(wave)))
                    with torch.no_grad():
                        x = torch.from_numpy(patches[candidates]).cuda().float()/255
                        scores = teacher(x).sigmoid().flatten().unfold(0,5,1).mean(1)
                        end = int(candidates[int(scores.argmax())+4])+1
                assert end >= 5
                bank[arm][label].append(patches[end-5:end].copy())
                row["inputs"][arm+"."+label] = dict(pcm_sha256=ch.digest(mic), blocks=blocks[end-5:end].tolist())
                if suffix_test:
                    row["inputs"][arm+"."+label]["clipped"] = int(np.count_nonzero(rounded != mic))
            if suffix_test and arm == "command" and src["positive"]:
                assert bank["command"]["positive"][-1].tobytes() == bank["background"]["positive"][-1].tobytes()
        rows.append(row)
    assert not admitted and not original
    return bank, rows


def train(*, suffix_test=False):
    stem = STEM + "-post120" if suffix_test else STEM
    source_path = ch.RESULTS / (stem + "-sources.json")
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    bank, rows = collect(suffix_test=suffix_test)
    mining = prior.music.mining.WORK/"negative-banks.npz"
    assert ch.sha(mining)==ch.read(ch.RESULTS/(prior.music.mining.PREFIX+"-mining-v1.json"))["cache_sha256"]
    with np.load(mining, allow_pickle=False) as z:
        hard = torch.from_numpy(z["hard"]).cuda().float()/255
    with TemporaryDirectory(dir=ch.RESULTS) as folder:
        work = Path(folder)
        ch.save(work/source_path.name,dict(records=rows, seed=SEED, panel_sha256=ch.sha(mixed.panel.demo.WAVE),
                training_music_manifest_sha256=ch.sha(ch.RESULTS/"4090-c161-causal-speech-mask-fit.json"),
                parent_sha256=ch.sha(ch.RESULTS/(prior.music.STEM+"-weights-v1.npz")),
                hard_sha256=ch.sha(mining), script_sha256=ch.sha(Path(__file__))))
        for arm in bank:
            p,a,n = (torch.from_numpy(np.stack(bank[arm][k])).cuda().float()/255
                     for k in ("positive","absent","negative"))
            model = prior.parent_model()
            rng = np.random.default_rng(SEED)
            opt = torch.optim.AdamW(model.parameters(), lr=.00025, weight_decay=.001)
            logs = []
            for step in range(1, STEPS+1):
                ix = rng.choice(len(p),48)
                batch = torch.cat((p[ix], a[ix], n[rng.choice(len(n),64)], hard[rng.choice(len(hard),24)]))
                pos, neg = model(batch.flatten(0,1)).reshape(-1,5).split([48,len(batch)-48])
                loss = F.softplus(4.5-pos).mean() + F.softplus(neg+2).mean() + F.softplus(neg.mean(-1).topk(24).values+2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step in (1,100,200,400):
                    logs.append(dict(step=step, loss=loss.item()))
            def path(kind):
                return work / f"{stem}-{arm}-{kind}-v1.npz"
            np.savez_compressed(path("weights"), **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            cal = torch.cat([b[rng.choice(len(b),64),rng.integers(0,5,64)] for b in (p,a,n,hard)])
            with torch.no_grad():
                np.savez_compressed(path("calibration"), inputs=(cal*255).round().byte().cpu().numpy(),
                                    probabilities=model(cal).sigmoid().flatten().cpu().numpy())
            ch.save(work/f"{stem}-{arm}-training-v1.json", dict(arm=arm, steps=STEPS, seed=SEED, logs=logs,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__, positive_bags=len(p),
                    paired_absent_bags=len(a), negative_bags=len(n), hard_bags=len(hard),
                    sources_sha256=ch.sha(work/source_path.name), weights_sha256=ch.sha(path("weights")),
                    calibration_sha256=ch.sha(path("calibration")), policy=("Five ticks deliverable before address end plus 120 ms. Background and command arms "
                    "use identical targets, other negatives, hard negatives, RNG and initialization. The command "
                    "arm keeps the external suffix of its paired absent input before room filtering. "
                    "Threshold and smoothing frozen for evaluation." if suffix_test else "Last five consecutive ticks with the "
                    "256-sample delivery strictly before the training target end. Both arms share sources, external "
                    "room, gain, initialization, hard bank and minibatch IDs. Overlap adds training music plus the "
                    "known panel demo before selecting negative peaks; paired absent control removes external source.")))
            print(arm, logs[-1],flush=True)
        for f in work.iterdir():
            f.replace(ch.RESULTS/f.name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("train", "export"), default="train")
    parser.add_argument("--post120", action="store_true")
    args = parser.parse_args()
    if args.action == "export":
        import export_c161_word_window_v1 as exporter
        stem = STEM + "-post120" if args.post120 else STEM
        for arm in (("background", "command") if args.post120 else ARMS):
            exporter.STEM = stem+"-"+arm
            exporter.main()
    else:
        train(suffix_test=args.post120)
