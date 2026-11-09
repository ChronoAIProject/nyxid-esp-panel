"""Prepare 10 ms MicroFrontend features and speaker-disjoint human negatives."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pymicro_features import MicroFrontend
from scipy.signal import fftconvolve, lfilter
import soundfile as sf

FRAMES = 180
RATE = 16000


def features(pcm, frontend=None):
    frontend = frontend or MicroFrontend()
    pcm = np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)
    pcm = np.pad(pcm, (0, (-len(pcm)) % 160))
    result = []
    for at in range(0, len(pcm), 160):
        frame = frontend.process_samples(pcm[at:at + 160].tobytes())
        assert frame.samples_read == 160
        if frame.features:
            result.append(np.rint(np.asarray(frame.features) * 25.6).astype(np.uint16))
    return np.asarray(result, dtype=np.uint16).reshape(-1, 40)


def human_worker(job):
    corpus, root, split, speaker = job
    frontend = MicroFrontend()
    parts = []
    samples = 0
    files = sorted((Path(corpus) / str(speaker)).rglob("*.flac"))
    for path in files:
        pcm, rate = sf.read(path, dtype="int16")
        assert rate == RATE and pcm.ndim == 1
        # Continuous frontend state across this speaker's utterances. Add a
        # short real-time silence at boundaries, and count it in exposure.
        pcm = np.concatenate((pcm, np.zeros(3200, dtype=np.int16)))
        parts.append(features(pcm, frontend))
        samples += len(pcm)
    path = Path(root) / "human" / split / f"{speaker}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.concatenate(parts)
    np.save(path, array)
    return {"split": split, "speaker": speaker, "path": str(path.relative_to(root)),
            "frames": len(array), "seconds": samples / RATE, "files": len(files)}


def background(rng, n):
    noise = lfilter([1], [1, -float(rng.uniform(.5, .96))], rng.normal(size=n))
    noise /= max(1e-8, np.sqrt(np.mean(noise**2)))
    return noise.astype(np.float32)


def render(item, dataset, rng, augmented):
    pcm, rate = sf.read(Path(dataset) / item["path"], dtype="float32")
    assert rate == RATE
    lead = 28800  # 1.8 seconds of frontend/model history before the phrase.
    n = lead + len(pcm) + 11200
    signal = np.zeros(n, dtype=np.float32)
    signal[lead:lead + len(pcm)] = pcm
    if augmented:
        ir = np.zeros(2400, dtype=np.float32)
        ir[0] = 1
        taps = rng.integers(80, len(ir), size=24)
        ir[taps] += rng.normal(0, .2, len(taps)) * np.exp(-taps / rng.uniform(500, 1400))
        signal = fftconvolve(signal, ir)[:n]
    level = float(rng.uniform(.025, .25)) if augmented else .12
    speech_rms = np.sqrt(np.mean(pcm**2))
    signal *= level / max(.02, speech_rms)
    noise = background(rng, n)
    snr = float(rng.uniform(8, 30)) if augmented else 45
    mixed = signal + noise * level * 10 ** (-snr / 20)
    return np.clip(mixed, -1, 1) * 32767, lead


def synthetic_worker(job):
    item, dataset, variants = job
    seed = int(hashlib.sha256(item["path"].encode()).hexdigest()[:8], 16)
    result = []
    for variant in range(variants):
        rng = np.random.default_rng(seed + variant)
        pcm, lead = render(item, dataset, rng, augmented=variant > 0)
        raw = features(pcm)
        if item["label"]:
            end = (lead + item["keyword_end"] + int(rng.uniform(0, .30) * RATE)) // 160 - 2
            start = max(0, end - FRAMES)
        else:
            # Keep the complete confusing phrase in the window where possible.
            end = (lead + min(item["samples"], 24000)) // 160 - 2
            start = max(0, end - FRAMES)
        clip = raw[start:start + FRAMES]
        if len(clip) < FRAMES:
            clip = np.pad(clip, ((FRAMES - len(clip), 0), (0, 0)))
        result.append(clip)
    return item, np.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    corpus = args.corpus.resolve()
    rng = np.random.default_rng(20261110)
    speakers = np.array(sorted(int(p.name) for p in corpus.iterdir() if p.is_dir()))
    rng.shuffle(speakers)
    nval = max(1, len(speakers) // 5)
    partitions = {"training": speakers[:-2*nval], "validation": speakers[-2*nval:-nval], "testing": speakers[-nval:]}
    started = time.monotonic()
    human = []
    jobs = [(str(corpus), str(root), split, int(speaker)) for split, ids in partitions.items() for speaker in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(human_worker, jobs):
            human.append(item)
            print(json.dumps({"human_speakers": len(human), "total": len(jobs)}), flush=True)
    (root / "human.json").write_text(json.dumps(human, indent=2) + "\n")
    entries = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    assert (args.dataset / "generation.json").exists(), "Synthetic generation must finish first"
    manifest = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in ("training", "validation", "testing"):
            for label in (1, 0):
                records = [e for e in entries if e["split"] == split and e["label"] == label]
                variants = 4 if split == "training" else 2
                array = np.lib.format.open_memmap(root / f"{split}-{label}.npy", mode="w+", dtype=np.uint16,
                                                shape=(len(records)*variants, FRAMES, 40))
                for index, (record, data) in enumerate(pool.map(synthetic_worker, [(r, str(args.dataset), variants) for r in records], chunksize=12)):
                    array[index*variants:(index+1)*variants] = data
                    for variant in range(variants):
                        manifest.append({"split": split, "label": label, "row": index*variants+variant,
                                         "source": record["path"], "variant": variant,
                                         "speakers": record["speakers"]})
                    if index % 300 == 0:
                        print(json.dumps({"split": split, "label": label, "prepared": index, "total": len(records)}), flush=True)
                array.flush()
    (root / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    summary = {"frames": FRAMES, "sample_rate": RATE, "window_ms": 30, "hop_ms": 10,
               "human_speakers": {s: ids.tolist() for s, ids in partitions.items()},
               "human_hours": {s: sum(r["seconds"] for r in human if r["split"] == s)/3600 for s in partitions},
               "synthetic_windows": len(manifest), "elapsed_seconds": round(time.monotonic()-started, 1)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
