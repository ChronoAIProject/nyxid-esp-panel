"""V2 waveform rendering and variable-length features for random alignments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from features import FRAMES, RATE, background, features


def render_stream(item, dataset, rng, variant, backgrounds):
    pcm, rate = sf.read(Path(dataset) / item["path"], dtype="float32")
    assert rate == RATE and pcm.ndim == 1
    lead = 32000 if variant == 0 else int(rng.uniform(1.8, 4.5)*RATE)
    size = lead + len(pcm) + 16000
    signal = np.zeros(size, dtype=np.float32)
    signal[lead:lead+len(pcm)] = pcm
    level = float(rng.uniform(.015, .35))
    signal *= level / max(.01, float(np.sqrt(np.mean(pcm**2))))
    if variant >= 2:
        ir = np.zeros(6400, dtype=np.float32)
        ir[0] = 1
        taps = rng.integers(60, len(ir), size=60)
        ir[taps] += rng.normal(0, .15, len(taps))*np.exp(-taps/rng.uniform(500, 2400))
        signal = fftconvolve(signal, ir)[:size]
    # Zero is genuinely quiet. One covers sub-audible microphone floors. Two
    # covers colored noise; three uses a real source-disjoint background file.
    if variant == 0:
        noise = np.zeros(size, dtype=np.float32)
    elif variant == 1:
        noise = rng.normal(0, rng.uniform(.5, 25)/32768, size).astype(np.float32)
    elif variant == 3 and backgrounds:
        noise, rate = sf.read(str(rng.choice(backgrounds)), dtype="float32")
        assert rate == RATE and noise.ndim == 1
        offset = int(rng.integers(0, max(1, len(noise))))
        noise = np.resize(np.roll(noise, -offset), size)
        rms = max(1e-6, float(np.sqrt(np.mean(noise**2))))
        noise *= level * 10**(-rng.uniform(5, 25)/20) / rms
    else:
        noise = background(rng, size)*level*10**(-rng.uniform(5, 35)/20)
    return np.clip(signal+noise, -1, 1)*32767, lead


def worker(job):
    item, dataset, variant, backgrounds = job
    seed = int(hashlib.sha256(item["path"].encode()).hexdigest()[:8], 16)+variant
    pcm, lead = render_stream(item, dataset, np.random.default_rng(seed), variant, backgrounds)
    raw = features(pcm)
    record = {**item, "variant": variant, "lead_frame": lead//160,
              "keyword_frame": (lead+item["keyword_end"])//160-2 if item["label"] else None}
    return record, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--human-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"features.json").exists(), "Use a fresh feature directory"
    dataset = root/"dataset"
    dataset.mkdir(exist_ok=True)
    entries = []
    for source in args.source:
        assert (source/"generation.json").exists(), f"Generation incomplete: {source.name}"
        link = dataset/source.name
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
        for line in (source/"manifest.jsonl").read_text().splitlines():
            item = json.loads(line)
            item["path"] = str(Path(source.name)/item["path"])
            item.setdefault("engine", "piper")
            if item["label"] and item["engine"] != "piper":
                # Some engines append >1 second of quiet. Align the label,
                # retaining every PCM sample (including weak final /s/).
                # This is offline fixture metadata, never live segmentation.
                pcm, rate = sf.read(dataset/item["path"], dtype="float32")
                assert rate == RATE and pcm.ndim == 1
                count = len(pcm)//160
                rms = np.sqrt(np.mean(pcm[:count*160].reshape(-1, 160)**2, axis=1))
                active = np.flatnonzero(rms > max(1e-6, float(rms.max())*.005))
                if not len(active):
                    raise ValueError(f"No active speech: {item['path']}")
                item["synthesis_end"] = item["keyword_end"]
                item["keyword_end"] = min(len(pcm), (int(active[-1])+1)*160+960)
                item["keyword_end_method"] = "Last 10 ms RMS frame above -46 dB of peak, plus 60 ms; PCM retained"
            entries.append(item)
    (dataset/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in entries))
    (dataset/"generation.json").write_text(json.dumps({"sources": [p.name for p in args.source]})+"\n")
    bg_rows = json.loads((args.backgrounds/"manifest.json").read_text())["records"]
    backgrounds = {s: [str(args.backgrounds/r["path"]) for r in bg_rows if r["split"] == s]
                   for s in ("training", "validation", "testing")}
    index = []
    offset = 0
    with (root/"features.bin").open("wb") as handle, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in backgrounds:
            group = [r for r in entries if r["split"] == split]
            variants = 4  # same rendering regimes in every split
            jobs = [(r, str(dataset), v, backgrounds[split]) for r in group for v in range(variants)]
            for count, (record, raw) in enumerate(pool.map(worker, jobs, chunksize=10), 1):
                handle.write(raw.astype("<u2").tobytes())
                index.append({**record, "offset": offset, "frames": len(raw)})
                offset += len(raw)
                if count % 500 == 0:
                    print(json.dumps({"split": split, "clips": count, "total": len(jobs)}), flush=True)
    (root/"features.json").write_text(json.dumps(index)+"\n")
    ambient = []
    for number, item in enumerate(bg_rows):
        pcm, rate = sf.read(args.backgrounds/item["path"], dtype="int16")
        assert rate == RATE
        raw = features(pcm)
        relative = Path("ambient")/item["split"]/f"{number:03d}.npy"
        (root/relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(root/relative, raw)
        ambient.append({"split": item["split"], "path": str(relative), "seconds": len(pcm)/RATE,
                        "source_sha256": item["sha256"]})
    (root/"ambient.json").write_text(json.dumps(ambient, indent=2)+"\n")
    human_link = root/"human-source"
    human_link.symlink_to(args.human_features.resolve(), target_is_directory=True)
    human = json.loads((args.human_features/"human.json").read_text())
    for row in human:
        row["path"] = str(Path("human-source")/row["path"])
    (root/"human.json").write_text(json.dumps(human, indent=2)+"\n")
    print(json.dumps({"sequences": len(index), "frames": offset,
                      "sources": sorted({r["engine"] for r in index})}), flush=True)


if __name__ == "__main__":
    main()
