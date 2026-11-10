"""Generate a speaker-disjoint Hey Nyx dataset with the pinned Piper generator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
import torch
import torchaudio

from phrases import HARD_NEGATIVES

CHECKPOINT_SHA = "e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6"
SUFFIXES = ["", "", "", ", what time is it?", ", turn on the light.",
            ", can you hear me?", ", check the volume."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto",
                        help="Upstream selects CUDA inputs automatically when available")
    args = parser.parse_args()
    source = args.upstream.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "manifest.jsonl").exists(), "Use a fresh dataset directory"
    checkpoint = source / "models/en_US-libritts_r-medium.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA
    sys.path.insert(0, str(source))
    from generate_samples import generate_audio, get_phonemes, audio_float_to_int16
    torch.set_num_threads(4)
    torch.manual_seed(726)
    rng = np.random.default_rng(726)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device)
    config = json.loads((source / "models/en_US-libritts_r-medium.pt.json").read_text())
    assert config["num_speakers"] == 904 and config["audio"]["sample_rate"] == 22050
    voice = config["espeak"]["voice"]
    resample = torchaudio.transforms.Resample(22050, 16000)
    speakers = rng.permutation(904)
    partitions = {"training": speakers[:672], "validation": speakers[672:784], "testing": speakers[784:]}
    plan = []
    for split, ids in partitions.items():
        count = args.per_class if split == "training" else 480
        for label in (1, 0):
            for index in range(count):
                keyword = "Hey Nyx"
                text = keyword + rng.choice(SUFFIXES) if label else str(rng.choice(HARD_NEGATIVES))
                if label and text == keyword:
                    text += "."
                pair = rng.choice(ids, size=2, replace=False).tolist()
                if rng.random() < .25:
                    pair[1] = pair[0]
                plan.append({"split": split, "label": label, "index": index,
                             "text": text, "speakers": pair})
    # Keep each batch in one split, but shuffle labels/texts within the split.
    ordered = []
    for split in partitions:
        group = [p for p in plan if p["split"] == split]
        rng.shuffle(group)
        ordered.extend(group)
    started = time.monotonic()
    counts = {}
    with (root / "manifest.jsonl").open("w") as manifest, torch.inference_mode():
        for at in range(0, len(ordered), args.batch):
            batch = ordered[at:at + args.batch]
            phonemes = [get_phonemes(voice, config, item["text"])[0] for item in batch]
            lengths = [len(p) for p in phonemes]
            padded = [p + [1] * (max(lengths) - len(p)) for p in phonemes]
            length_scale = float(rng.choice([.9, 1., 1.1]))
            audio, durations = generate_audio(
                model, torch.LongTensor([b["speakers"][0] for b in batch]),
                torch.LongTensor([b["speakers"][1] for b in batch]), padded,
                float(rng.choice([.25, .5, .75])), .667, .8, length_scale, None)
            durations = durations.cpu().numpy()[:, 0]
            waves = resample(audio.cpu()).numpy()[:, 0]
            for i, item in enumerate(batch):
                end = min(len(waves[i]), int(np.ceil(durations[i, :lengths[i]].sum() * 16000 / 22050)))
                pcm = audio_float_to_int16(waves[i, :end])
                assert len(pcm) > 1600 and np.max(np.abs(pcm.astype(np.int32))) > 1000
                keyword_end = None
                if item["label"]:
                    prefix = get_phonemes(voice, config, "Hey Nyx")[0][:-1]
                    assert phonemes[i][:len(prefix)] == prefix, (item["text"], prefix, phonemes[i])
                    keyword_end = int(np.ceil(durations[i, :len(prefix)].sum() * 16000 / 22050))
                    assert 0 < keyword_end <= len(pcm)
                relative = Path(item["split"]) / str(item["label"]) / f'{item["index"]:05d}.wav'
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(pcm.astype("<i2").tobytes())
                item.update(path=str(relative), samples=len(pcm), keyword_end=keyword_end,
                            length_scale=length_scale, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                manifest.write(json.dumps(item) + "\n")
                counts[item["split"] + "/" + str(item["label"])] = counts.get(item["split"] + "/" + str(item["label"]), 0) + 1
            manifest.flush()
            if (at // args.batch) % 10 == 0:
                print(json.dumps({"generated": at + len(batch), "total": len(ordered), "seconds": round(time.monotonic() - started, 1)}), flush=True)
            if device == "mps":
                torch.mps.empty_cache()
    report = {"phrase": "Hey Nyx", "sample_rate": 16000, "device": device,
              "counts": counts, "speaker_partitions": {s: ids.tolist() for s, ids in partitions.items()},
              "checkpoint_sha256": CHECKPOINT_SHA, "seconds": round(time.monotonic() - started, 1),
              "scope": "Synthetic Piper voices; real human keyword recall remains unmeasured"}
    (root / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
