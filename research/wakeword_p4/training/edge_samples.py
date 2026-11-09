"""Generate normal-speed English fixtures with a sixth, independent TTS engine.

Uses edge-tts 7.2.3 and a saved voice catalogue. No audio/device credentials are
sent: requests contain only the fixed public test phrases and voice names.
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import re

import edge_tts
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from phrases import HARD_NEGATIVES

POSITIVES = ("Hey Nicks.", "Hey, Nicks.", "Hey Nicks!", "Hey Nicks?",
             "Hey Nix.", "Hey, Nix!", "Hey Nyx.", "Hey, Nyx.")


async def generate(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    voices = json.loads(args.voices_json.read_text())
    voices = [r for r in voices if r["Locale"].startswith("en-")]
    # Multilingual and ordinary variants of one named voice share a split.
    def family(row):
        return re.sub(r"(?:Multilingual)?Neural$", "", row["ShortName"].split("-", 2)[2])
    families = sorted({family(r) for r in voices})
    np.random.default_rng(20261111).shuffle(families)
    reserved = max(2, len(families)//6)
    if len(families) < 12:
        raise ValueError("Too few English voice families")
    partitions = {"training": families[:-2*reserved], "validation": families[-2*reserved:-reserved],
                  "testing": families[-reserved:]}
    args.output.mkdir(parents=True)
    (args.output/"voices.json").write_text(json.dumps(voices, indent=2)+"\n")
    (args.output/"plan.json").write_text(json.dumps({"family_partitions": partitions,
        "catalogue_sha256": hashlib.sha256(args.voices_json.read_bytes()).hexdigest(),
        "rate": "+0%", "pitch": "+0Hz", "volume": "+0%", "positive_texts": POSITIVES,
        "negative_texts": HARD_NEGATIVES}, indent=2)+"\n")
    semaphore = asyncio.Semaphore(2)
    completed = 0

    async def one(voice, split, text, label, index):
        nonlocal completed
        stem = Path(split)/voice["ShortName"]/f"{index:03d}"
        mp3 = args.output/stem.with_suffix(".mp3")
        wav = args.output/stem.with_suffix(".wav")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        async with semaphore:
            for attempt in range(3):
                try:
                    await asyncio.wait_for(edge_tts.Communicate(text, voice=voice["ShortName"],
                        rate="+0%", pitch="+0Hz", volume="+0%").save(str(mp3)), timeout=60)
                    break
                except (asyncio.TimeoutError, edge_tts.exceptions.NoAudioReceived, OSError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
            pcm, rate = sf.read(mp3, dtype="float32")
            if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
                raise ValueError("TTS returned invalid audio")
            divisor = math.gcd(rate, 16000)
            pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
            sf.write(wav, pcm, 16000, subtype="PCM_16")
            completed += 1
            if completed % 28 == 0:
                print(json.dumps({"completed": completed, "total": len(voices)*(len(POSITIVES)+len(HARD_NEGATIVES))}), flush=True)
            return {"path": str(wav.relative_to(args.output)), "engine": "edge-tts", "split": split,
                    "speakers": [family(voice)], "preset": voice["ShortName"], "locale": voice["Locale"],
                    "text": text, "label": label, "samples": len(pcm), "keyword_end": len(pcm) if label else None,
                    "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}

    jobs = [one(voice, split, text, label, index)
            for split, group in partitions.items()
            for voice in sorted(voices, key=lambda r: r["ShortName"]) if family(voice) in group
            for index, (text, label) in enumerate([(t, 1) for t in POSITIVES]+[(t, 0) for t in HARD_NEGATIVES])]
    records = await asyncio.gather(*jobs)
    (args.output/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (args.output/"generation.json").write_text(json.dumps({"engine": "edge-tts", "package_version": "7.2.3",
        "source": "https://github.com/rany2/edge-tts", "clips": len(records), "voice_families": len(families),
        "presets": len(voices), "family_partitions": partitions,
        "disclosure": "All speech is AI-generated Microsoft Edge TTS. Corpus is private research input, not published audio.",
        "limits": "Fixed named-voice families, not verified independent human speakers. Neutral normal-speed synthesis; no PCM speed transformation. MP3 synthesis resampled to 16 kHz. Split precedes synthesis and augmentation."}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
