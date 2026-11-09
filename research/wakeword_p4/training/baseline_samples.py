"""Create paired Mac TTS wake phrases for an offline Hi ESP/Hey Nyx comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from stream_data import render_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-eval", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"manifest.jsonl").exists():
        parser.error("Use a fresh output directory")
    originals = list(map(json.loads, (args.mac_eval/"manifest.jsonl").read_text().splitlines()))
    backgrounds = [str(args.backgrounds/r["path"]) for r in json.loads((args.backgrounds/"manifest.json").read_text())["records"] if r["split"] == "testing"]
    source = args.output/"source"
    source.mkdir()
    records = []
    for voice in sorted({r["voice"] for r in originals}):
        group = [r for r in originals if r["voice"] == voice]
        positive = [r for r in group if r["label"]]
        plan = [(r, "hey_nyx", None) for r in positive]
        # Spell the letters explicitly: macOS abbreviation expansion makes a
        # bare "ESP" inconsistent (the diagnostic ASR heard "especially").
        plan += [(r, "hi_esp", r["text"].replace("Hey", "Hi").replace("Nicks", "E S P").replace("Nix", "E S P")) for r in positive]
        plan += [(r, "none", None) for r in group if not r["label"] and r["text"] != "Hi ESP."]
        plan += [({"voice": voice, "text": text, "path": f"new-{i}.wav"}, "none", text)
                 for i, text in enumerate(("Hi SP.", "Hi ESB.", "Hi ASP.", "Hi EST.", "I see ESP.", "High speed."))]
        for index, (row, target, text) in enumerate(plan):
            if text is not None:
                aiff = source/f"{voice}-{index}.aiff"
                subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, timeout=30)
                pcm, rate = sf.read(aiff, dtype="float32")
                divisor = math.gcd(rate, 16000)
                pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
            else:
                text = row["text"]
                pcm, rate = sf.read(args.mac_eval/row["path"], dtype="float32")
                assert rate == 16000
            path = source/f"{voice}-{index}.wav"
            sf.write(path, pcm, 16000, subtype="PCM_16")
            blocks = len(pcm)//160
            rms = np.sqrt(np.mean(pcm[:blocks*160].reshape(-1,160)**2,axis=1))
            active = np.flatnonzero(rms>max(1e-6,float(rms.max())*.005))
            endpoint = min(len(pcm),(int(active[-1])+1)*160+960)
            # Positive pairs share augmentation randomness and speech RMS.
            pair = positive.index(row) if row in positive else index
            for variant in (0, 3):
                seed = int(hashlib.sha256(f"{voice}:{pair}:{variant}".encode()).hexdigest()[:8],16)
                wave, lead = render_stream({"path": path.name},source,np.random.default_rng(seed),variant,backgrounds)
                relative = Path("rendered")/f"{voice}-{index}-{variant}.wav"
                output = args.output/relative
                output.parent.mkdir(exist_ok=True)
                sf.write(output,np.clip(np.rint(wave),-32768,32767).astype(np.int16),16000,subtype="PCM_16")
                records.append({"path":str(relative),"voice":voice,"target":target,"text":text,
                                "variant":variant,"samples":len(wave),"keyword_end":lead+endpoint if target!="none" else None,
                                "sha256":hashlib.sha256(output.read_bytes()).hexdigest()})
        print(json.dumps({"voice":voice,"rendered":len(records)}),flush=True)
    (args.output/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (args.output/"generation.json").write_text(json.dumps({"clips":len(records),"disclosure":"All speech is AI-generated Mac system TTS.",
        "scope":"Paired development regression, not human acceptance. Existing Mac fixtures have informed model development. Quiet and noise/RIR renders are correlated."},indent=2)+"\n")


if __name__ == "__main__":
    main()
