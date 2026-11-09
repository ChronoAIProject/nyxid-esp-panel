"""Add Mac system TTS, separating voice families and excluding old regressions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from phrases import HARD_NEGATIVES

FAMILIES = {"Aman", "Rishi", "Tara", "Albert", "Fred", "Junior", "Kathy", "Ralph",
            "Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley"}
OLD_REGRESSION = {"Samantha", "Daniel", "Karen", "Moira", "Tessa", "Serena"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root/"manifest.jsonl").exists():
        parser.error("Use a fresh output directory")
    voices = {}
    for line in subprocess.check_output(["say","-v","?"],text=True).splitlines():
        match = re.match(r"^(.*?)\s+(en_[A-Z]{2})\s+#",line)
        if match:
            name,locale = match.groups()
            family = name.split(" (")[0]
            if family in FAMILIES and family not in OLD_REGRESSION:
                voices.setdefault(family,[]).append((name,locale))
    families = sorted(voices)
    np.random.default_rng(20261110).shuffle(families)
    if len(families)<10:
        raise RuntimeError("Too few installed voice families for the reserved split")
    partitions = {"training":families[:-4],"validation":families[-4:-2],"testing":families[-2:]}
    records = []
    for split,group in partitions.items():
        for family in group:
            for preset,(voice,locale) in enumerate(voices[family]):
                texts = [(text,1) for text in ("Hey Nicks.","Hey, Nicks.","Hey Nicks!","Hey Nicks?",
                                               "Hey Nix.","Hey, Nix!","Hey Nyx.","Hey, Nyx.")]
                texts += [(text,0) for text in HARD_NEGATIVES]
                for index,(text,label) in enumerate(texts):
                    stem=Path(split)/f"{family}-{preset}"/f"{index:03d}"
                    aiff=root/stem.with_suffix(".aiff")
                    aiff.parent.mkdir(parents=True,exist_ok=True)
                    subprocess.run(["say","-v",voice,"-o",str(aiff),text],check=True,timeout=30)
                    pcm,rate=sf.read(aiff,dtype="float32")
                    divisor=math.gcd(rate,16000)
                    pcm=resample_poly(pcm,16000//divisor,rate//divisor)
                    path=root/stem.with_suffix(".wav")
                    sf.write(path,pcm,16000,subtype="PCM_16")
                    records.append({"path":str(path.relative_to(root)),"engine":"macos-say","split":split,
                        "speakers":[family],"preset":voice,"locale":locale,"text":text,"label":label,
                        "samples":len(pcm),"keyword_end":len(pcm) if label else None,
                        "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            print(json.dumps({"family":family,"split":split,"clips":len(records)}),flush=True)
    (root/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (root/"generation.json").write_text(json.dumps({"engine":"macos-say","family_partitions":partitions,
        "clips":len(records),"excluded_old_regression_families":sorted(OLD_REGRESSION),
        "limits":"Synthetic system voices; family IDs prevent accent presets of one family crossing splits, but do not establish independent human speakers. Old five-voice regressions are excluded from training. No audio speed transforms."},indent=2)+"\n")


if __name__ == "__main__":
    main()
