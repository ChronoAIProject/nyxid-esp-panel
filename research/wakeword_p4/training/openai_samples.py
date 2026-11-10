"""Generate disclosed AI speech fixtures with voice-disjoint dataset splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import time
import urllib.error
import urllib.request

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from phrases import HARD_NEGATIVES

VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova",
          "onyx", "sage", "shimmer", "verse", "marin", "cedar"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if args.key_file:
        for line in args.key_file.read_text().splitlines():
            match = re.match(r"^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
            if match:
                fields = shlex.split(match.group(1), comments=True)
                key = fields[0] if fields else None
    if not key:
        raise SystemExit("OPENAI_API_KEY is not configured")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"generation.json").exists(), "Use a fresh completed-run directory"
    completed = {}
    if (root/"manifest.jsonl").exists():
        completed = {r["path"]: r for r in map(json.loads, (root/"manifest.jsonl").read_text().splitlines())}
    ids = np.random.default_rng(726).permutation(VOICES).tolist()
    splits = {"training": ids[:9], "validation": ids[9:11], "testing": ids[11:]}
    plan = []
    for split, voices in splits.items():
        for voice in voices:
            samples = [(text, 1, style) for style in ("neutral", "warm", "direct")
                       for text in ("Hey Nicks.", "Hey Nix!", "Hey Nicks?")]
            samples += [(s, 0, "neutral") for s in HARD_NEGATIVES[:8]+HARD_NEGATIVES[-4:]]
            for number, (text, label, style) in enumerate(samples):
                plan.append({"path": f"{split}/{voice}/{number:03d}.wav", "split": split,
                             "speakers": [voice], "engine": "openai-tts", "text": text,
                             "label": label, "style": style})
    def generate(row):
        if row["path"] in completed:
            return completed[row["path"]]
        instruction = f"Read exactly the supplied English text, with {row['style']} natural conversational delivery. Do not add an introduction or any other words."
        body = json.dumps({"model": "gpt-4o-mini-tts", "voice": row["speakers"][0],
                           "input": row["text"], "instructions": instruction,
                           "response_format": "wav", "speed": 1.0}).encode()
        for attempt in range(3):
            request = urllib.request.Request("https://api.openai.com/v1/audio/speech", data=body,
                headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    audio = response.read()
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503) or attempt == 2:
                    raise RuntimeError(f"Speech request failed: HTTP {error.code}; response body omitted") from None
                time.sleep(2**attempt)
        pcm, rate = sf.read(io.BytesIO(audio), dtype="float32")
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Invalid speech waveform")
        divisor = math.gcd(rate, 16000)
        pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
        if not (.25 <= len(pcm)/16000 <= 5):
            raise ValueError(f"Unexpected duration: {row['path']}")
        path = root/row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, pcm, 16000, subtype="PCM_16")
        return {**row, "samples": len(pcm), "keyword_end": len(pcm) if row["label"] else None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (root/"manifest.jsonl").open("a") as log:
        for number, row in enumerate(pool.map(generate, plan), 1):
            if row["path"] not in completed:
                log.write(json.dumps(row)+"\n"); log.flush()
                completed[row["path"]] = row
            if number % 21 == 0:
                print(json.dumps({"clips": number, "total": len(plan),
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    report = {"engine": "OpenAI gpt-4o-mini-tts", "voices": splits,
              "clips": len(completed), "generated_at": datetime.now(timezone.utc).isoformat(),
              "disclosure": "All fixture audio is AI-generated speech, not human recordings.",
              "docs": "https://developers.openai.com/api/docs/guides/text-to-speech",
              "limits": "Server model alias and output sampling can change; hashes identify the actual generated corpus."}
    (root/"generation.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
