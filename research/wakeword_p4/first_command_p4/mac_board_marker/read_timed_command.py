"""Read a normalized English command from timestamped words on a decoded 16 kHz send."""
import argparse
import json
import math
from pathlib import Path
import re

from crop_received_command import sample_offset


def select(words, offset):
    # Work in PCM sample units, including an exact tie at the word midpoint.
    first = len(words)
    for i, (word, start, end) in enumerate(words):
        if not isinstance(word, str) or not (math.isfinite(end) and 0 <= start <= end):
            raise ValueError("Each word needs text and valid start/end times in seconds")
        if first == len(words) and round(start * 16000) + round(end * 16000) >= 2 * offset:
            first = i
    text = " ".join(w for w, _, _ in words[first:])
    return first, re.findall(r"[a-z]+", text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("words", type=Path, help='JSON array of {"word": "...", "start": seconds, "end": seconds}')
    parser.add_argument("--opened", required=True, type=int, help="first open on the 16 kHz mic clock")
    parser.add_argument("--first", required=True, type=int, help="first FIFO sample on the same clock")
    parser.add_argument("--prefix", required=True, type=int, help="decoded samples before the first captured packet")
    args = parser.parse_args()
    offset = sample_offset(args.opened, args.first, args.prefix)
    with args.words.open() as stream:
        items = json.load(stream)
    first, kept = select([(w["word"], w["start"], w["end"]) for w in items], offset)
    print(json.dumps(dict(boundary_sample=offset, first_retained_word=first, text=" ".join(kept))))


if __name__ == "__main__":
    main()
