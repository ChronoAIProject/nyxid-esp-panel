"""Make a 100 ms pre-open listening copy of a complete decoded mono PCM16 send."""
import argparse
from contextlib import suppress
import json
from pathlib import Path
import wave


def sample_offset(opened, first, prefix):
    if first < 0 or opened < first or prefix < 0:
        raise ValueError("Expected 0 <= first FIFO sample <= first open, and a nonnegative sent prefix")
    return prefix + max(0, opened - first - 1600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("received", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opened", required=True, type=int, help="first open, in 16 kHz mic samples")
    parser.add_argument("--first", required=True, type=int, help="first captured FIFO sample on the same clock")
    parser.add_argument("--prefix", required=True, type=int, help="decoded samples sent before the first captured packet")
    args = parser.parse_args()
    offset = sample_offset(args.opened, args.first, args.prefix)
    with wave.open(str(args.received), "rb") as src:
        if (src.getframerate(), src.getnchannels(), src.getsampwidth()) != (16000, 1, 2):
            parser.error("Expected a mono 16 kHz PCM16 WAV")
        if offset >= src.getnframes():
            parser.error("The measured offset falls beyond the received audio")
        src.setpos(offset)
        count = src.getnframes() - offset
        pcm = src.readframes(count)
        if len(pcm) != count * 2:
            parser.error("Received WAV has a truncated payload")
    with args.output.open("xb") as f:
        try:
            with wave.open(f, "wb") as dest:
                dest.setparams((1, 2, 16000, count, "NONE", "not compressed"))
                dest.writeframes(pcm)
        except BaseException:
            with suppress(FileNotFoundError):
                args.output.unlink()
            raise
    print(json.dumps(dict(start_sample=offset, samples=count)))


if __name__ == "__main__":
    main()
