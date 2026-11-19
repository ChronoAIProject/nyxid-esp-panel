"""Estimate a handed-off command boundary on the microphone clock of one Mac board recording."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np

PACK = Path(__file__).resolve().parents[2] / "results/c161-balanced-connected-commands.zip"
PACK_SHA = "00184ed5175fdd3bf738cbafc4c6d192a259c770b140acf3233291d8e29b6466"
RATE = 16000
WINDOW = 3200


def channel(wav, permitted):
    channels = wav.getnchannels()
    if (wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) != (RATE, 2, "NONE") or channels not in permitted:
        raise ValueError("Expected uncompressed 16 kHz PCM16 with the requested channel layout")
    count = wav.getnframes()
    data = wav.readframes(count)
    if not count or len(data) != count * channels * 2:
        raise ValueError("Incomplete or empty recording")
    return np.frombuffer(data, dtype="<i2").reshape(-1, channels)[:, 0].copy()


def track(index):
    contents = PACK.read_bytes()
    if sha256(contents).hexdigest() != PACK_SHA:
        raise ValueError("The balanced playback archive differs from the handoff")
    with ZipFile(BytesIO(contents)) as pack:
        entries = json.loads(pack.read("manifest.json"))["records"]
        entry = next((r for r in entries if r["index"] == index), None)
        if entry is None:
            raise ValueError(f"Unknown track index {index}")
        item = entry["tracks"]["balanced_louder"]
        data = pack.read(item["file"])
        if sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Playback file digest mismatch")
        with wave.open(BytesIO(data), "rb") as wav:
            return channel(wav, (1,)), entry["command_at"], item["file"]


def emphasize(pcm):
    x = pcm.astype(np.float64)
    return x - 0.85 * np.r_[0., x[:-1]]


def templates(dry, command_at):
    # Choose one 200 ms window in each half of the active command. Use no samples from the cue.
    suffix = emphasize(dry[command_at:])
    active = np.flatnonzero(np.abs(suffix) >= np.max(np.abs(suffix)) * .10)
    if not len(active):
        raise ValueError("Command is silent")
    first, last = int(active[0]), int(active[-1] + 1)
    middle = (first + last) // 2
    for start, end in ((first, middle), (middle, last)):
        if end - start < WINDOW:
            raise ValueError("Too little speech to locate two command windows")
        offsets = range(start, end - WINDOW + 1, 80)
        pos = max(offsets, key=lambda at: np.var(suffix[at:at+WINDOW]))
        ref = suffix[pos:pos+WINDOW]
        yield pos, ref - ref.mean()


def locate(mic, dry, command_at, open_sample=None):
    if open_sample is not None and not 0 <= open_sample <= len(mic):
        raise ValueError("OPEN must use the microphone recording's sample clock")
    if len(mic) < WINDOW:
        return dict(status="inconclusive_short_recording")
    x = emphasize(mic)
    count = 1 << (len(x) + WINDOW - 2).bit_length()
    spectrum = np.fft.rfft(x, count)
    cumulative = np.r_[0., np.cumsum(x)]
    squared = np.r_[0., np.cumsum(x*x)]
    sums = cumulative[WINDOW:] - cumulative[:-WINDOW]
    energy = np.maximum(squared[WINDOW:] - squared[:-WINDOW] - sums*sums/WINDOW, 0.)
    matches = []
    for offset, ref in templates(dry, command_at):
        cross = np.fft.irfft(spectrum * np.fft.rfft(ref[::-1], count), count)[WINDOW-1:len(x)]
        norm = np.sqrt(energy * np.dot(ref, ref))
        scores = np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 1.)
        abs_scores = np.abs(scores)
        peak = int(abs_scores.argmax())
        # Exclude the local reflection cluster when testing for a second playback.
        margin = 640
        runner = max(float(abs_scores[:max(0, peak-margin)].max(initial=0.)),
                     float(abs_scores[peak+margin:].max(initial=0.)))
        matches.append(dict(suffix_offset=offset, peak=peak, correlation=float(scores[peak]),
                            alternative=runner, command_boundary=peak-offset))
    result = dict(status="inconclusive", windows=matches)
    times = [m["command_boundary"] for m in matches]
    # Room filtering can reverse the strongest correlation lobe between two speech windows.
    # Compare arrival clocks and the absolute ambiguity margins independently of their signs.
    if (min(abs(m["correlation"]) for m in matches) < .25
            or any(abs(m["correlation"])-m["alternative"] < .06 for m in matches)
            or max(times) - min(times) > 192 or min(times) < 0):
        return result
    boundary = int(round(sum(times)/2))
    result.update(status="matched", estimated_command_sample=boundary, window_spread_samples=max(times)-min(times))
    if open_sample is not None:
        result.update(open_sample=open_sample, estimated_lead_ms=(boundary-open_sample)*1000/RATE)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="One isolated trial, 16 kHz PCM16 M/R or LCD-X8 four-slot TDM")
    parser.add_argument("--track", required=True, type=int, help="Balanced playback archive index (e.g. 005)")
    parser.add_argument("--open-sample", type=int, help="UART OPEN converted to elapsed samples from WAV start")
    args = parser.parse_args()
    dry, command_at, member = track(args.track)
    with wave.open(str(args.wav), "rb") as wav:
        mic = channel(wav, (2, 4))
    report = locate(mic, dry, command_at, args.open_sample)
    print(json.dumps(dict(capture=args.wav.name, track=member, mic_sha256=sha256(mic.tobytes()).hexdigest(),
                          **report), indent=2))


if __name__ == "__main__":
    main()
