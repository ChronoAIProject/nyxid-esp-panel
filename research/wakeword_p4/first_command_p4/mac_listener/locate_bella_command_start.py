"""Estimate the end of a known 1.45 s cue on a 16 kHz microphone clock at first open."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from measure_command_capture import channel


def locate_end(microphone, opened, cue):
    if len(cue) != 23200 or not (6400 <= opened <= len(microphone)):
        raise ValueError("Expected a 23,200-sample cue and at least 6,400 recorded samples before open")
    prefix = cue[:6400].astype(np.float64)
    first = max(0, opened - len(cue))
    available = microphone[first:opened].astype(np.float64)
    dot = correlate(available, prefix, mode="valid", method="fft")
    cumulative = np.r_[0., np.cumsum(available * available)]
    energy = cumulative[len(prefix):] - cumulative[:-len(prefix)]
    denominator = np.linalg.norm(prefix) * np.sqrt(np.maximum(0., energy))
    if not np.any(denominator > 0):
        raise ValueError("Cue prefix or recorded search window is silent")
    score = np.divide(np.abs(dot), denominator, out=np.zeros_like(dot), where=denominator > 0)
    offset = int(score.argmax())
    return first + offset + len(cue), float(score[offset])


def locate_available_prefix(microphone, opened, cue):
    if len(cue) != 23200 or not (6400 <= opened <= len(microphone)):
        raise ValueError("Expected a 23,200-sample cue and at least 6,400 recorded samples before open")
    first = max(0, opened - len(cue))
    observed = microphone[first:opened].astype(np.float64)
    template = cue.astype(np.float64)
    lengths = np.arange(len(observed), 6399, -1)
    # Nonnegative lags pair the beginning of the cue with each possible start in observed.
    dot = correlate(observed, template, mode="full", method="fft")[len(cue)-1:][:len(lengths)]
    norm = np.sqrt(np.cumsum(template*template)[lengths-1])
    if not np.any(norm > 0) or not np.any(observed):
        raise ValueError("Cue or microphone search window is silent")
    score = np.divide(np.abs(dot), norm, out=np.zeros_like(dot), where=norm > 0)
    offset = int(score.argmax())
    return first + offset + len(cue), float(score[offset])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cue", type=Path, help="Standalone three-triplet 16 kHz PCM WAV")
    parser.add_argument("recording", type=Path, help="16 kHz PCM16/PCM24 WAV containing the raw microphone")
    parser.add_argument("--open-sample", type=int, required=True, help="First open on this file's sample clock")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based microphone channel")
    parser.add_argument("--method", choices=("400ms", "available"), default="400ms",
                        help="Match the first 400 ms or all cue samples captured by first open")
    args = parser.parse_args()
    try:
        locate = locate_available_prefix if args.method == "available" else locate_end
        end, score = locate(channel(args.recording, args.channel), args.open_sample, channel(args.cue, 0))
    except ValueError as exc:
        parser.error(str(exc))
    metric = "projection_amplitude" if args.method == "available" else "correlation"
    print(json.dumps(dict(open_sample=args.open_sample, estimated_cue_end_sample=end,
                          **{metric:score}, sample_rate=16000)))


if __name__ == "__main__":
    main()
