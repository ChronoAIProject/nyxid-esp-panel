"""Locate three parts of a known playback command inside one mic or decoded uplink recording."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import correlate, resample_poly

RATE = 16000
DECIMATION = 4
FRAME = 80  # 20 ms on the 4 kHz analysis clock.
WIDTH = 8*FRAME


def channel(path, index):
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getsampwidth(), w.getcomptype()) != (RATE, 2, "NONE"):
            raise ValueError("Expected 16 kHz PCM16 WAV")
        channels, n = w.getnchannels(), w.getnframes()
        if not 0 <= index < channels:
            raise ValueError("Channel index outside WAV")
        data = w.readframes(n)
        if len(data) != n*channels*2:
            raise ValueError("Truncated WAV")
    return np.frombuffer(data, dtype="<i2").reshape(n,channels)[:,index]


def anchor(source, capture, offset, energies):
    x = source[offset:offset+WIDTH]
    x = x - np.mean(x)
    energy = x@x
    if energy < 1 or len(capture)<WIDTH:
        return dict(source_sample=offset*DECIMATION, status="insufficient_signal")
    corr = correlate(capture, x, mode="valid", method="fft")
    scores = np.abs(corr)/np.sqrt(np.maximum(energy*energies,1))
    k = int(np.argmax(scores))
    # Nearby room reflections may have the same origin; mark distant competing matches.
    other = scores.copy()
    other[max(0,k-FRAME):k+FRAME+1] = 0
    score, alt = float(scores[k]), float(other.max())
    accepted = score >= .45 and score-alt >= .06
    return dict(source_sample=offset*DECIMATION, captured_sample=k*DECIMATION,
                lag_samples=(k-offset)*DECIMATION, correlation=score, alternative=alt,
                polarity=int(np.sign(corr[k])), status="matched" if accepted else "inconclusive")


def locate(command, recorded):
    if not len(command) or not len(recorded):
        raise ValueError("Empty command or capture")
    x, y = (resample_poly(v.astype(float),1,DECIMATION) for v in (command,recorded))
    frame_energy = np.mean(np.pad(x*x,(0,-len(x)%FRAME)).reshape(-1,FRAME), axis=1)
    # Mark active source bounds at -35 dB relative to the strongest 20 ms frame.
    active = np.flatnonzero(frame_energy >= max(1., float(frame_energy.max())*10**(-3.5)))
    if not len(active):
        return dict(status="inconclusive_quiet_source")
    a, b = int(active[0])*FRAME, min(len(x),(int(active[-1])+1)*FRAME)
    if b-a < 3*WIDTH:
        return dict(status="inconclusive_short_source")
    power = np.r_[0., np.cumsum(y*y)]
    sums = np.r_[0., np.cumsum(y)]
    energies = np.maximum(0., power[WIDTH:]-power[:-WIDTH]-(sums[WIDTH:]-sums[:-WIDTH])**2/WIDTH)
    positions = (a, (a+b-WIDTH)//2, b-WIDTH)
    marks = {label:anchor(x,y,p,energies) for label,p in zip(("begin","middle","end"),positions)}
    good = [v for v in marks.values() if v["status"]=="matched"]
    result = dict(source_active_samples=[a*DECIMATION,min(len(command),b*DECIMATION)], anchors=marks)
    if len(good)<2:
        return dict(result,status="inconclusive_matches")
    # A constant offset approximates an unchanged playback path over one short utterance.
    # Reject unmatched timebases instead of declaring a clip from a large, inconsistent offset.
    lags = [m["lag_samples"] for m in good]
    if max(lags)-min(lags) > 320:
        return dict(result,status="inconclusive_clocks")
    lag = int(np.median(lags))
    bounds = [t+lag for t in result["source_active_samples"]]
    result.update(active_projection_samples=bounds, lag_spread_samples=max(lags)-min(lags))
    if bounds[0] < -320:
        status = "begin_outside_recording"
    elif bounds[1] > len(recorded)+320:
        status = "end_outside_recording"
    elif len(good) == 3:
        status = "three_windows_matched"
    else:
        status = "inconclusive_partial_matches"
    return dict(result,status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("playback", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--command-at", type=int, required=True, help="Zero-based first command sample in playback")
    parser.add_argument("--command-end", type=int, help="Exclusive last command sample in playback; defaults to EOF")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based recording channel (MIC1 or mono defaults to 0)")
    args = parser.parse_args()
    source, mic = channel(args.playback,0), channel(args.recording,args.channel)
    end = len(source) if args.command_end is None else args.command_end
    if not 0<=args.command_at<end<=len(source):
        parser.error("Command interval outside playback")
    command = source[args.command_at:end]
    result = locate(command,mic)
    print(json.dumps(dict(playback=args.playback.name, recording=args.recording.name, command_at=args.command_at,
        command_end=end, channel=args.channel, command_pcm_sha256=hashlib.sha256(command.tobytes()).hexdigest(),
        channel_pcm_sha256=hashlib.sha256(mic.tobytes()).hexdigest(), **result),indent=2))


if __name__ == "__main__":
    main()
