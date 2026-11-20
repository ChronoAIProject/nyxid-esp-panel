"""Locate three parts of a known playback command inside one mic or decoded uplink recording."""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import wave
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

RATE = 16000
DECIMATION = 4
FRAME = 80  # 20 ms on the 4 kHz analysis clock.
WIDTH = 8*FRAME


def read_channel(path, index):
    try:
        reader = wave.open(str(path), "rb")
    except wave.Error:
        # The stereo handoff uses IEEE float WAV. Keep its original precision for hashes.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Reached EOF prematurely.*", category=wavfile.WavFileWarning)
            rate, samples = wavfile.read(path)
        if samples.dtype.kind != "f" or samples.dtype.itemsize not in (4, 8):
            raise ValueError("Expected PCM16, PCM24, float32 or float64 WAV")
        channels = samples.reshape(len(samples), 1) if samples.ndim == 1 else samples
        if not 0 <= index < channels.shape[1]:
            raise ValueError("Channel index outside WAV")
        track = channels[:, index]
        if not np.isfinite(track).all():
            raise ValueError("Non-finite audio samples")
        return track.astype(np.float64) * 32768, track.tobytes(), track.dtype.itemsize, rate
    with reader as w:
        rate, width = w.getframerate(), w.getsampwidth()
        channels, n = w.getnchannels(), w.getnframes()
        if width not in (2, 3) or w.getcomptype() != "NONE":
            raise ValueError("Expected a PCM16 or PCM24 WAV")
        if not 0 <= index < channels:
            raise ValueError("Channel index outside WAV")
        data = w.readframes(n)
        if len(data) != n*channels*width:
            raise ValueError("Truncated WAV")
    raw = np.frombuffer(data, dtype=f"V{width}").reshape(n, channels)[:,index].tobytes()
    if width == 2:
        pcm = np.frombuffer(raw, dtype="<i2")
    else:
        # Sign-extend packed little-endian PCM24 and retain fractional PCM16 units.
        lo, mid, hi = np.frombuffer(raw, dtype=np.uint8).reshape(-1,3).T.astype(np.int32)
        signed = lo + (mid << 8) + ((hi - 256*(hi >= 128)) << 16)
        pcm = signed / 256.0
    return pcm, raw, width, rate


def channel(path, index):
    pcm, _, _, rate = read_channel(path, index)
    if rate != RATE:
        raise ValueError("Expected a 16 kHz source")
    return pcm


def anchor(source, capture, offset, energies):
    x = source[offset:offset+WIDTH]
    x = x - np.mean(x)
    energy = x@x
    if energy < 1 or len(capture)<WIDTH:
        return dict(source_sample=offset*DECIMATION, status="insufficient_signal")
    corr = correlate(capture, x, mode="valid", method="fft")
    scores = np.zeros_like(energies)
    active = energies >= 1
    scores[active] = np.abs(corr[active]) / np.sqrt(energy*energies[active])
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
    if len(y) < WIDTH:
        return dict(status="inconclusive_short_capture")
    # Local sums retain the variance of a quiet tail after a loud passage.
    box = np.ones(WIDTH)
    sums = np.convolve(y, box, mode="valid")
    powers = np.convolve(y*y, box, mode="valid")
    energies = np.maximum(0., powers - sums*sums/WIDTH)
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
    parser.add_argument("--command-at", type=int, required=True, help="Zero-based first command sample at the playback WAV rate")
    parser.add_argument("--command-end", type=int, help="Exclusive last command sample at the playback WAV rate; defaults to EOF")
    parser.add_argument("--channel", type=int, default=0, help="Zero-based recording channel (MIC1 or mono defaults to 0)")
    args = parser.parse_args()
    source, source_bytes, width, source_rate = read_channel(args.playback,0)
    mic, mic_bytes, mic_width, mic_rate = read_channel(args.recording,args.channel)
    if mic_rate != RATE:
        divisor = gcd(RATE, mic_rate)
        mic = resample_poly(mic.astype(float), RATE//divisor, mic_rate//divisor)
    end = len(source) if args.command_end is None else args.command_end
    if not 0<=args.command_at<end<=len(source):
        parser.error("Command interval outside playback")
    command = source[args.command_at:end]
    if source_rate != RATE:
        divisor = gcd(RATE, source_rate)
        command = resample_poly(command.astype(float), RATE//divisor, source_rate//divisor)
    result = locate(command,mic)
    print(json.dumps(dict(playback=args.playback.name, recording=args.recording.name, command_at=args.command_at,
        command_end=end, channel=args.channel, analysis_rate=RATE, playback_rate=source_rate, playback_bits=8*width,
        recording_rate=mic_rate, recording_bits=8*mic_width,
        command_pcm_sha256=hashlib.sha256(source_bytes[args.command_at*width:end*width]).hexdigest(),
        channel_pcm_sha256=hashlib.sha256(mic_bytes).hexdigest(), **result),indent=2))


if __name__ == "__main__":
    main()
