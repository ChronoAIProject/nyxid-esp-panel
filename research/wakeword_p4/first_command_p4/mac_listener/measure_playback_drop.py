"""Compare the electrical playback references in two 16 kHz captures on their own OPEN clocks."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

RATE = 16000
PRE = 12800
POST = 16960
SEARCH = 960
FRAME = 160


def read_reference(path, opened):
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        if (w.getframerate(), w.getsampwidth(), w.getcomptype()) != (RATE, 2, "NONE") or channels not in (2, 4):
            raise ValueError("Expected 16 kHz PCM16 stereo M/R or four-slot LCD-X8")
        if opened < PRE or opened + SEARCH + 3*FRAME > w.getnframes():
            raise ValueError("Expected 800 ms of capture before OPEN and at least 90 ms after")
        start = opened - PRE
        count = min(w.getnframes()-start, PRE+POST)
        w.setpos(start)
        pcm = w.readframes(count)
    if len(pcm) != count*channels*2:
        raise ValueError("Truncated capture")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    info = dict(file=Path(path).name, channels=channels, opened=opened, first_sample=start,
                frames_read=count, interleaved_window_sha256=hashlib.sha256(pcm).hexdigest())
    return samples[1::channels], info


def rms(v):
    mean = sum(v)/len(v)
    return math.sqrt(sum((x-mean)**2 for x in v)/len(v))


def align(control, trial, start, end):
    # Compare against every sample lag. Sparse speech samples limit runtime on a stock Mac.
    step = 17
    y = trial[start:end:step]
    n = len(y)
    sy = sum(y)
    yy = sum(i*i for i in y) - sy*sy/n
    scores = []
    for lag in range(-SEARCH, SEARCH+1):
        x = control[start+lag:end+lag:step]
        sx = sum(x)
        xx = sum(i*i for i in x) - sx*sx/n
        xy = sum(a*b for a,b in zip(x,y)) - sx*sy/n
        correlation = xy/math.sqrt(xx*yy) if xx > 0 and yy > 0 else 0.
        scores.append((correlation, xy/xx if xx > 0 else 0.))
    idx = max(range(len(scores)), key=lambda j: abs(scores[j][0]))
    corr, gain = scores[idx]
    alternative = max(abs(score[0]) for j,score in enumerate(scores) if abs(j-idx) > 16)
    return dict(lag=idx-SEARCH, correlation=corr, gain=gain, alternative=alternative, rms=rms(y))


def compare(control_path, trial_path, control_open, trial_open):
    control, first = read_reference(control_path, control_open)
    trial, second = read_reference(trial_path, trial_open)
    # Disjoint pre-OPEN windows check that both runs carry the same speech at a stable lag.
    matches = [align(control, trial, PRE-9600, PRE-5120), align(control, trial, PRE-5120, PRE-640)]
    out = dict(control=first, trial=second, alignment=matches)
    if any(abs(m["correlation"]) < .85 or abs(m["correlation"])-m["alternative"] < .03
           or abs(m["lag"]) == SEARCH or m["rms"] < 64 for m in matches):
        return dict(out, status="inconclusive_alignment")
    a,b = matches
    if abs(a["lag"]-b["lag"]) > 2 or not .8 <= a["gain"]/b["gain"] <= 1.25:
        return dict(out, status="inconclusive_alignment")
    lag = b["lag"]
    gain = abs(b["gain"])
    active_floor = max(64., .10*b["rms"])
    frames = []
    end = min(PRE+RATE, len(trial), len(control)-lag)
    for t in range(PRE, end-FRAME+1, FRAME):
        original = gain*rms(control[t+lag:t+lag+FRAME])
        received = rms(trial[t:t+FRAME])
        state = "quiet_control"
        if original >= active_floor:
            state = "attenuated" if received <= original*(10**(-12/20)) else "above_limit"
        frames.append(dict(start=t-PRE, expected_rms=round(original,2), observed_rms=round(received,2), state=state))
    # Identify the first three consecutive observable frames at least 12 dB lower.
    onset = next((i for i in range(len(frames)-2)
                  if all(f["state"] == "attenuated" for f in frames[i:i+3])), None)
    out.update(frames=frames, observable_floor_rms=active_floor, control_lag_samples=lag)
    if onset is None:
        observed = [f["state"] for f in frames if f["state"] != "quiet_control"]
        # Isolated low frames and gaps in the control cannot establish a sustained drop.
        status = "inconclusive_level"
        if len(observed) >= 3 and all(state == "above_limit" for state in observed):
            status = "no_sustained_drop_observed"
        return dict(out, status=status)
    last = next((f["start"] for f in reversed(frames[:onset]) if f["state"] == "above_limit"), None)
    return dict(out, status="drop_observed", first_low_frame_ms=frames[onset]["start"]/16,
                last_above_limit_frame_ms=None if last is None else last/16)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control", type=Path, help="Continuous panel playback capture")
    p.add_argument("trial", type=Path, help="Paired trial with the same panel playback phase")
    p.add_argument("--control-open", type=int, required=True, help="Zero-based OPEN sample in control WAV")
    p.add_argument("--trial-open", type=int, required=True, help="Zero-based OPEN sample in trial WAV")
    args = p.parse_args()
    print(json.dumps(compare(args.control, args.trial, args.control_open, args.trial_open), indent=2))
