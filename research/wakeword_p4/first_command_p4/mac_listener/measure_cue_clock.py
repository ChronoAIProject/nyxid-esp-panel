"""Compare M/R capture sample counts with seven scheduled starts of the frozen v5 cue."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import wave

from read_mr_capture import analyze

RESULTS = Path(__file__).resolve().parents[2] / "results"
CUE = RESULTS / "hye-nyx-c161-v5-early-80ms.wav"
CUE_SHA256 = "cc59349578cbca6c63ba13a45f7303d9de98d11560a31799b0a884e227eabf0f"
RATE = 16000
# Unequal intervals make a missed cue plus an extra event easier to identify.
STARTS = tuple(t * RATE for t in (1, 9, 20, 30, 38, 50, 61))
RESIDUAL_LIMIT = 16


def playback():
    wav = CUE.read_bytes()
    if hashlib.sha256(wav).hexdigest() != CUE_SHA256:
        raise ValueError("Handoff WAV SHA256 mismatch")
    with wave.open(io.BytesIO(wav)) as src:
        if src.getparams()[:4] != (1, 2, RATE, 17600):
            raise ValueError("Unexpected handoff WAV format")
        cue = src.readframes(src.getnframes())
    signal = bytearray(63 * RATE * 2)
    for start in STARTS:
        signal[2*start:2*start+len(cue)] = cue
    data = io.BytesIO()
    with wave.open(data, "wb") as out:
        out.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        out.writeframes(signal)
    return data.getvalue()


def fit(events):
    if len(events) != len(STARTS):
        return dict(status="inconclusive_event_count", count=len(events))
    # The intercept absorbs recording lead-in, propagation, and the in-cue match time.
    x_mean, y_mean = sum(STARTS)/len(STARTS), sum(events)/len(events)
    slope = (sum((x-x_mean)*(y-y_mean) for x, y in zip(STARTS, events)) /
             sum((x-x_mean)**2 for x in STARTS))
    residuals = [round(y - y_mean - slope*(x-x_mean), 3) for x,y in zip(STARTS, events)]
    info = dict(residual_samples=residuals, span_samples=events[-1]-events[0])
    if max(abs(v) for v in residuals) > RESIDUAL_LIMIT:
        return dict(info, status="inconclusive_irregular_spacing")
    return dict(info, status="measured", capture_samples_per_source_sample=slope,
                excess_capture_ppm=round((slope-1)*1000000, 3))


def measure(path):
    report = analyze(path)
    mic, ref = (report["events"][k] for k in ("mic_three", "ref_three"))
    m, r = fit(mic), fit(ref)
    out = dict(capture=report, source_cue_sha256=CUE_SHA256,
               playback_sha256=hashlib.sha256(playback()).hexdigest(), source_start_samples=STARTS,
               maximum_fit_residual_samples=RESIDUAL_LIMIT, mic=m, reference=r)
    if m["status"] == r["status"] == "measured":
        out["ref_minus_mic_samples"] = [b-a for a, b in zip(mic, ref)]
        out["ref_minus_mic_ppm_on_source_clock"] = round(
            (r["capture_samples_per_source_sample"] - m["capture_samples_per_source_sample"]) * 1000000, 3)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "measure"))
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        data = playback()
        args.wav.write_bytes(data)
        print(hashlib.sha256(data).hexdigest(), args.wav)
    else:
        print(json.dumps(measure(args.wav), indent=2))
