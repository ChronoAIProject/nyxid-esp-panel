"""Read the fixed-cue clocks and levels from a 16 kHz PCM16 MR or LCD-X8 TDM WAV."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import wave

APP = Path(__file__).resolve().parent


def analyze(path, idle_frames=0):
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, frames, compression, _ = wav.getparams()
        if (rate, width, compression) != (16000, 2, "NONE") or channels not in (2, 4):
            raise ValueError("Expected uncompressed 16 kHz PCM16 MR (2 channels) or LCD-X8 TDM (4 channels)")
        if not frames or not 0 <= idle_frames <= frames:
            raise ValueError("The capture must have audio and contain the requested leading quiet interval")
        digest = hashlib.sha256()
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "read_mr"
            subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(APP/"main"),
                            "-I", str(APP.parents[1]/"portable"), str(APP/"read_mr_capture.cpp"),
                            "-o", str(binary)], check=True)
            # Spool the short event output so even hours of input cannot deadlock on a pipe.
            with tempfile.TemporaryFile("w+b") as events, subprocess.Popen(
                    [str(binary), str(channels), str(idle_frames)], stdin=subprocess.PIPE, stdout=events) as proc:
                count = 0
                with proc.stdin:
                    while chunk := wav.readframes(4096):
                        proc.stdin.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                if proc.wait() or count != frames * channels * width:
                    raise ValueError("Incomplete PCM frames in capture")
                events.seek(0)
                report = parse_events(events.read().decode().splitlines())
    return dict(file=path.name, pcm_sha256=digest.hexdigest(), sample_rate=rate, frames=frames,
                layout="MR" if channels == 2 else "MIC1,MIC3,MIC2,MIC4", **report)


def parse_events(lines):
    hits = {name:[] for name in ("mic_three", "ref_three", "ref_two", "ref_signed_two")}
    levels = {}
    activity = {}
    for line in lines:
        tag, name, *fields = line.split()
        if tag == "EVENT":
            hits[name].append(int(fields[0]))
        elif tag == "LEVEL":
            count, peak, rms, active, full = fields
            levels[name] = dict(samples=int(count), peak=int(peak), rms=float(rms),
                                magnitude_at_least_64=int(active), full_scale_samples=int(full))
        elif tag == "ACTIVITY":
            active, total = map(int, fields)
            activity[name] = dict(active_frames=active, complete_frames=total)
    # Compare only unambiguous fixed-code event pairs. A C161-only microphone wake has no such clock.
    delta = None
    if len(hits["ref_three"]) == len(hits["mic_three"]) == 1:
        delta = hits["ref_three"][0] - hits["mic_three"][0]
    return dict(events=hits, levels=levels, reference_activity=activity, ref_minus_mic_three_samples=delta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wav", type=Path)
    p.add_argument("--idle-frames", type=int, default=0,
                   help="Number of known quiet frames at the start for separate level statistics")
    args = p.parse_args()
    print(json.dumps(analyze(args.wav, args.idle_frames), indent=2))
