"""Paired clean-vs-room training of the existing 960 ms INT8 word classifier."""
from fractions import Fraction
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, sha

STEM = "4090-c161-room-word"
ORIGINAL = "4090-c161-word-window"


def room_render():
    rooms = []
    for row in read(RESULTS/"4090-mit-rir-audit.json")["records"]:
        if row["split"] != "training" or row["path"].split("/")[-1][:4] in {"h004","h012"}:
            continue
        # Spread the bank over measured training rooms, keeping the household test pair separate.
        if int(row["path"].split("/h")[1][:3]) % 9 != 1:
            continue
        p = Path(".local/wakeword/mit-rirs") / row["path"]
        assert sha(p) == row["sha256"]
        h, rate = sf.read(p)
        assert rate == 16000 and h.ndim == 1
        rooms.append((row, h / np.linalg.norm(h)))
    noises = []
    for row in read(RESULTS/"4090-backgrounds-manifest.json")["records"]:
        if row["split"] != "training" or row["seconds"] < 20 or not row["path"].endswith(("_3.wav","_4.wav","_5.wav")):
            continue
        p = Path(".local/wakeword/backgrounds")/row["path"]
        assert sha(p) == row["sha256"]
        wav, rate = sf.read(p, always_2d=True)
        ratio = Fraction(16000,rate)
        noises.append((row, resample_poly(wav.mean(axis=1),ratio.numerator,ratio.denominator)))

    def render(row, pcm, onset):
        seed = int.from_bytes(hashlib.sha256(row["key"].encode()).digest()[:8],"little")
        rng = np.random.default_rng(seed)
        if rng.integers(3) == 0:
            return pcm, dict(acoustic="dry")
        room, h = rooms[rng.integers(len(rooms))]
        noise, bg = noises[rng.integers(len(noises))]
        wet = fftconvolve(pcm.astype(float),h)[:len(pcm)]
        pos = int(rng.integers(len(bg)-len(pcm)+1))
        clip = bg[pos:pos+len(pcm)]
        # Use speech after the prefix. Both classes receive the same acoustic sampling procedure.
        speech = wet[onset:len(pcm)-8000]
        mask = np.abs(speech) > max(1,np.max(np.abs(speech)) * .035)
        assert mask.any()
        snr = int(rng.choice([0,6,12,18]))
        gain = np.sqrt(np.mean(speech[mask]**2)/max(1e-14,np.mean(clip[onset:len(pcm)-8000][mask]**2))) * 10**(-snr/20)
        mixture = wet + gain * clip
        scale = min(1, 30000 / max(1,np.abs(mixture).max()))
        return np.rint(scale*mixture).astype("<i2"),dict(acoustic="room",rir=room["path"],rir_sha256=room["sha256"],
                background=noise["path"],background_sha256=noise["sha256"],background_start=pos,snr_db=snr,scale=scale)
    return render


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "train":
        import train_c161_word_window_v1 as word
        word.train(STEM, room_render())
        fit = read(RESULTS/f"{STEM}-training-v1.json")
        old = read(RESULTS/f"{ORIGINAL}-training-v1.json")
        assert fit["positive_windows"]==old["positive_windows"] and fit["negative_windows"]==old["negative_windows"]
        for new, clean in zip(fit["training"],old["training"],strict=True):
            assert (new["key"],new["clean_stream_sha256"]) == (clean["key"],clean["stream_sha256"])
    elif action == "export":
        import export_c161_word_window_v1 as exporter
        exporter.STEM = STEM
        exporter.main()
    elif action == "replay":
        import replay_c161_word_window_v1 as replay
        replay.STEM = STEM
        replay.main()
    else:
        raise ValueError(action)
