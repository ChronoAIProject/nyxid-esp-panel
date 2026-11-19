"""Reconstruct the original measured-room PCM and locate the strongest coherent v5 chain."""
import hashlib
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import check_c161_coded_cue_v4 as playback
from audit_c161_history_paths_v1 import decode, times
from check_c161_coded_cue_v5 import COEFF, HEADER, STEM, WAV
from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_pcm_cues_v1 import detect, joined
from replay_c161_vowel_name_pair_v1 import binding as c161_binding
from select_c161_synthesized_cue_v2 import load_wav
from verify_c161_ambient_stream_v1 import run_stream

REPORT = RESULTS / f"{STEM}-measured.json"
OUT = RESULTS / f"{STEM}-room-diagnosis.json"


def coefficients(name):
    body = COEFF.read_text().split(name, 1)[1].split("= {", 1)[1].split("};", 1)[0]
    return np.array([int(s) for s in re.findall(r"-?\d+", body)])


def correlations(pcm):
    fir = coefficients("kCueBandFirV5")
    templates = coefficients("kCueBandPatternsV5").reshape(3, 320)
    blocks = pcm[:len(pcm) // 4 * 4].reshape(-1, 4).sum(axis=1)
    down = np.trunc(blocks / 16).astype(np.int64)
    filtered = np.trunc(np.convolve(down, fir)[:len(down)] / 128)
    windows = torch.as_tensor(np.pad(filtered, (319, 0)), device="cuda").unfold(0, 320, 1)
    patterns = torch.as_tensor(templates, device="cuda", dtype=torch.float64)
    dots = windows @ patterns.T
    norm = windows.norm(dim=1)[:, None] * patterns.norm(dim=1)
    return (dots / norm.clamp_min(1)).cpu().numpy().T


def strongest_chain(corr):
    # Anchor every possible middle peak. For either polarity pick the best first and
    # last score inside their respective +/-6 decimated-sample windows.
    middle = np.arange(646, corr.shape[1] - 486)
    offsets = np.arange(-6, 7)
    winners = []
    for polarity in (-1, 1):
        scores = polarity * corr
        first = middle - 640 + offsets[scores[0][middle[:, None] - 640 + offsets].argmax(1)]
        last = middle + 480 + offsets[scores[2][middle[:, None] + 480 + offsets].argmax(1)]
        strength = np.minimum.reduce((scores[0, first], scores[1, middle], scores[2, last]))
        i = strength.argmax()
        peaks = [first[i], middle[i], last[i]]
        winners.append(dict(min_corr=float(strength[i]), end_samples=[int(4*(p+1)) for p in peaks],
                            signed_corr=[float(corr[k, p]) for k, p in enumerate(peaks)]))
    return max(winners, key=lambda w: w["min_corr"])


def peak(corr, start, end):
    i = start + np.abs(corr[start:end]).argmax()
    return dict(end_sample=int(4*(i+1)), signed_corr=float(corr[i]))


def main():
    original = read(REPORT)
    assert (sha(WAV), sha(COEFF), sha(HEADER), sha(MODEL)) == (
        original["cue_sha256"], original["coefficients_sha256"], original["header_sha256"], MODEL_SHA)
    cue = load_wav(dict(path=WAV, sha256=original["cue_sha256"]))
    # Include one successful room as a positive check of the independent clock reconstruction.
    control = next(r["source"]["rir"] for r in original["records"] if r["source"]["role"] == "v5" and r["events"])
    records = []
    native = c161_binding()
    with tempfile.TemporaryDirectory() as tmp, patch.object(playback, "TARGET", "v5"):
        lib = binding(Path(tmp)/"frozen.so", coded="v5")
        for old, (meta, pcm) in zip(original["records"], playback.measured(cue), strict=True):
            assert old["source"] == meta and old["pcm_sha256"] == hashlib.sha256(pcm.tobytes()).hexdigest()
            if meta["role"] != "v5" or (old["events"] and meta["rir"] != control):
                continue
            events = detect(lib, [pcm])
            assert events == old["events"]
            corr = correlations(pcm)
            chain = strongest_chain(corr)
            assert bool(events) == (chain["min_corr"] >= .3)
            # The original time labels use the strongest RIR tap. Search within
            # +/-80 ms of each expected template end for independent peak diagnostics.
            peaks = []
            for row, end in zip(corr, (13120, 15680, 17600), strict=True):
                center = (meta["start"] + end) // 4
                peaks.append(peak(row, center - 320, center + 321))
            codes, clocks, _, _ = run_stream(native, pcm)
            base = times(decode(np.asarray(codes[0], np.uint8), np.asarray(clocks)))
            records.append(dict(**old, independent_peaks=peaks, best_chain=chain, c161=base, merged=joined(base, events)))
            print(meta["rir"], meta["noisy"], chain, base, flush=True)
    save(OUT, dict(original_sha256=sha(REPORT), cue_sha256=sha(WAV), model_sha256=MODEL_SHA,
                   header_sha256=sha(HEADER), coefficients_sha256=sha(COEFF), device=torch.cuda.get_device_name(),
                   verified_pcm_records=len(original["records"]), control_room=control, records=records))


if __name__ == "__main__":
    main()
