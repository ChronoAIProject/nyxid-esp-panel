"""Replay the same ordinary synthetic wake at the same clock before and after room convolution."""
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from c161_v5_household_rooms import PLAN, STEM
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from verify_c161_ambient_stream_v1 import run_stream


def acoustic_controls(case, z, manifest):
    voice, scene = case["voice"], case["scene"]
    signals = []
    for kind in ("ordinary", "command"):
        pcm, rate = sf.read(io.BytesIO(z.read(f"{voice}-{scene}-{kind}.wav")), dtype="int16")
        assert rate == 16000 and digest(pcm) == case["source_cue_sha256" if kind=="ordinary" else "source_command_sha256"]
        signals.append(pcm.astype(float))
    rir = Path(".local/wakeword/mit-rirs")/case["room"]
    assert sha(rir) == manifest["files"][str(rir)]
    h,rate=sf.read(rir)
    assert rate==16000
    h /= np.linalg.norm(h)
    start, clock = case["start"], case["command_at"]
    for mode in ("dry", "room"):
        wave = np.zeros(case["samples"])
        for offset, speech in zip((start, clock), signals, strict=True):
            data = speech if mode=="dry" else fftconvolve(speech, h)
            wave[offset:offset+len(data)] += data
        samples = np.rint(wave*case["common_gain"])
        assert np.abs(samples).max() < 32768
        pcm = samples.astype("<i2")
        yield mode, pcm


def main():
    scored = RESULTS/f"{STEM}-v1.json"
    baseline = read(scored)
    manifest = read(PLAN)
    assert baseline["plan_sha256"] == sha(PLAN) and sha(MODEL) == MODEL_SHA
    archive = RESULTS/f"{STEM}-dry.zip"
    assert sha(archive) == manifest["waveform_archive_sha256"]
    native, rows = binding(), []
    with ZipFile(archive) as z:
        for case in baseline["records"]:
            if case["role"] != "ordinary" or case["command_snr_db"] != 12:
                continue
            start, clock = case["start"], case["command_at"]
            row = dict(voice=case["voice"],scene=case["scene"],room=case["room"],start=start,command_at=clock,
                       noisy_pcm_sha256=case["pcm_sha256"],noisy_events=case["original_events"])
            for mode, pcm in acoustic_controls(case, z, manifest):
                codes, ticks, _, _ = run_stream(native,pcm)
                trace = np.asarray(codes[0],np.uint8)
                ticks = np.asarray(ticks)
                sums = np.convolve(trace,np.ones(5,dtype=np.int32),"valid")
                window = (ticks[4:]>=start) & (ticks[4:]<clock)
                events = times(decode(trace,ticks))
                row[mode] = dict(pcm_sha256=digest(pcm),original_events=events,peak_five_sum=int(sums[window].max()))
            rows.append(row)
    assert len(rows)==8
    save(RESULTS/f"{STEM}-quiet.json",dict(reference_sha256=sha(scored), plan_sha256=sha(PLAN), model_sha256=MODEL_SHA, records=rows))


if __name__=="__main__":
    main()
