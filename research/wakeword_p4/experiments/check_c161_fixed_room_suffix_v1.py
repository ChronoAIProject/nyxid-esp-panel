"""Confirm only spectral proposals against the original v2/v3 waveform suffix."""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import correlate

from check_c161_fixed_room_filter_v1 import inputs, old_prospective
import check_c161_fixed_room_filter_v1 as replay
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_fixed_room_filter_v1 import STEM


def difference(pcm):
    pairs = pcm[:len(pcm)//2*2].astype(np.int32).reshape(-1,2)
    low = np.trunc(pairs.sum(1)/2).astype(np.int32)
    return np.trunc(np.diff(low,prepend=0)/16).clip(-2047,2047).astype(np.float64)


def reference():
    plan = read(RESULTS/"4090-c161-cue-suffix-templates-v1.json")
    r = plan["records"][1]
    cue = load_wav(dict(path=RESULTS/"hye-nyx-c161-synthetic-cue-v2.wav",sha256=r["wav_sha256"]))
    template = difference(cue)[r["start_8khz"]:r["start_8khz"]+1024]
    assert template@template == r["energy"] and plan["threshold"] == .20
    return template


def confirm(pcm,events,template):
    readings = []
    for t in events:
        # Probe only the trailing 360 ms plus up to four subsequent 30 ms ticks.
        # Endpoints use even 16 kHz sample counts, and include only arrived PCM.
        lo = max(2*1024, t-5760)
        hi = min(len(pcm),t+1920)
        left = (lo+1)//2-1024
        if hi//2 < left+1024:
            readings.append(dict(proposal=t,sample=None,peak=None))
            continue
        # Include the previous pair for the exact first difference of the cropped stream.
        prior = max(0,left-1)
        signal = difference(pcm[2*prior:2*(hi//2)])[left-prior:]
        dot = correlate(signal,template,"valid",method="fft")
        energy = np.r_[0,np.cumsum(signal*signal)]
        denom = (energy[1024:]-energy[:-1024])*(template@template)
        corr = np.abs(dot)/np.sqrt(np.maximum(1.,denom))
        passed = np.flatnonzero((denom>0)&(corr>=.20))
        sample = None
        if len(passed):
            first_match = int((left+1024+passed[0])*2)
            sample = t+max(0,(first_match-t+479)//480)*480
            if sample > len(pcm):
                sample = None
        readings.append(dict(proposal=t,sample=sample,peak=float(corr.max())))
    return readings


def run(group):
    template = reference()
    report = RESULTS/f"{STEM}-{group}-v1.json"
    original = read(report)["records"]
    def score(pcm,r):
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==r["source"]["pcm_sha256"]
        checks = confirm(pcm,r["events"],template)
        return dict(source_sha256=r["source"]["pcm_sha256"],checks=checks,
                    confirmed=[x["sample"] for x in checks if x["sample"] is not None])
    if group=="prospective":
        records=[]
        def callback(_,pcm):
            current=score(pcm,original[len(records)])
            records.append(current)
            return {}
        with patch.object(replay,"analyze",callback):
            old_prospective(None)
    else:
        records=[score(pcm,r) for (_,pcm),r in zip(inputs(group),original,strict=True)]
    save(RESULTS/f"{STEM}-suffix-{group}-v1.json",dict(source_report_sha256=sha(report),records=records))
    print(group,len(records),sum(len(r["confirmed"]) for r in records),flush=True)


if __name__=="__main__":
    for group in sys.argv[1:]:
        run(group)
