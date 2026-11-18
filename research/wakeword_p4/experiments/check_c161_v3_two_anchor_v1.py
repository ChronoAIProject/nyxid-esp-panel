"""Replay the portable v3 waveform matcher on the identical physical input samples."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import check_c161_fixed_room_filter_v1 as replay
from fit_c161_history_gain_v1 import RESULTS, save, sha
from replay_c161_pcm_cues_v1 import detect

ROOT = RESULTS.parent
STEM = "4090-c161-v3-two-anchor"
HEADER = ROOT/"portable/c161_v3_two_anchor_v1.h"


def binding(out, coded=False):
    subprocess.run(["c++","-O3","-std=c++17","-fPIC","-shared",*({"v5":["-DNYX_CODED_CUE_V5"],"aligned":["-DNYX_ALIGNED_CODED_CUE_V4"],"band":["-DNYX_BAND_CODED_CUE_V4"],True:["-DNYX_CODED_CUE_V4"],False:[]}[coded]),str(ROOT/"frontend_probe/c161_fixed_playback_probe.cpp"),"-o",str(out)],check=True)
    lib = ctypes.CDLL(str(out.resolve()))
    ptr = ctypes.c_void_p
    lib.make_pcm_cue.restype = ptr
    lib.destroy_pcm_cue.argtypes = [ptr]
    lib.destroy_pcm_cue.restype = None
    lib.pcm_cue_block.argtypes = [ptr,ptr,ctypes.c_int,ptr,ptr]
    return lib


def run(lib,group):
    def score(pcm):
        return dict(events=detect(lib,[pcm]),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
    if group=="prospective":
        with patch.object(replay,"analyze",lambda _,pcm:score(pcm)):
            rows=replay.old_prospective(None)
    else:
        rows=[]
        for src,pcm in replay.inputs(group):
            r=score(pcm)
            assert r["pcm_sha256"]==src["pcm_sha256"]
            rows.append(dict(source=src,**r))
    save(RESULTS/f"{STEM}-{group}-v1.json",dict(header_sha256=sha(HEADER),state_bytes=lib.pcm_cue_bytes(),records=rows))
    print(group,len(rows),sum(len(r["events"]) for r in rows),flush=True)


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        lib=binding(Path(tmp)/"filter.so")
        for group in sys.argv[1:]:
            run(lib,group)
