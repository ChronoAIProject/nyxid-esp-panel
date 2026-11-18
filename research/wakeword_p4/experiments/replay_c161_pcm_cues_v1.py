"""Measure waveform anchors on paired cue trials, old ordinary language and full backgrounds."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from fit_c161_history_gain_v1 import PRIME, RESULTS, read, save, sha
from replay_c161_cue_matcher_v1 import cohort_data, ROOT

STEM = "4090-c161-pcm-cue"
LIB = Path(".local/wakeword/c161-vowel-name-v1/pcm-cue.so")


def binding():
    expected=read(RESULTS/f"{STEM}-anchors-v1.json")["header_sha256"]
    assert sha(ROOT/"portable/c161_pcm_cue_coefficients_v1.h")==expected
    subprocess.run(["c++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"frontend_probe/c161_pcm_cue_matcher_v1.cpp"),
                    "-o",str(LIB)],check=True)
    lib=ctypes.CDLL(str(LIB.resolve()))
    ptr=ctypes.c_void_p
    lib.make_pcm_cue.argtypes=[]
    lib.make_pcm_cue.restype=ptr
    lib.destroy_pcm_cue.argtypes=[ptr]
    lib.destroy_pcm_cue.restype=None
    lib.pcm_cue_bytes.argtypes=[]
    lib.pcm_cue_block.argtypes=[ptr,ptr,ctypes.c_int,ptr,ptr]
    return lib


def detect(lib, chunks, digest=None):
    state=lib.make_pcm_cue()
    if not state:
        raise MemoryError("PCM cue matcher allocation")
    position=-PRIME
    events=[]
    try:
        from itertools import chain
        for pcm in chain((np.zeros(PRIME,np.int16),),chunks):
            pcm=np.ascontiguousarray(pcm,dtype=np.int16)
            if digest is not None and position>=0:
                digest.update(pcm.tobytes())
            ends=np.empty(2+len(pcm)//32160,np.int32)
            ids=np.empty(len(ends),np.uint8)
            n=lib.pcm_cue_block(state,pcm.ctypes.data,len(pcm),ends.ctypes.data,ids.ctypes.data)
            assert 0<=n<=len(ends)
            events.extend([dict(t=position+int(t),cue_bits=int(k)) for t,k in zip(ends[:n],ids[:n],strict=True)])
            position+=len(pcm)
    finally:
        lib.destroy_pcm_cue(state)
    return events


def joined(base, extra):
    merged=[]
    for t in sorted([*base,*[v["t"] for v in extra]]):
        if not merged or t-merged[-1]>=32160:
            merged.append(t)
    return merged


def run(lib, cohort):
    rows=[]
    for key,r,pcm,base,pair in cohort_data(cohort):
        events=detect(lib,(pcm,))
        rows.append(dict(key=key,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),events=events,
                         original=base,pair=pair,original_with_cue=joined(base,events),pair_with_cue=joined(pair,events)))
    save(RESULTS/f"{STEM}-{cohort}-v1.json",dict(records=rows,state_bytes=lib.pcm_cue_bytes()))
    print(cohort,len(rows),flush=True)


def background(lib, cohort):
    rows=[]
    for x in read(RESULTS/f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
        if x["phase"]!="background":
            continue
        source=x["source"]
        path=Path(source.get("wav_path") or source["path"])
        digest=hashlib.sha256()
        with sf.SoundFile(path) as f:
            assert f.samplerate==16000 and f.channels==1 and f.subtype=="PCM_16" and len(f)==source["samples"]
            events=detect(lib,f.blocks(blocksize=96000,dtype="int16"),digest)
        assert digest.hexdigest()==source["pcm_sha256"]
        rows.append(dict(source_id=source.get("source_id",path.stem),samples=source["samples"],pcm_sha256=digest.hexdigest(),events=events))
    save(RESULTS/f"{STEM}-background-{cohort}-v1.json",dict(records=rows,samples=sum(r["samples"] for r in rows),
         events=sum(len(r["events"]) for r in rows)))
    print(cohort,len(rows),flush=True)


if __name__=="__main__":
    lib=binding()
    for cohort in sys.argv[1:]:
        if cohort in ("primed","jenny","melo"):
            background(lib,cohort)
        else:
            run(lib,cohort)
