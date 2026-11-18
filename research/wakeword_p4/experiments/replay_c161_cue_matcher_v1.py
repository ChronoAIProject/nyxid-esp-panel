"""Replay the fixed two-template matcher alongside authenticated original C161 events."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.signal import lfilter
import soundfile as sf
import micro_features_cpp  # registers symbols required by the upstream MicroFrontend

from audit_c161_history_paths_v1 import arbitrate
from fit_c161_history_gain_v1 import PRIME, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import cases
from select_c161_synthesized_cue_v2 import load_wav
from verify_c161_ambient_stream_v1 import BUILD

STEM = "4090-c161-cue-matcher"
LIB = Path(".local/wakeword/c161-vowel-name-v1/cue.so")
ROOT = RESULTS.parent


def binding():
    assert read(RESULTS/"4090-c161-ambient-history-pair-live-v1.json")["build"]["library_sha256"] == sha(BUILD/"stream.so")
    assert read(RESULTS/"4090-c161-cue-templates-v1.json")["header_sha256"] == sha(ROOT/"portable/c161_cue_templates_v1.h")
    subprocess.run(["c++","-O2","-std=c++17","-fPIC","-shared",str(ROOT/"frontend_probe/c161_cue_matcher_v1.cpp"),
                    str(BUILD/"stream.so"),"-o",str(LIB)], check=True)
    lib = ctypes.CDLL(str(LIB.resolve()))
    ptr = ctypes.c_void_p
    lib.make_cue.argtypes = []
    lib.make_cue.restype = ptr
    lib.destroy_cue.argtypes = [ptr]
    lib.destroy_cue.restype = None
    lib.cue_block.argtypes = [ptr,ptr,ctypes.c_int,ctypes.c_bool,ptr,ptr]
    lib.cue_state_bytes.argtypes = []
    return lib


def score(lib, pcm):
    native = lib.make_cue()
    if not native:
        raise MemoryError("C161 frontend allocation")
    try:
        full = np.r_[np.zeros(PRIME,np.int16), pcm]
        padded = np.pad(full, (0,-len(full)%160))
        hops = len(padded)//160
        code = np.empty((hops//3+2,2),np.float32)
        events = np.empty(len(code),np.uint8)
        n = lib.cue_block(native,padded.ctypes.data,hops,True,code.ctypes.data,events.ctypes.data)
        assert n == (max(0,hops-2)+2)//3
        clock = np.minimum((np.arange(n)+2)*480,len(full))-PRIME
        return code[:n], events[:n], clock
    finally:
        lib.destroy_cue(native)


def fixed():
    rows = read(RESULTS/"4090-c161-synthetic-cue-validation-v2.json")["records"]
    training = read(RESULTS/"4090-chatterbox-daily-training-admission-v1.json")["admitted"]
    by_hash = {r["wav"]["sha256"]:r["wav"] for r in training}
    voices = ("af_bella","am_onyx","bf_isabella","bm_fable")
    for i,r in enumerate(rows):
        cue, rate = sf.read(RESULTS/f"hye-nyx-c161-synthetic-cue-{r['cue']}.wav", dtype="int16")
        assert rate == 16000
        prefix = load_wav(by_hash[r["history_sha256"]])
        source = cue.astype(float)
        if r["room"]:
            ir = np.zeros(721)
            ir[[0,240,720]] = [1,.22,-.12]
            source = np.convolve(source,ir)
        full = np.r_[prefix,np.zeros(r["gap"]),source*10**(r["gain_db"]/20),np.zeros(8000)]
        if r["nominal_snr"] is not None:
            seed = 57638+voices.index(r["voice"])*1000+r["gap"]
            noise = lfilter([1],[1,-.9],np.random.default_rng(seed).normal(size=len(full)))
            rms = np.sqrt(np.mean(cue.astype(float)**2))
            full += noise / np.sqrt(np.mean(noise**2)) * rms / 10**(r["nominal_snr"]/20)
        pcm = np.rint(full).clip(-32768,32767).astype("<i2")
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
        yield str(i), r, pcm, r["original"], r["pair"]


def cohort_data(name):
    if name == "cue":
        yield from fixed()
    elif name == "official":
        for i,r in enumerate(read(RESULTS/"4090-c161-cue-willow-v1.json")["records"]):
            yield str(i),r,load_wav(r),r["original_events"],r["pair_events"]
    else:
        prior = read(RESULTS/f"4090-c161-vowel-name-{name}-v1.json")["records"]
        for r, (info,pcm) in zip(prior,cases(name),strict=True):
            assert info["key"]==r["key"] and hashlib.sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
            yield r["key"], r, pcm, r["events"]["c161"]["original"],r["events"]["c161"]["pair"]


def join(base, extra, clock):
    def to_events(values):
        return [dict(step=int(np.searchsorted(clock,t))+1,relative_sample=t) for t in values]
    emitted,_ = arbitrate(to_events(base),to_events(extra))
    return [e["relative_sample"] for e in emitted]


def evaluate(lib, name):
    out, arrays = [], {}
    for key,r,pcm,base,pair in cohort_data(name):
        scores, flags, clock = score(lib, pcm)
        extra = clock[flags.astype(bool)].tolist()
        arrays[key] = scores
        meta = {k:v for k,v in r.items() if k not in ("events","files")}
        out.append(dict(source=meta, max_scores=scores.max(axis=0).tolist(), template=extra,
                        original=base,pair=pair,original_with_template=join(base,extra,clock),
                        pair_with_template=join(pair,extra,clock)))
    archive = RESULTS/f"{STEM}-{name}-scores-v1.npz"
    with archive.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-{name}-v1.json", dict(state_bytes=lib.cue_state_bytes(),scores_sha256=sha(archive),records=out))
    print(name,len(out),flush=True)


if __name__ == "__main__":
    native = binding()
    for cohort in sys.argv[1:]:
        evaluate(native,cohort)
