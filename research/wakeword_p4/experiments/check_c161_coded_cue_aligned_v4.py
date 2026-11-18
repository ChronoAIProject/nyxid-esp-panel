"""Check the signed, narrowly timed detector and the merged C161 path for fixed-file playback."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import hashlib
import json

import numpy as np
import soundfile as sf

import check_c161_coded_cue_v4 as playback
import check_c161_coded_cue_band_v4 as band
from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import RESULTS, MODEL, MODEL_SHA, read, save, sha
from replay_c161_pcm_cues_v1 import detect, joined
from replay_c161_vowel_name_pair_v1 import binding as c161_binding
from select_c161_synthesized_cue_v2 import load_wav
from verify_c161_ambient_stream_v1 import run_stream
from audit_c161_history_paths_v1 import decode, times

STEM="4090-c161-coded-cue-aligned-v4"
HEADER=RESULTS.parent/"portable/c161_coded_cue_aligned_v4.h"


def background(lib,group):
    plan=read(RESULTS/f"4090-c161-history-gain-{group}-v1.json")
    records=[]
    for r in plan["records"]:
        if r["phase"]!="background":
            continue
        src=r["source"]
        path=Path(src.get("wav_path") or src["path"])
        digest=hashlib.sha256()
        with sf.SoundFile(path) as f:
            assert len(f)==src["samples"] and f.samplerate==16000 and f.channels==1 and f.subtype=="PCM_16"
            events=detect(lib,f.blocks(blocksize=96000,dtype="int16"),digest)
        assert digest.hexdigest()==src["pcm_sha256"]
        records.append(dict(source_id=src.get("source_id",path.stem),samples=src["samples"],pcm_sha256=digest.hexdigest(),events=events))
    save(RESULTS/f"{STEM}-background-{group}.json",dict(records=records,samples=sum(r["samples"] for r in records),
         header_sha256=sha(HEADER),coefficients_sha256=sha(band.COEFF)))
    print(group,len(records),sum(len(r["events"]) for r in records),flush=True)


def combine(cue):
    ref=read(RESULTS/f"{STEM}-commands.json")
    assert sha(MODEL)==MODEL_SHA
    native=c161_binding()
    rows=[]
    for r,(s,pcm) in zip(ref["records"],playback.command_streams(cue),strict=True):
        assert json.loads(json.dumps(s))==r["source"] and hashlib.sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
        codes,clock,_,_=run_stream(native,pcm)
        base=times(decode(np.array(codes[0],np.uint8),np.array(clock)))
        rows.append(dict(source=s,pcm_sha256=r["pcm_sha256"],c161=base,fixed=r["events"],merged=joined(base,r["events"])))
    save(RESULTS/f"{STEM}-merged.json",dict(c161_sha256=MODEL_SHA,commands_sha256=sha(RESULTS/f"{STEM}-commands.json"),records=rows))


if __name__=="__main__":
    spec=read(RESULTS/"4090-c161-coded-cue-v4-source.json")
    cue=load_wav(dict(path=playback.WAV,sha256=spec["wav_sha256"]))
    with tempfile.TemporaryDirectory() as tmp:
        lib=binding(Path(tmp)/"aligned.so",coded="aligned")
        with patch.multiple(playback,STEM=STEM,HEADER=HEADER,COEFF=band.COEFF),patch.multiple(band,STEM=STEM,HEADER=HEADER):
            for group in sys.argv[1:]:
                if group in ("primed","jenny","melo"):
                    background(lib,group)
                elif group=="fresh":
                    band.fresh(lib,cue)
                elif group=="merged":
                    combine(cue)
                else:
                    playback.run(lib,group,cue)
