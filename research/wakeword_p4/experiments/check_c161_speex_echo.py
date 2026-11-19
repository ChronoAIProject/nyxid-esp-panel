"""Replay the same stereo scenes through the synchronous SpeexDSP 1.2.1 echo canceller."""
import ctypes
from pathlib import Path

import numpy as np
from pocketsphinx import Decoder

from check_c161_compact_lexical import WORK
from check_c161_lexical_echo import OUT, RATE, STEM, cases
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts
from score_c161_lexical_echo import infer

DEB = "a418ba872a3c006502d5e70d855262507ee81b94227efef52512b5973f067e9d"
LIB = OUT/"speexdsp/usr/lib/x86_64-linux-gnu/libspeexdsp.so.1.5.2"


def binding():
    assert sha(OUT/"libspeexdsp1.deb") == DEB
    api = ctypes.CDLL(str(LIB.resolve()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    api.speex_echo_state_init.argtypes = [integer, integer]
    api.speex_echo_state_init.restype = ptr
    api.speex_echo_ctl.argtypes = [ptr,integer,ptr]
    api.speex_echo_cancellation.argtypes = [ptr,ptr,ptr,ptr]
    api.speex_echo_cancellation.restype = None
    api.speex_echo_state_destroy.argtypes = [ptr]
    api.speex_echo_state_destroy.restype = None
    return api


def cancel(api, mic, ref):
    # The synchronous API accepts the playback sample clock paired with the capture block.
    frame = 160
    state = api.speex_echo_state_init(frame,2048)
    assert state and len(mic) == len(ref) and len(mic)%frame==0
    try:
        assert api.speex_echo_ctl(state,24,ctypes.byref(ctypes.c_int(RATE))) == 0
        result=np.empty_like(mic)
        for i in range(0,len(mic),frame):
            api.speex_echo_cancellation(state,mic[i:].ctypes.data,ref[i:].ctypes.data,result[i:].ctypes.data)
        return result
    finally:
        api.speex_echo_state_destroy(state)


def main():
    api=binding()
    reader, identity=load_reader()
    parent=RESULTS/f"{STEM}.json"
    previous=read(parent)
    known={r["key"]:r for r in previous["records"]}
    assert previous["asr"]==identity
    spec=read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]
    model=RESULTS/spec["path"]
    assert sha(model)==previous["model_sha256"]["midlevel"]
    files=WORK/"q8/files"
    manifest=read(RESULTS/"4090-c161-lexical-p4-q8-packing.json")
    assert all(sha(files/k)==h for k,h in manifest["files"].items())
    decoder=Decoder(hmm=str(files/"am"),dict=str(files/"two.dict"),keyphrase="hey nicks",kws_threshold=1e-15,
                    loglevel="ERROR",lm=None)
    cmn=decoder.get_cmn()
    rows=[]
    for key,mic,ref,meta in cases():
        assert known.pop(key)["source"]==meta
        pcm=cancel(api,mic,ref)
        scored=infer(pcm,model,decoder,cmn)
        role=meta["role"]
        external=role in ("external_only","external_over_reply","simultaneous_mentions")
        # Allow one extra processing frame at each utterance end in this host front-end.
        end=(meta["external_end"] if external else meta["self_end"])+160
        hits=[h for h in scored["events"]["both_proposals"] if meta["start"]<=h["sample"]<end]
        row=dict(key=key,pcm_sha256=digest(pcm),detections=scored,hits=hits)
        if hits and role!="reply_only":
            begin=max(0,(hits[0]["sample"]+159)//160*160-32000)
            text=transcripts(reader,pcm[begin:end])
            cmd=meta["external_command"] if external else "turn on the kitchen lights."
            expected=normalized(cmd)
            row.update(begin=begin,end=end,transcript=text,exact_suffix=normalized(text["text"])[-len(expected):]==expected)
        rows.append(row)
    assert not known
    save(RESULTS/"4090-c161-speex-echo.json",dict(previous_sha256=sha(parent),deb_sha256=DEB,library_sha256=sha(LIB),
         version="Debian libspeexdsp1 1.2.1-1 amd64",frame=160,taps=2048,sample_rate=RATE,records=rows))


if __name__=="__main__":
    main()
