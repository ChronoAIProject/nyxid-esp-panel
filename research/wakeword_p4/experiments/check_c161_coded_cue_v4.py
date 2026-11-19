"""Replay the exact fixed v4 file, controls and complete backgrounds through the portable matcher."""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

from check_c161_fixed_room_filter_v1 import inputs
from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v4 import COEFF, STEM, WAV
from replay_c161_pcm_cues_v1 import detect
from select_c161_synthesized_cue_v2 import load_wav

HEADER=RESULTS.parent/"portable/c161_coded_cue_v4.h"


TARGET = "v4"


def command_streams(cue, noise_reference=None):
    # Fresh room channels on the same dry matched triplets. The seed, carrier, timing
    # and absolute noise are identical for v4, original v3, Willow and silence.
    plan=read(RESULTS/"4090-c161-v3-command-willow-plan-v1.json")
    noise_source = cue if noise_reference is None else noise_reference
    rooms=(
        ("digital",None,None,None,1.),
        ("near",[(0,.83),(92,-.17),(389,.27),(846,.14)],(110,6400),18,1.0008),
        ("far",[(0,.38),(241,-.32),(498,.27),(1145,-.22),(1731,.15),(2358,.11)],(270,3700),7,.9987),
    )
    for source in plan["records"]:
        if source["channel"]!="digital":
            continue
        dry=load_wav(source)
        assert hashlib.sha256(dry.tobytes()).hexdigest()==source["pcm_sha256"]
        start,end=source["start"],source["end"]
        for kind in ((TARGET,"v3") if source["target_id"]=="nyx" else (source["target_id"],)):
            target=cue if kind==TARGET else dry[start:end]
            stream=np.r_[dry[:start],target,dry[end:]]
            for name,taps,cut,snr,rate in rooms:
                pcm=stream
                if taps:
                    shifted=np.interp(np.arange(round(len(stream)/rate))*rate,np.arange(len(stream)),stream)
                    ir=np.zeros(max(t for t,_ in taps)+1)
                    for t,val in taps:
                        ir[t]=val
                    wet=sosfilt(butter(2,cut,fs=16000,btype="bandpass",output="sos"),fftconvolve(shifted,ir))*.53
                    seed=33601+source["scene"]+19*("af_aoede","am_eric","bf_lily","am_fenrir").index(source["voice"])
                    # Fixed noise level from the original cue; common leading sequence even for different file lengths.
                    noise=lfilter([1],[1,-.82],np.random.default_rng(seed).normal(size=len(wet)))
                    noise*=.53*np.sqrt(np.mean(noise_source.astype(float)**2))*np.sqrt(1-.82**2)*10**(-snr/20)
                    signal=wet+noise
                    assert np.abs(signal).max()<32768
                    pcm=np.rint(signal).astype("<i2")
                yield dict(role=kind,voice=source["voice"],scene=source["scene"],channel=name,rate=rate,
                           start=round(start/rate),end=round(end/rate),source_pcm_sha256=source["pcm_sha256"],
                           room_taps=taps,band_hz=cut,noise_db=snr),pcm


def measured(cue):
    # v3 and v4 have identical lengths. Apply identical h, gain and noise per response.
    v3=load_wav(dict(path=RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav",sha256=read(RESULTS/"4090-c161-coded-cue-v4-source.json")["v3_sha256"]))
    for i,r in enumerate(read(RESULTS/"4090-mit-rir-audit.json")["records"]):
        path=Path(".local/wakeword/mit-rirs")/r["path"]
        assert r["split"]=="training" and sha(path)==r["sha256"]
        h,rate=sf.read(path,dtype="float64")
        assert rate==16000
        h/=np.linalg.norm(h)
        waves=[fftconvolve(w,h) for w in (v3,cue)]
        gain=min(.65,.78*32767/max(np.abs(w).max() for w in waves))
        lead=14537
        noise=lfilter([1],[1,-.85],np.random.default_rng(52631+i).normal(size=lead+len(waves[0])+8000))
        noise*=gain*np.sqrt(np.mean(v3.astype(float)**2)/np.mean(noise**2))/10
        start=lead+int(np.abs(h).argmax())
        for role,w in zip(("v3",TARGET),waves,strict=True):
            for noisy in (False,True):
                signal=np.r_[np.zeros(lead),gain*w,np.zeros(8000)]
                if noisy:
                    signal+=noise
                yield dict(role=role,rir=r["path"],rir_sha256=r["sha256"],noisy=noisy,start=start,end=start+len(cue),
                           clipped=int(np.count_nonzero((signal>32767)|(signal< -32768)))),np.rint(signal).clip(-32768,32767).astype("<i2")


def run(lib,group,cue):
    rows=[]
    if group=="commands":
        recordings=command_streams(cue)
    elif group=="measured":
        recordings=measured(cue)
    else:
        recordings=inputs(group)
    for meta,pcm in recordings:
        events=detect(lib,[pcm])
        digest=hashlib.sha256(pcm.tobytes()).hexdigest()
        if "pcm_sha256" in meta:
            assert digest==meta["pcm_sha256"]
        row=dict(source=meta,pcm_sha256=digest,events=events)
        if group=="commands" and meta["role"]==TARGET and meta["scene"]==0 and meta["voice"]=="af_aoede":
            path=RESULTS/f"{STEM}-command-{meta['channel']}.wav"
            with path.open("xb") as f:
                sf.write(f,pcm,16000,format="WAV",subtype="PCM_16")
            assert detect(lib,(pcm[j:j+347] for j in range(0,len(pcm),347)))==events
            row.update(wav=path.name,wav_sha256=sha(path))
        rows.append(row)
    save(RESULTS/f"{STEM}-{group}.json",dict(header_sha256=sha(HEADER),coefficients_sha256=sha(COEFF),
         cue_sha256=sha(WAV),state_bytes=lib.pcm_cue_bytes(),records=rows))
    print(group,len(rows),sum(len(r["events"]) for r in rows),flush=True)


if __name__=="__main__":
    spec=read(RESULTS/f"{STEM}-source.json")
    assert sha(COEFF)==spec["coefficients_sha256"]
    cue=load_wav(dict(path=WAV,sha256=spec["wav_sha256"]))
    with tempfile.TemporaryDirectory() as tmp:
        lib=binding(Path(tmp)/"coded.so",coded=True)
        for group in sys.argv[1:]:
            run(lib,group,cue)
