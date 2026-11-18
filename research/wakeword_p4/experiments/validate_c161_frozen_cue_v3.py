"""Prospective simulated acoustic playback of two fixed files after v3 was frozen."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
import replay_c161_pcm_cues_v1 as matcher
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

SPEC = RESULTS/"4090-c161-frozen-cue-prospective-plan-v1.json"
OUT = RESULTS/"4090-c161-frozen-cue-prospective-v1.json"


def plan():
    pools=read(RESULTS/"4090-chatterbox-daily-training-admission-v1.json")["admitted"]
    rows=[]
    for i,(voice,group) in enumerate([("af_bella","conversation"),("bm_fable","ordinary_request"),("am_onyx","number_quantity")]):
        row=[r for r in pools if r["request"]["voice"]==voice and r["request"]["group"]==group][-2]
        rows.append(dict(voice=voice,text=row["request"]["text"],wav=row["wav"],gap=5213+i*723))
    files={name:sha(RESULTS/f"hye-nyx-c161-synthetic-cue-{name}.wav") for name in ("v2","v3")}
    save(SPEC,dict(waves=files,history=rows,rooms=[
        dict(name="desk", taps=[[0,.85],[139,-.15],[377,.11],[927,.09]], cutoff=[120,6100]),
        dict(name="room", taps=[[0,.53],[183,.31],[601,-.23],[991,.16],[1783,.09]],cutoff=[220,4200])],
        levels_db=[-2,-8], snr_db=[10,20], ratios_ppm=[-750,1200], seed=19337, source_sha256=sha(Path(__file__)),
        matcher_header=sha(RESULTS.parent/"portable/c161_cue_suffix_v1.h")))


def run():
    spec=read(SPEC)
    assert spec["source_sha256"]==sha(Path(__file__))
    matcher.LIB=matcher.LIB.with_name("pcm-suffix.so")
    lib=matcher.binding(True)
    ref=binding()
    cues={}
    for name,h in spec["waves"].items():
        cues[name]=load_wav(dict(path=str(RESULTS/f"hye-nyx-c161-synthetic-cue-{name}.wav"),sha256=h))
    rms=np.sqrt(np.mean(cues["v2"].astype(float)**2))
    records=[]
    for history_index,history in enumerate(spec["history"]):
        prefix=np.r_[load_wav(history["wav"]),np.zeros(history["gap"])]
        for room in spec["rooms"]:
            ir=np.zeros(room["taps"][-1][0]+1)
            for delay,amp in room["taps"]:ir[delay]=amp
            sos=butter(2,room["cutoff"],fs=16000,btype="bandpass",output="sos")
            for ppm in spec["ratios_ppm"]:
                speed=1+ppm/1_000_000
                resampled={k:np.interp(np.arange(round(len(v)/speed))*speed,np.arange(len(v)),v) for k,v in cues.items()}
                for gain in spec["levels_db"]:
                    for snr in spec["snr_db"]:
                        for name,x in resampled.items():
                            source=sosfilt(sos,np.convolve(x,ir))*10**(gain/20)
                            full=np.r_[prefix,source,np.zeros(8000)]
                            rawnoise=lfilter([1],[1,-.85],np.random.default_rng(spec["seed"]+history_index).normal(size=len(full)))
                            full+=rawnoise*rms/np.sqrt(np.mean(rawnoise**2))/10**(snr/20)
                            pcm=np.rint(full).clip(-32768,32767).astype("<i2")
                            extra=matcher.detect(lib,(pcm,))
                            codes,clock,pair,_=live.run_stream(ref,pcm)
                            base=times(decode(np.asarray(codes[0],np.uint8),np.asarray(clock)))
                            records.append(dict(cue=name,history=history_index,room=room["name"],gain_db=gain,nominal_snr=snr,
                                ppm=ppm,start=len(prefix),end=len(prefix)+len(x),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                clipped=int(((full>32767)|(full< -32768)).sum()),original=base,pair=times(pair),
                                confirmed=extra,original_with_cue=matcher.joined(base,extra),pair_with_cue=matcher.joined(times(pair),extra)))
    assert len(records)==96
    save(OUT,dict(plan_sha256=sha(SPEC),records=records))


if __name__=="__main__":
    import sys
    {"plan":plan,"run":run}[sys.argv[1]]()
