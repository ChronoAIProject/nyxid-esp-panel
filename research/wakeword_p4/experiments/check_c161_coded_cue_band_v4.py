"""Check paired filtering of received PCM and the two fixed code templates."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import sys

import numpy as np
from scipy.signal import firwin

import check_c161_coded_cue_v4 as original
from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v4 import WAV
from select_c161_synthesized_cue_v2 import load_wav
from replay_c161_pcm_cues_v1 import detect

STEM="4090-c161-coded-cue-band-v4"
PORT=RESULTS.parent/"portable"
COEFF=PORT/"c161_coded_cue_band_coefficients_v4.h"
HEADER=PORT/"c161_coded_cue_band_v4.h"


def coefficients():
    import re
    text=original.COEFF.read_text().split(" = {",1)[1].split("};",1)[0]
    raw=np.array([int(x) for x in re.findall(r"-?\d+",text)]).reshape(2,320)
    fir=np.rint(firwin(17,[620,1680],fs=4000,pass_zero=False)*128).astype(int)
    refs=[]
    for r in raw:
        q=np.convolve(r,fir)[:320]
        refs.append(np.rint(q*120/max(abs(q))).astype(int))
    lines=["#pragma once","#include <cstdint>","namespace nyx {",
           "inline constexpr int kCueBandFir[17] = {"+", ".join(map(str,fir))+"};",
           "inline constexpr std::int8_t kCueBandPatterns[2][320] = {"]
    for r in refs:
        lines.append("    {")
        for i in range(0,320,16):
            lines.append("        "+", ".join(map(str,r[i:i+16]))+",")
        lines.append("    },")
    lines.extend(["};","inline constexpr int kCueBandEnergy[2] = {"+", ".join(str(r@r) for r in refs)+"};","}  // namespace nyx",""])
    with COEFF.open("x") as f:
        f.write("\n".join(lines))


def fresh(lib,cue, target="v4", seed_base=937757):
    from scipy.signal import butter, fftconvolve, sosfilt, lfilter
    from check_c161_event_transcripts_v1 import source_pcm
    bot=dict(source_pcm("bot"))
    bot_plan=read(RESULTS/"4090-c161-bot-replies-v1.json")
    rows=[]
    for i in range(24):
        rng=np.random.default_rng(seed_base+i)
        voice=("af_kore","bm_fable")[i%2]
        b=next(x for x in bot_plan["records"] if x["key"]==voice+"_digital")
        a=b["spans"][i//2]
        history=bot[b["key"]][a["start"]:a["end"]]
        start=len(history)+int(rng.integers(1000,10000))
        end=start+len(cue)
        # The following command is the identical first-phoneme-aligned excerpt across the pair.
        src=read(RESULTS/"4090-c161-signed-forward-plan-v1.json")["records"][2*(i%12)]
        assert src["target_id"]=="nyx"
        cmd=load_wav(src)[src["first_phone"]:]
        ir=np.zeros(2241)
        ir[0]=rng.uniform(.28,.85)
        lags=rng.choice(np.arange(50,len(ir)),18,replace=False)
        ir[lags]=rng.normal(0,.22,len(lags))*np.exp(-lags/1550)
        rate=rng.uniform(.997,1.003)
        cuts=[rng.uniform(100,360),rng.uniform(3100,6100)]
        tail=np.r_[np.zeros(start),cue,cmd]
        count=round(len(tail)/rate)+len(ir)-1
        noise=lfilter([1],[1,-.78],rng.normal(size=count))
        snr=rng.uniform(6,22)
        noise*=.60*np.sqrt(np.mean(cue.astype(float)**2)/np.mean(noise**2))*10**(-snr/20)
        for role,wave in ((target,cue),("silence",np.zeros_like(cue))):
            dry=np.r_[history,np.zeros(start-len(history)),wave,cmd]
            resampled=np.interp(np.arange(round(len(dry)/rate))*rate,np.arange(len(dry)),dry)
            analog=.60*sosfilt(butter(2,cuts,fs=16000,btype="bandpass",output="sos"),fftconvolve(resampled,ir))+noise
            pcm=np.rint(analog).clip(-32768,32767).astype("<i2")
            r=dict(case=i,seed=seed_base+i,bot=voice,utterance=a["text"],command=src["carrier"],role=role,rate=rate,
                   snr=snr,cutoffs=cuts,start=round(start/rate),end=round(end/rate),clipped=int((abs(analog)>32767).sum()),
                   pcm_sha256=original.hashlib.sha256(pcm.tobytes()).hexdigest(),events=detect(lib,[pcm]))
            rows.append(r)
    save(RESULTS/f"{STEM}-fresh.json",dict(header_sha256=sha(HEADER),coefficients_sha256=sha(COEFF),cue_sha256=sha(WAV),records=rows))


if __name__=="__main__":
    spec=read(RESULTS/"4090-c161-coded-cue-v4-source.json")
    cue=load_wav(dict(path=WAV,sha256=spec["wav_sha256"]))
    with tempfile.TemporaryDirectory() as temp:
        lib=binding(Path(temp)/"coded.so",coded="band")
        with patch.multiple(original,HEADER=HEADER,COEFF=COEFF,STEM=STEM):
            for group in sys.argv[1:]:
                if group=="fresh":
                    fresh(lib,cue)
                else:
                    original.run(lib,group,cue)
