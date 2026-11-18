"""Add two reproducible, band-limited noise bursts inside the frozen cue's /ks/."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav

STEM="4090-c161-coded-cue-v4"
WAV=RESULTS/"hye-nyx-c161-synthetic-cue-v4.wav"
COEFF=RESULTS.parent/"portable/c161_coded_cue_coefficients_v4.h"


def main():
    source=read(RESULTS/"4090-c161-suffix-cue-v3-source.json")
    pcm=load_wav(dict(path=RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav",sha256=source["wav_sha256"]))
    out=pcm.astype(float)
    starts=(11840,14400)
    length=1280
    templates=[]
    for i,start in enumerate(starts):
        rng=np.random.default_rng(74130+i)
        noise=rng.normal(size=length+512)
        band=sosfiltfilt(butter(4,[700,1580],fs=16000,btype="bandpass",output="sos"),noise)[256:-256]
        env=np.ones(length)
        env[:160]=np.sin(np.linspace(0,np.pi/2,160))**2
        env[-160:]=env[:160][::-1]
        signal=band*env*(.32*source["vowel_rms"]/np.sqrt(np.mean(band**2)))
        out[start:start+length]+=signal
        low=signal.reshape(-1,4).mean(1)
        q=np.rint(120*low/np.abs(low).max()).astype(np.int8)
        templates.append(q)
    assert np.max(np.abs(out))<=32767
    result=np.rint(out).astype("<i2")
    with WAV.open("xb") as f:
        sf.write(f,result,16000,format="WAV",subtype="PCM_16")
    lines=["#pragma once","#include <cstdint>","namespace nyx {","inline constexpr std::int8_t kCodedCueV4[2][320] = {"]
    for q in templates:
        lines.append("    {")
        for i in range(0,len(q),16):
            lines.append("        "+", ".join(map(str,q[i:i+16]))+",")
        lines.append("    },")
    lines.extend(["};","inline constexpr int kCodedCueEnergyV4[2] = {"+", ".join(str(int(q.astype(int)@q.astype(int))) for q in templates)+"};","}  // namespace nyx",""])
    with COEFF.open("x") as f:
        f.write("\n".join(lines))
    save(RESULTS/f"{STEM}-source.json",dict(v3_sha256=source["wav_sha256"],wav_sha256=sha(WAV),coefficients_sha256=sha(COEFF),
         samples=len(pcm),starts=starts,length=length,level_relative_to_vowel=.32,band_hz=[700,1580],seeds=[74130,74131],
         peak=int(abs(result.astype(int)).max()),changed_samples=int(np.count_nonzero(result!=pcm))))


if __name__=="__main__":
    main()
