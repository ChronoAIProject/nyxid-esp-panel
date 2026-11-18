"""Add a third independent, time-locked fricative segment to the fixed v4 playback file."""
import re
import numpy as np
from scipy.signal import butter, sosfiltfilt
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v4 import WAV as V4
from check_c161_coded_cue_band_v4 import COEFF as PAIR
from select_c161_synthesized_cue_v2 import load_wav

STEM="4090-c161-coded-cue-v5"
WAV=RESULTS/"hye-nyx-c161-synthetic-cue-v5.wav"
COEFF=RESULTS.parent/"portable/c161_coded_cue_coefficients_v5.h"


def main():
    origin=read(RESULTS/"4090-c161-coded-cue-v4-source.json")
    pcm=load_wav(dict(path=V4,sha256=origin["wav_sha256"]))
    level=read(RESULTS/"4090-c161-suffix-cue-v3-source.json")["vowel_rms"]*.32
    seed,start,length=74132,16320,1280
    rng=np.random.default_rng(seed)
    band=sosfiltfilt(butter(4,[700,1580],fs=16000,btype="bandpass",output="sos"),rng.normal(size=length+512))[256:-256]
    fade=np.sin(np.linspace(0,np.pi/2,160))**2
    env=np.r_[fade,np.ones(length-320),fade[::-1]]
    signal=band*env*level/np.sqrt(np.mean(band**2))
    out=pcm.astype(float)
    out[start:start+length]+=signal
    assert np.abs(out).max()<=32767
    with WAV.open("xb") as f:
        sf.write(f,np.rint(out).astype("<i2"),16000,format="WAV",subtype="PCM_16")
    text=PAIR.read_text()
    fir=[int(v) for v in re.search(r"kCueBandFir\[17\] = \{([^}]+)\}",text)[1].split(",")]
    low=signal.reshape(-1,4).mean(1)
    q=np.rint(120*low/max(abs(low))).astype(int)
    coeff=np.convolve(q,fir)[:320]
    third=np.rint(120*coeff/max(abs(coeff))).astype(int)
    marker="};\ninline constexpr int kCueBandEnergy[2]"
    assert text.count(marker)==1
    before,after=text.split(marker)
    before=before.replace("kCueBandPatterns[2]","kCueBandPatternsV5[3]").replace("kCueBandFir","kCueBandFirV5")
    rows=["    {"]+["        "+", ".join(map(str,third[i:i+16]))+"," for i in range(0,320,16)]+["    },"]
    after=after.replace("};",", "+str(third@third)+"};",1)
    with COEFF.open("x") as f:
        f.write(before+"\n".join(rows)+"\n};\ninline constexpr int kCueBandEnergyV5[3]"+after)
    save(RESULTS/f"{STEM}-source.json",dict(v4_sha256=sha(V4),v3_sha256=origin["v3_sha256"],wav_sha256=sha(WAV),
         coefficients_sha256=sha(COEFF),samples=len(pcm),third_start=start,third_length=length,third_seed=seed,
         level_relative_to_vowel=.32,peak=int(np.abs(out).max()),band_hz=[700,1580]))


if __name__=="__main__":
    main()
