"""Match the selected cue to the original WAV's peak or RMS before replay."""
from pathlib import Path

import numpy as np
import soundfile as sf

import check_marker_gain_music as study

ch, fit = study.ch, study.fit
FIT = ch.RESULTS/(fit.STEM+"-levels.json")
WAV = {name:ch.RESULTS/f"hye-nyx-v5-marker-gains-{name}.wav" for name in ("rms","peak")}


def main():
    dry = study.tracks()
    scales = {}
    extra = {}
    a,b = dry["v5"].astype(float),dry["markers"].astype(float)
    for name,measure in (("rms",np.linalg.norm),("peak",lambda x:np.abs(x).max())):
        gain = measure(a)/measure(b)
        out = np.rint(b*gain).astype("<i2")
        sf.write(WAV[name],out,16000,subtype="PCM_16")
        extra[name]=out
        scales[name]=dict(gain=float(gain),pcm_sha256=ch.digest(out),wav_sha256=ch.sha(WAV[name]),
                          rms_db_vs_original=float(20*np.log10(np.linalg.norm(out.astype(float))/np.linalg.norm(a))),
                          peak=int(abs(out.astype(float)).max()))
    reader,asr=ch.load_reader()
    readings=study.prior.p4.reading(dict(v5=dry["v5"],**extra),reader)
    ch.save(FIT,dict(records=scales,readings=readings,asr=asr,sources={p.name:ch.sha(p) for p in
                (fit.FIT,fit.WAV,study.grid.fit.fixed.CUE,Path(__file__))},
                policy="Retain the fit ratios, with one scalar per entire file. Freeze both original-RMS and "
                "original-peak playback before room scoring. Measure digital PCM levels after rounding."))


if __name__=="__main__":
    main()
