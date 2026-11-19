"""Measure fixed-marker timing and cross the source placement with 270 measured training RIRs."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve, lfilter
import soundfile as sf

import coded_advance as advance
from check_c161_v5_code_gain import REPORT, WAV
from replay_c161_pcm_cues_v1 import detect
from voice_cues import RESULTS, binding, digest, read, save, sha


def main():
    inventory = RESULTS/"4090-mit-rir-audit.json"
    prior = { (v["source"]["rir"],v["source"]["noisy"]):v for v in read(REPORT)["records"] if v["source"]["role"]=="v5" }
    v3 = sf.read(RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav",dtype="int16")[0]
    v5 = sf.read(WAV,dtype="int16")[0]
    ch = advance.challenge
    bank,_ = ch.dry_bank()
    clips = {(v,m):w for v in ch.VOICES for m,w in zip(("tail","advance"),advance.cue_pair(bank[v,ch.NAMES[0]]),strict=True)}
    rows=[]
    with TemporaryDirectory() as tmp:
        detector = binding(Path(tmp)/"v5.so",coded="v5")
        for i,r in enumerate(read(inventory)["records"]):
            path=Path(".local/wakeword/mit-rirs")/r["path"]
            assert r["split"]=="training" and sha(path)==r["sha256"]
            h,rate=sf.read(path)
            assert rate==16000
            h=h/np.linalg.norm(h)
            old=fftconvolve(v5,h)
            gain=min(.65,.78*32767/max(abs(fftconvolve(v3,h)).max(),abs(old).max()))
            lead,tail=14537,8000
            ref=np.r_[np.zeros(lead),gain*old,np.zeros(tail)]
            count=len(ref)
            rng=np.random.default_rng(52631+i)
            max_len=max(count,lead+max(map(len,clips.values()))+len(h)-1+tail)
            noise=lfilter([1],[1,-.85],rng.normal(size=max_len))
            noise*=gain*np.sqrt(np.mean(v3.astype(float)**2)/np.mean(noise[:count]**2))/10
            direct=int(abs(h).argmax())
            for noisy in (False,True):
                q=np.rint(ref+noise[:count] if noisy else ref).clip(-32768,32767).astype("<i2")
                witness=prior.pop((r["path"],noisy))
                assert digest(q)==witness["pcm_sha256"] and detect(detector,[q])==witness["events"]
            for (voice,mode),cue in clips.items():
                wet=np.r_[np.zeros(lead),gain*fftconvolve(cue,h),np.zeros(tail)]
                for noisy in (False,True):
                    q=np.rint(wet+noise[:len(wet)] if noisy else wet)
                    mic=q.clip(-32768,32767).astype("<i2")
                    events=detect(detector,[mic])
                    first=(events[0]["t"]+255)//256*256 if events else None
                    rows.append(dict(room=r["path"],voice=voice,mode=mode,noisy=noisy,pcm_sha256=digest(mic),
                          peak_rir_sample=direct,source_cue_end=lead+len(cue),events=events,delivery=first,
                          clipped=int((q!=mic).sum())))
    assert len(rows)==4320 and not prior
    save(RESULTS/f"{advance.STEM}-rooms.json",dict(records=rows,code_sha256=sha(Path(__file__)),inventory_sha256=sha(inventory),
         dry_sha256=sha(ch.DRY),reference_sha256=sha(REPORT),policy="Full isolated cue response plus tail; no command appended. Fixed marker only. Delivery rounded to next 256-sample acquisition block. An external command beginning at the dry cue end cannot affect microphones before source emission. RIR peak is separately provided."))


if __name__ == "__main__":
    main()
