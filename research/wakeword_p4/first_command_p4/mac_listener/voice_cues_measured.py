"""Pair every voice with the original MIT training-room impulse responses and noise."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve, lfilter
import soundfile as sf

from check_c161_v5_code_gain import CANDIDATE, REPORT, WAV
from voice_cues import PACK, RESULTS, STEM, VOICES, binding, c161_binding, codes, decode, detect, digest
from voice_cues import read, run_stream, save, sha, source_bank, times, variants


def main():
    inventory=RESULTS/"4090-mit-rir-audit.json"
    prior=read(REPORT)
    originals={(r["source"]["rir"],r["source"]["noisy"]):r for r in prior["records"] if r["source"]["role"]=="v5"}
    base=sf.read(WAV,dtype="int16")[0]
    v3=sf.read(RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav",dtype="int16")[0]
    clips={k:variants(w,codes())[1]["coded"] for (k,label),w in source_bank().items() if label=="Hey Nyx."}
    assert set(clips)==set(VOICES)
    clips.update(v5=base,doubled=sf.read(CANDIDATE,dtype="int16")[0])
    native=c161_binding()
    rows=[]
    with TemporaryDirectory() as tmp:
        lib=binding(Path(tmp)/"v5.so",coded="v5")
        for i,r in enumerate(read(inventory)["records"]):
            path=Path(".local/wakeword/mit-rirs")/r["path"]
            assert r["split"]=="training" and sha(path)==r["sha256"]
            h,rate=sf.read(path,dtype="float64")
            assert rate==16000
            h/=np.linalg.norm(h)
            wet={v:fftconvolve(x,h) for v,x in clips.items()}
            gain=min(.65,.78*32767/max(np.abs(fftconvolve(v3,h)).max(),np.abs(wet["v5"]).max()))
            lead,tail=14537,8000
            count=lead+len(wet["v5"])+tail
            max_len=lead+max(map(len,wet.values()))+tail
            n=lfilter([1],[1,-.85],np.random.default_rng(52631+i).normal(size=max_len))
            n*=gain*np.sqrt(np.mean(v3.astype(float)**2)/np.mean(n[:count]**2))/10
            start=lead+int(np.abs(h).argmax())
            for v,signal in wet.items():
                dry=np.r_[np.zeros(lead),gain*signal,np.zeros(tail)]
                for noisy in (False,True):
                    q=np.rint(dry+n[:len(dry)] if noisy else dry)
                    pcm=q.clip(-32768,32767).astype("<i2")
                    ev=detect(lib,[pcm])
                    checksum=digest(pcm)
                    end=start+len(clips[v])
                    hit=bool(ev and start<=ev[0]["t"]<end)
                    if v=="v5":
                        old=originals.pop((r["path"],noisy))
                        assert checksum==old["pcm_sha256"] and ev==old["events"]
                    row=dict(voice=v,rir=r["path"],noisy=noisy,pcm_sha256=checksum,fixed=ev,hit=hit,
                             clipped=int(np.count_nonzero((q < -32768)|(q>32767))),start=start,end=end)
                    if not hit:
                        c,clk,_,_=run_stream(native,pcm)
                        events=times(decode(np.asarray(c[0],np.uint8),np.asarray(clk)))
                        row["fallback_c161"]=events
                    rows.append(row)
            if (i+1)%90==0:
                print(i+1,flush=True)
    assert not originals and len(rows)==3240
    save(RESULTS/f"{STEM}-measured.json",dict(inventory_sha256=sha(inventory),pack_sha256=sha(PACK),
         baseline_sha256=sha(REPORT),records=rows))


if __name__=="__main__":
    main()
