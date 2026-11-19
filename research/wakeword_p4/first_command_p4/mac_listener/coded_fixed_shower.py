"""Put a command immediately after the handoff cue in the two rooms with C161-only notifications."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve, lfilter
import soundfile as sf

import coded_fixed_advance as fixed
from check_c161_v5_playback_opus import codec_api, roundtrip
from reference_signature import qualify

ch=fixed.ch


def main():
    cue,rate=sf.read(fixed.CUE,dtype="int16")
    assert rate==16000 and ch.sha(fixed.CUE)==ch.read(fixed.SOURCE_INFO)["wav_sha256"]
    lib=codec_api()
    waves=dict(raw=cue)
    for kbps in (12,48):
        waves[f"voip-{kbps}"]=roundtrip(lib,cue,2048,kbps*1000)[0]
    bank,_=ch.dry_bank()
    text=ch.TEXTS[0]
    dry=bank["am_eric",text]
    cmd=ch.limited(dry)[ch.active(dry)[0]:]
    inventory=ch.read(ch.RESULTS/"4090-mit-rir-audit.json")["records"]
    previous=ch.read(ch.RESULTS/f"{fixed.STEM}-rooms.json")
    reader,asr=ch.load_reader()
    native=ch.c161_binding()
    original=sf.read(fixed.WAV,dtype="int16")[0]
    v3=sf.read(fixed.SOURCE,dtype="int16")[0]
    rows=[]
    with TemporaryDirectory() as tmp:
        work=Path(tmp)
        detector=ch.binding(work/"full.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        for i,r in enumerate(inventory):
            if Path(r["path"]).name[:4] not in ("h024","h081"):
                continue
            path=Path(".local/wakeword/mit-rirs")/r["path"]
            assert ch.sha(path)==r["sha256"] and r["split"]=="training"
            h=sf.read(path)[0]
            h=h/np.linalg.norm(h)
            gain=min(.65,.78*32767/max(abs(fftconvolve(v3,h)).max(),abs(fftconvolve(original,h)).max()))
            start=14537
            at=start+len(cue)
            wetcmd=gain*fftconvolve(cmd.astype(float),h)
            n=at+len(wetcmd)+8000
            count=start+len(original)+len(h)-1+8000
            noise=lfilter([1],[1,-.85],np.random.default_rng(52631+i).normal(size=max(n,count)))
            noise*=gain*np.sqrt(np.mean(v3.astype(float)**2)/np.mean(noise[:count]**2))/10
            for name,wave in waves.items():
                ref=np.zeros(n,np.int16)
                ref[start:at]=wave
                response=gain*fftconvolve(wave.astype(float),h)
                for noisy in (False,True):
                    signal=noise[:n].copy() if noisy else np.zeros(n)
                    signal[start:start+len(response)]+=response
                    if name=="raw":
                        length=start+len(response)+8000
                        witness,=[x for x in previous["records"] if x["room"]==r["path"] and x["noisy"]==noisy and x["mode"]=="advance"]
                        assert ch.digest(np.rint(signal[:length]).clip(-32768,32767).astype("<i2"))==witness["pcm_sha256"]
                    signal[at:at+len(wetcmd)]+=wetcmd
                    q=np.rint(signal)
                    mic=q.clip(-32768,32767).astype("<i2")
                    scored=ch.routes(detector,native,mic)
                    meta=dict(room=r["path"],mode=name,noisy=noisy,cue_at=start,command_at=at,carrier=text)
                    for player in ("external","self"):
                        digital=ref if player=="self" else np.zeros_like(ref)
                        result=fixed.self_score(work,dict(**meta,player=player),mic,digital,detector,len(rows),
                                                build=fixed.BUILD,prefix=pair,qualify=qualify)
                        assert result["fixed"]==[a["t"] for a in scored["fixed"]] and result["c161"]==scored["c161"]
                        if result["opened"]:
                            begin=max(0,result["opened"][0]//160*160-32000)
                            result["raw_capture"]=fixed.uplink.recognition(reader,mic[begin:],text)
                        rows.append(dict(**meta,player=player,clipped=int((q!=mic).sum()),result=result))
    assert len(rows)==24
    ch.save(ch.RESULTS/f"{fixed.STEM}-difficult.json",dict(records=rows,firmware=fixed.uplink.firmware(),asr_files=asr,
        cue_sha256=ch.sha(fixed.CUE),code_sha256=ch.sha(Path(__file__)),source_sha256=ch.sha(ch.RESULTS/f"{fixed.STEM}-rooms.json"),
        policy="Quiet and original colored noise. Raw, 12 or 48 kbit/s VoIP playback of the wake cue, remove signaled source codec lookahead. External command starts at source cue end. Identical mic with silent or decoded-cue reference. Host Whisper reads actual OPEN-clock raw capture; no Opus uplink in this stress probe."))


if __name__=="__main__":
    main()
