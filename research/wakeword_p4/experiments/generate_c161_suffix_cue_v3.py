"""Bring the synthesized suffix closer to the wake vowel's playback level."""
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
from audit_c161_history_paths_v1 import decode, times
import verify_c161_ambient_stream_v1 as live
import replay_c161_pcm_cues_v1 as matcher

WAV = RESULTS / "hye-nyx-c161-synthetic-cue-v3.wav"
STEM = "4090-c161-suffix-cue-v3"


def generate():
    old = RESULTS/"hye-nyx-c161-synthetic-cue-v2.wav"
    pcm, rate = sf.read(old,dtype="int16")
    assert rate==16000
    suffix = read(RESULTS/"4090-c161-cue-suffix-templates-v1.json")["records"][1]
    anchor = read(RESULTS/"4090-c161-pcm-cue-anchors-v1.json")["records"][1]
    assert sha(old)==suffix["wav_sha256"]==anchor["wav_sha256"]
    first = anchor["starts_2khz"][1]*8
    s0, s1 = np.asarray(suffix["range_8khz"])*2
    rms = lambda a: float(np.sqrt(np.mean(a.astype(float)**2)))
    vowel_rms = rms(pcm[first:first+1280])
    suffix_rms = rms(pcm[s0:s1])
    gain = min(10**(18/20),0.5*vowel_rms/suffix_rms)
    envelope=np.ones(len(pcm))
    fade=320
    delta=(gain-1)*(1-np.cos(np.arange(1,fade+1)*np.pi/fade))/2
    envelope[s0-fade:s0]+=delta
    envelope[s0:s1]=gain
    envelope[s1:s1+fade]+=delta[::-1]
    amplified=np.rint(pcm*envelope)
    assert np.max(amplified)<=32767 and np.min(amplified)>=-32768
    with WAV.open("xb") as f, wave.open(f,"wb") as w:
        w.setparams((1,2,16000,0,"NONE","not compressed"))
        w.writeframes(amplified.astype("<i2").tobytes())
    from faster_whisper import WhisperModel
    whisper=Path(".local/wakeword/whisper-small-en")
    plan=read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")
    assert all(sha(p)==plan["files"][str(p)] for p in whisper.iterdir() if p.is_file())
    model=WhisperModel(str(whisper),device="cuda",compute_type="float16",cpu_threads=2,num_workers=1)
    segments,_=model.transcribe(amplified.astype(np.float32)/32768,language="en",beam_size=3,temperature=0,
                                condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
    words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability) for s in segments for w in s.words or []]
    save(RESULTS/f"{STEM}-source.json",dict(wav_sha256=sha(WAV),original_sha256=sha(old),suffix_range=[int(s0),int(s1)],
         fade_samples=fade,gain=gain,gain_db=float(20*np.log10(gain)),vowel_rms=vowel_rms,original_suffix_rms=suffix_rms,
         peak=int(np.abs(amplified).max()),transcript="".join(w["word"] for w in words).strip(),words=words))


def replay():
    assert sha(WAV)==read(RESULTS/f"{STEM}-source.json")["wav_sha256"]
    native=binding()
    matcher.LIB=matcher.LIB.with_name("pcm-suffix.so")
    lib=matcher.binding(True)
    adm=read(RESULTS/"4090-chatterbox-daily-training-admission-v1.json")
    row=next(r for r in adm["admitted"] if r["request"]["voice"]=="bm_fable" and r["request"]["group"]=="number_quantity")
    history=load_wav(row["wav"])
    originals=read(RESULTS/"4090-c161-confirmed-cue-transport-v1.json")["records"]
    records=[]
    for v,path in [("v2",RESULTS/"hye-nyx-c161-synthetic-cue-v2.wav"),("v3",WAV)]:
        cue,rate=sf.read(path,dtype="int16")
        assert rate==16000
        for original in (r for r in originals if r["cue"]=="v2"):
            ppm,polarity,channel = original["ppm"],original["polarity"],original["channel"]
            speed=1+ppm/1_000_000
            audio=np.interp(np.arange(round(len(cue)/speed))*speed,np.arange(len(cue)),cue)
            source=audio
            if channel=="filtered":
                ir=np.zeros(1418)
                ir[[0,117,449,1417]]=[.65,.33,-.17,.09]
                source=lfilter([.3,.4,.3],[1],np.convolve(audio,ir))
            raw=np.r_[history,np.zeros(4213),polarity*source*.6,np.zeros(16000)]
            if channel=="filtered":
                noise=lfilter([1],[1,-.7],np.random.default_rng(6900).normal(size=len(raw)))
                # Same absolute noise waveform across the two playback levels of the suffix.
                clean,_=sf.read(RESULTS/"hye-nyx-c161-synthetic-cue-v2.wav",dtype="int16")
                raw+=noise*np.sqrt(np.mean((clean.astype(float)*.6)**2)/np.mean(noise**2))/10**.75
            pcm=np.rint(raw).clip(-32768,32767).astype("<i2")
            digest=hashlib.sha256(pcm.tobytes()).hexdigest()
            if v=="v2":assert digest==original["pcm_sha256"]
            codes,clock,pair,_=live.run_stream(native,pcm)
            base=times(decode(np.asarray(codes[0],np.uint8),np.asarray(clock)))
            extra=matcher.detect(lib,(pcm,))
            if v=="v2":assert extra==original["events"]
            records.append(dict(cue=v,ppm=ppm,channel=channel,polarity=polarity,start=original["start"],end=original["end"],
                                pcm_sha256=digest,original=base,pair=times(pair),confirmed=extra,
                                original_with_cue=matcher.joined(base,extra),pair_with_cue=matcher.joined(times(pair),extra)))
    save(RESULTS/f"{STEM}-transport.json",dict(records=records,wav_sha256=sha(WAV)))


if __name__=="__main__":
    generate()
    replay()
