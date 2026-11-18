"""Measure command (rather than cue) SNR and ordinary command transcription in the frozen far room."""
import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt
import soundfile as sf

from check_c161_v3_two_anchor_v1 import binding
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import transcripts
from select_c161_synthesized_cue_v2 import load_wav


def normalized(s):
    return re.findall(r"[a-z]+", s.lower())


def rms(signal):
    return np.sqrt(np.mean(signal*signal))


def main():
    source=read(RESULTS/"4090-c161-signed-forward-plan-v1.json")
    merged=read(RESULTS/"4090-c161-coded-cue-v5-commands.json")
    original={(r["source"]["voice"],r["source"]["scene"]):r for r in merged["records"]
              if r["source"]["role"]=="v5" and r["source"]["channel"]=="far"}
    cue=load_wav(dict(path=WAV,sha256=merged["cue_sha256"]))
    identity=read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")
    path=Path(".local/wakeword/whisper-small-en")
    assert all(sha(f)==identity["files"][str(f)] for f in path.iterdir() if f.is_file())
    reader=WhisperModel(str(path),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    rows=[]
    with TemporaryDirectory() as tmp:
        detector=binding(Path(tmp)/"v5.so",coded="v5")
        for src in source["records"]:
            if src["target_id"]!="nyx":
                continue
            ref=original[src["voice"],src["scene"]]
            meta=ref["source"]
            pcm=load_wav(src)
            command=pcm[src["first_phone"]:]
            dry=np.r_[pcm[:src["start"]],cue,command]
            rate=meta["rate"]
            shifted=np.interp(np.arange(round(len(dry)/rate))*rate,np.arange(len(dry)),dry)
            ir=np.zeros(max(t for t,_ in meta["room_taps"])+1)
            for t,val in meta["room_taps"]:
                ir[t]=val
            wet=.53*sosfilt(butter(2,meta["band_hz"],fs=16000,btype="bandpass",output="sos"),fftconvolve(shifted,ir))
            seed=33601+src["scene"]+19*("af_aoede","am_eric","bf_lily","am_fenrir").index(src["voice"])
            noise=lfilter([1.],[1.,-.82],np.random.default_rng(seed).normal(size=len(wet)))
            noise*=.53*rms(cue.astype(float))*np.sqrt(1-.82**2)*10**(-meta["noise_db"]/20)
            start=meta["end"]
            # Exclude the renderer's 8000-sample terminal padding from the reported SNR.
            end=round((len(dry)-8000)/rate)
            span=slice(start,end)
            for reduction in (0,6,12,None):
                n=noise*10**(-reduction/20) if reduction is not None else np.zeros_like(noise)
                analog=wet+n
                assert np.abs(analog).max()<32768
                sample=np.rint(analog).astype("<i2")
                digest=hashlib.sha256(sample.tobytes()).hexdigest()
                if reduction==0:
                    assert digest==ref["pcm_sha256"]
                events=detect(detector,[sample])
                transcript=transcripts(reader,sample[start:])
                words=normalized(transcript["text"])
                expected=normalized(src["carrier"])
                rows.append(dict(voice=src["voice"],scene=src["scene"],expected=src["carrier"],noise_reduction_db=reduction,
                     command_rms=float(rms(wet[span])),noise_rms=float(rms(n[span])),
                     command_snr_db=float(20*np.log10(rms(wet[span])/rms(n[span]))) if reduction is not None else None,
                     command_span=[start,end],pcm_sha256=digest,events=events,cue_start=meta["start"],
                     first_word=words[:1]==expected[:1],all_words=words==expected,transcript=transcript))
                if src["voice"]=="af_aoede" and src["scene"]==0 and reduction==12:
                    dest=RESULTS/"4090-c161-v5-far-command-minus12.wav"
                    with dest.open("xb") as f:
                        sf.write(f,sample,16000,format="WAV",subtype="PCM_16")
                    rows[-1].update(wav=dest.name,wav_sha256=sha(dest))
    save(RESULTS/"4090-c161-v5-far-command-noise.json",dict(source_plan_sha256=sha(RESULTS/"4090-c161-signed-forward-plan-v1.json"),
         reference_sha256=sha(RESULTS/"4090-c161-coded-cue-v5-commands.json"),records=rows))


if __name__=="__main__":
    main()
