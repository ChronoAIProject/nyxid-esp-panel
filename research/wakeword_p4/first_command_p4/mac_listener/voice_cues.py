"""Separate room and noise effects on four ordinary voices and test the same code in each."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_command_playback_level import CANDIDATE, RESULTS, STEM as LEVEL, digest, limited, load_reader, recognition
from check_c161_v5_code_gain import MODEL, MODEL_SHA, WAV, binding, c161_binding, decode, detect, joined, read, run_stream, save, sha, times
from command_level_rooms import TEXTS, VOICES
from run_c161_first_command_v5 import transcripts
from transition_inputs import read_noise, read_room

STEM = "4090-c161-coded-voices"
DRY = RESULTS/f"{LEVEL}-new-dry.zip"
PRIOR = RESULTS/f"{LEVEL}-new-rooms.json"
PACK = RESULTS/f"{STEM}-playback.zip"


def source_bank():
    assert sha(DRY) == read(PRIOR)["dry_zip_sha256"]
    bank = {}
    with ZipFile(DRY) as z:
        for voice in VOICES:
            for i, text in enumerate(("Hey Nyx.", *TEXTS)):
                pcm, rate = sf.read(io.BytesIO(z.read(f"{voice}-{i}.wav")), dtype="int16")
                assert rate == 16000 and pcm.ndim == 1
                bank[voice,text] = pcm
    return bank


def codes():
    spec = read(RESULTS/"4090-c161-coded-cue-v5-source.json")
    v3 = RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav"
    assert sha(WAV) == spec["wav_sha256"] and sha(v3) == spec["v3_sha256"]
    old = sf.read(v3,dtype="int16")[0].astype(np.int32)
    new = sf.read(WAV,dtype="int16")[0].astype(np.int32)
    return 2*(new-old)[11840:17600]


def variants(speech, code):
    power = np.mean(np.pad(speech.astype(float)**2,(0,-len(speech)%320)).reshape(-1,320),axis=1)
    end = min(len(speech), 320*(int(np.flatnonzero(power >= power.max()*.001)[-1])+1))
    start = end-len(code)
    assert start >= 0
    wrong = code.copy()
    wrong[-1280:] = wrong[-1280:][::-1]
    variants = dict(ordinary=speech, level=limited(speech))
    for name, pattern in (("coded",code),("sham",wrong)):
        mix = speech.astype(np.int32)
        mix[start:end] += pattern
        assert np.abs(mix).max() <= 32767
        variants[name] = mix.astype("<i2")
    return start, variants


def write_pack(bank, code):
    reader, identity = load_reader()
    records = []
    with ZipFile(PACK,"x", ZIP_DEFLATED) as z:
        for voice in VOICES:
            start, clips = variants(bank[voice,"Hey Nyx."],code)
            for mode, clip in clips.items():
                buf = io.BytesIO()
                sf.write(buf,clip,16000,format="WAV",subtype="PCM_16")
                data = buf.getvalue()
                name = f"{voice}-{mode}.wav"
                z.writestr(ZipInfo(name),data,compress_type=ZIP_DEFLATED)
                records.append(dict(voice=voice,mode=mode,file=name,template_start=start,samples=len(clip),
                    sha256=hashlib.sha256(data).hexdigest(),pcm_sha256=digest(clip),transcript=transcripts(reader,clip)))
        manifest = dict(sample_rate=16000, source_dry_sha256=sha(DRY),template_pcm_sha256=digest(code.astype("<i2")),
                        asr_files=identity,records=records)
        z.writestr(ZipInfo("manifest.json"),json.dumps(manifest,indent=2),compress_type=ZIP_DEFLATED)


def scene_materials():
    bank = source_bank()
    baseline = [r for r in read(PRIOR)["records"] if r["role"]=="ordinary" and r["mode"]=="baseline"]
    code = codes()
    with ZipFile(PACK) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert all(hashlib.sha256(z.read(r["file"])).hexdigest()==r["sha256"] for r in manifest["records"])
    files = {r["voice"]+"-"+r["mode"]:r for r in manifest["records"]}
    for rir,noise_name in (("h012_","air_conditioner"),("h004_","music")):
        h,room,rir_hash=read_room(rir)
        noise,noise_path,noise_hash=read_noise(noise_name)
        for row in baseline:
            if row["room"] != room:
                continue
            voice, text = row["voice"], row["text"]
            cmd = bank[voice,text]
            assert row["noise_source"]==noise_path and digest(cmd)==row["source_command_sha256"]
            start, onset = row["cue_at"],row["command_at"]
            wet_cmd=fftconvolve(cmd.astype(float),h)
            length=onset+len(wet_cmd)+8000
            bg=noise[row["noise_offset"]:row["noise_offset"]+length]*row["noise_gain"]
            anchor, clips=variants(bank[voice,"Hey Nyx."],code)
            for mode,cue in clips.items():
                assert digest(cue)==files[voice+"-"+mode]["pcm_sha256"]
                for acoustic in (("direct","noise","room","mixture") if mode=="ordinary" else ("mixture",)):
                    wet = acoustic in ("room","mixture")
                    signal = bg.copy() if acoustic in ("noise","mixture") else np.zeros(length)
                    c = wet_cmd if wet else cmd
                    signal[onset:onset+len(c)] += c
                    w = fftconvolve(cue.astype(float),h) if wet else cue
                    signal[start:start+len(w)] += w
                    signal *= row["common_gain"]
                    quantized = np.rint(signal)
                    clipped = int(np.count_nonzero((quantized < -32768) | (quantized > 32767)))
                    pcm=quantized.clip(-32768,32767).astype("<i2")
                    if acoustic=="mixture" and mode=="ordinary":
                        assert digest(pcm)==row["pcm_sha256"]
                    meta=dict(voice=voice,room=room,rir_sha256=rir_hash,noise_sha256=noise_hash,
                        mode=mode,acoustic=acoustic,expected=text,cue_at=start,command_at=onset,
                        cue_end=start+len(cue),template_start=start+anchor,source_sha256=digest(cue),
                        common_gain=row["common_gain"],noise_gain=row["noise_gain"],noise_offset=row["noise_offset"],
                        clipped_samples=clipped)
                    yield meta, pcm, cue, row


def score():
    reader, identity=load_reader()
    assert sha(MODEL)==MODEL_SHA
    native=c161_binding()
    rows=[]
    with TemporaryDirectory() as temp:
        lib=binding(Path(temp)/"cue.so",coded="v5")
        for meta,pcm,_,original in scene_materials():
            codes,clock,_,_=run_stream(native,pcm)
            c=np.asarray(codes[0],np.uint8)
            clock=np.asarray(clock)
            total=np.convolve(c.astype(int),np.ones(5,int),mode="valid")
            active=(clock[4:]>=meta["cue_at"]) & (clock[4:]<meta["command_at"])
            base=times(decode(c,clock))
            fixed=detect(lib,[pcm])
            merged=joined(base,fixed)
            if meta["mode"]=="ordinary" and meta["acoustic"]=="mixture":
                assert (base,fixed,merged)==(original["c161"],original["fixed"],original["merged"])
            hit=bool(merged and meta["cue_at"]<=merged[0]<meta["command_at"])
            capture=None
            if hit and meta["acoustic"]=="mixture":
                begin=max(0,((merged[0]+255)//256*256)//160*160-32000)
                capture=recognition(reader,pcm[begin:],meta["expected"])
            rows.append(dict(**meta,pcm_sha256=digest(pcm),c161=base,fixed=fixed,merged=merged,hit=hit,
                c161_peak5=int(total[active].max()),capture=capture))
    save(RESULTS/f"{STEM}-rooms.json",dict(asr_files=identity,model_sha256=MODEL_SHA,archive_sha256=sha(PACK),
         original_report_sha256=sha(PRIOR),records=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("render","score"))
    if parser.parse_args().action=="render":
        write_pack(source_bank(),codes())
    else:
        score()
