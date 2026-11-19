"""Same-voice Willow, ordinary and coded Nyx, and Nick controls through identical room channels."""
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import run_c161_willow_matched_scenes_v1 as official
from transition_inputs import commands, read_noise, read_room
from voice_cues import MODEL, MODEL_SHA, PACK, PRIOR, RESULTS, STEM, VOICES, binding, c161_binding, codes, decode, detect
from voice_cues import digest, joined, load_reader, read, recognition, run_stream, save, sha, source_bank, times, variants

PLAN=RESULTS/f"{STEM}-willow-plan.json"
OUT=Path(".local/wakeword/c161-coded-voices")
RUNS=OUT/"runs"
DRY=RESULTS/f"{STEM}-willow-dry.zip"
CASE=Path(__file__).with_name("voice_cues_willow_case.py").resolve()


def prepare():
    OUT.mkdir(exist_ok=True)
    bank = source_bank()
    pairs = [(v,t) for v in VOICES for t in ("Hey Willow.","Hey Nick.")]
    if not DRY.exists():
        speech, _ = commands(pairs)
        with ZipFile(DRY,"x",ZIP_DEFLATED) as z:
            for (v,t),pcm in zip(pairs,speech,strict=True):
                buf=io.BytesIO()
                sf.write(buf,pcm,16000,format="WAV",subtype="PCM_16")
                z.writestr(ZipInfo(f"{v}-{t.split()[1][:-1]}.wav"),buf.getvalue(),compress_type=ZIP_DEFLATED)
    extra={}
    with ZipFile(DRY) as z:
        for v,t in pairs:
            pcm,rate=sf.read(io.BytesIO(z.read(f"{v}-{t.split()[1][:-1]}.wav")),dtype="int16")
            assert rate==16000
            extra[v,t]=pcm
    sources={str(p):sha(p) for p in (Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx"),
             Path(".local/wakeword/kokoro-models/voices-v1.0.bin"))}
    reference=read(RESULTS/"4090-c161-polling-rooms-challenge-plan.json")
    files={p:sha(Path(p)) for p in reference["runtime"].values()}
    assert all(reference["files"][p]==h for p,h in files.items())
    for p in (Path(__file__).resolve(),CASE,official.official.SOURCE,Path(official.official.__file__),Path(official.__file__),DRY,PACK,PRIOR,MODEL):
        files[str(p)]=sha(p)
    files.update(sources)
    rows=[]
    for rir,bg_name in (("h012_","air_conditioner"),("h004_","music")):
        h,room,_=read_room(rir)
        noise,_,_=read_noise(bg_name)
        for old in read(PRIOR)["records"]:
            if (old["room"],old["role"],old["mode"])!=(room,"ordinary","baseline"):
                continue
            v,text=old["voice"],old["text"]
            plain=bank[v,"Hey Nyx."]
            _,alt=variants(plain,codes())
            roles=dict(ordinary=plain,coded=alt["coded"],willow=extra[v,"Hey Willow."],nick=extra[v,"Hey Nick."],
                       silence=np.zeros_like(plain))
            cmd=fftconvolve(bank[v,text].astype(float),h)
            start=old["cue_at"]
            # Native clips include TTS trailing silence. Reserve the longest complete WAV
            # for every role in this pair so all share identical command and background.
            end=max(old["command_at"],start+max(map(len,roles.values()))+320)
            length=end+len(cmd)+8000
            offset=old["noise_offset"]
            bg=noise[(offset+np.arange(length))%len(noise)]*old["noise_gain"]
            common=bg.copy()
            common[end:end+len(cmd)]+=cmd
            for role,cue in roles.items():
                signal=common.copy()
                wet=fftconvolve(cue.astype(float),h)
                signal[start:start+len(wet)]+=wet
                q=np.rint(old["common_gain"]*signal)
                pcm=q.clip(-32768,32767).astype("<i2")
                path=OUT/f"{len(rows):03}.wav"
                sf.write(path,pcm,16000,subtype="PCM_16")
                files[str(path)]=sha(path)
                if end==old["command_at"] and role=="ordinary":
                    assert digest(pcm)==old["pcm_sha256"]
                rows.append(dict(path=str(path),sha256=files[str(path)],samples=len(pcm),pcm_sha256=digest(pcm),
                    target_id=role,voice=v,carrier=text,room=room,start=start,end=end,old_end=old["command_at"],
                    noise_offset=offset,noise_gain=old["noise_gain"],gain=old["common_gain"],cue_pcm_sha256=digest(cue),
                    clipped=int(np.count_nonzero((q < -32768)|(q > 32767)))))
    assert len(rows)==80
    plan=dict(runtime=reference["runtime"],files=files,records=rows,threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert read(PLAN)["records"]==rows
    else:
        save(PLAN,plan)


def score():
    plan=read(PLAN)
    reader,identity=load_reader()
    native=c161_binding()
    rows=[]
    with TemporaryDirectory() as tmp, ZipFile(RESULTS/f"{STEM}-willow-uart.zip","x",ZIP_DEFLATED) as z:
        detector=binding(Path(tmp)/"v5.so",coded="v5")
        for i,src in enumerate(plan["records"]):
            folder=RUNS/f"{i:02}"
            report=read(folder/"report.json")
            assert report["model"]=="wn9_heywillow_tts" and report["total_prime_samples"]==81600
            assert report["actual_float32_bits"]==report["post_prime_getter_bits"]=="3f24dd2f" and report["stage"]==104
            assert report["execution_sha256"]==sha(PLAN) and report["source_pcm_sha256"]==src["pcm_sha256"]
            for name,field in (("raw-returns.bin","raw_returns_sha256"),("serial.log","serial_sha256")):
                assert sha(folder/name)==report[field]
                z.write(folder/name,f"{i:02}/{name}")
            z.write(folder/"report.json",f"{i:02}/report.json")
            raw=np.fromfile(folder/"raw-returns.bin",np.uint8)
            willow=(np.minimum((np.flatnonzero(raw==2)+1)*512, report["input_samples"])-49344).tolist()
            assert willow==[r["relative_sample"] for r in report["real_events"]] and not report["additional_priming_events"]
            pcm,rate=sf.read(src["path"],dtype="int16")
            assert rate==16000 and digest(pcm)==src["pcm_sha256"] and sha(MODEL)==MODEL_SHA
            scores,clock,_,_=run_stream(native,pcm)
            base=times(decode(np.asarray(scores[0],np.uint8),np.asarray(clock)))
            fixed=detect(detector,[pcm])
            events={"willow":willow,"nyx":joined(base,fixed)}
            readings={}
            for model, ts in events.items():
                if ts and src["start"]<=ts[0]<src["end"] and src["target_id"] in ("willow","coded","ordinary"):
                    begin=max(0,((ts[0]+255)//256*256)//160*160-32000)
                    readings[model]=recognition(reader,pcm[begin:],src["carrier"])
            rows.append(dict(index=i,voice=src["voice"],role=src["target_id"],room=src["room"],carrier=src["carrier"],
                  pcm_sha256=src["pcm_sha256"],cue_at=src["start"],command_at=src["end"],c161=base,fixed=fixed,
                  events=events,capture=readings))
    save(RESULTS/f"{STEM}-willow.json",dict(plan_sha256=sha(PLAN),model_sha256=MODEL_SHA,asr_files=identity,
         uart_sha256=sha(RESULTS/f"{STEM}-willow-uart.zip"),records=rows))


if __name__=="__main__":
    action=sys.argv.pop(1)
    if action=="run":
        official.PLAN=PLAN
        official.OUT=RUNS
        official.__file__=str(CASE)
        official.willow()
    else:
        {"prepare":prepare,"score":score}[action]()
