"""Deliver one compact fixed cue based on the original v3 speech, with the v5 codes advanced 80 ms."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve, lfilter
import soundfile as sf

import coded_advance as advance
from check_c161_v5_code_gain import CANDIDATE, WAV
from code_gain_guard import score as self_score
from prefix_guard import BUILD
from reference_signature import qualify
from run_c161_first_command_v5 import transcripts

ch, uplink = advance.challenge, advance.uplink
STEM = "4090-c161-single-cue-advance"
SOURCE = ch.RESULTS/"hye-nyx-c161-synthetic-cue-v3.wav"
CUE = ch.RESULTS/"hye-nyx-c161-v5-early-80ms.wav"
SOURCE_INFO = ch.RESULTS/f"{STEM}-source.json"


def cues():
    spec=ch.read(ch.RESULTS/"4090-c161-coded-cue-v5-source.json")
    assert ch.sha(SOURCE)==spec["v3_sha256"] and ch.sha(WAV)==spec["wav_sha256"]
    assert ch.sha(CANDIDATE)==ch.read(ch.RESULTS/"4090-c161-coded-cue-v5-code-gain-commands.json")["doubled_sha256"]
    base,rate=sf.read(SOURCE,dtype="int16")
    normal=sf.read(CANDIDATE,dtype="int16")[0]
    delta=normal.astype(np.int32)-base
    begin,end=11840,17600
    assert rate==16000 and np.count_nonzero(delta[:begin])==np.count_nonzero(delta[end:])==0
    np.testing.assert_array_equal(delta[begin:end],ch.codes())
    early=base.astype(np.int32)
    early[begin-advance.SHIFT:end-advance.SHIFT]+=delta[begin:end]
    assert abs(early).max()<=32767 and ch.active(normal)[1]==end
    return normal[:end],early[:end].astype("<i2")


def generate():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import check_c161_pronunciation_ctc_v1 as ctc
    tail,early=cues()
    sf.write(CUE,early,16000,subtype="PCM_16")
    with sf.SoundFile(CUE) as f:
        assert f.samplerate==16000 and f.channels==1 and f.subtype=="PCM_16"
        np.testing.assert_array_equal(f.read(dtype="int16"),early)
    files=json.loads((ctc.MODEL/"source.json").read_bytes())["files"]
    assert files["model.safetensors"]["sha256"]==ctc.read("ctc-training-audit-plan")["expected_safetensors_sha256"]
    assert all(ch.sha(ctc.MODEL/k)==v["sha256"] for k,v in files.items())
    proc=Wav2Vec2Processor.from_pretrained(ctc.MODEL,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(ctc.MODEL,local_files_only=True).cuda().eval()
    reader,identity=ch.load_reader()
    rows={}
    with torch.inference_mode():
        for k,pcm in (("tail",tail),("advance",early)):
            x=proc(pcm.astype("float32")/32768, sampling_rate=16000,return_tensors="pt")
            ids=model(input_values=x.input_values.cuda()).logits.argmax(-1)
            rows[k]=dict(whisper=transcripts(reader,pcm),wav2vec2=proc.batch_decode(ids.cpu())[0],pcm_sha256=ch.digest(pcm))
    ch.save(SOURCE_INFO,dict(wav_sha256=ch.sha(CUE),bytes=CUE.stat().st_size,samples=len(early),files={
        str(p):ch.sha(p) for p in (SOURCE,CANDIDATE,WAV,Path(__file__))},shift_samples=advance.SHIFT,
        code_positions=[11840-advance.SHIFT,14400-advance.SHIFT,16320-advance.SHIFT],code_length=1280,
        trim_end=17600,readings=rows,whisper_files=identity,wav2vec2_files=files))


def main():
    fw=uplink.firmware()
    cue,rate=sf.read(CUE,dtype="int16")
    assert rate==16000 and ch.sha(CUE)==ch.read(SOURCE_INFO)["wav_sha256"]
    bank,_=ch.dry_bank()
    plan=ch.read(ch.PLAN)["records"]
    uplink.WORK=ch.RUNS/"single"
    uplink.WORK.mkdir(exist_ok=True)
    reader,identity=ch.load_reader()
    decoder=uplink.decoder_api()
    native=ch.c161_binding()
    records, members=[],{}
    uart=ch.RESULTS/f"{STEM}-uart.zip"
    playback=ch.RESULTS/f"{STEM}-commands.zip"
    with TemporaryDirectory() as tmp, ZipFile(uart.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z, \
         ZipFile(playback.with_suffix(".tmp"),"w",ZIP_DEFLATED) as pack:
        work=Path(tmp)
        detector=ch.binding(work/"v5.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        for rir,kind in (("h012_","air_conditioner"),("h004_","music")):
            h,room,room_hash=ch.read_room(rir)
            noise,_,noise_hash=ch.read_noise(kind)
            for s in plan:
                if (s["room"],s["target_id"])!=(room,"coded"):
                    continue
                assert s["rir_sha256"]==room_hash and s["noise_sha256"]==noise_hash
                i=len(records)
                v,t,at=s["voice"],s["carrier"],s["end"]
                cmd=ch.limited(bank[v,t])[s["command_trim_samples"]:]
                assert ch.digest(cmd)==s["command_pcm_sha256"]
                wet=fftconvolve(cmd.astype(float),h)
                start=at-len(cue)
                signal=noise[s["noise_offset"]:s["noise_offset"]+s["samples"]]*s["noise_gain"]
                signal[at:at+len(wet)]+=wet
                wake=fftconvolve(cue.astype(float),h)
                signal[start:start+len(wake)]+=wake
                q=np.rint(s["common_gain"]*signal)
                assert np.abs(q).max()<=32767
                mic=q.astype("<i2")
                scored=ch.routes(detector,native,mic)
                meta=dict(voice=v,room=room,carrier=t,index=i,cue_at=start,command_at=at)
                ref=dict(fixed=scored["fixed"],c161=scored["c161"],events=dict(nyx=scored["merged"]))
                external=uplink.p4(meta,mic,detector,i,decoder,reader,ref,z,require_timely=False)
                digital=np.zeros_like(mic)
                digital[start:at]=cue
                panel=self_score(work,meta,mic,digital,detector,i,build=BUILD,prefix=pair,qualify=qualify)
                z.writestr(f"{i:02}-self.log","\n".join(panel["uart_events"])+"\n")
                records.append(dict(**meta,mic_sha256=ch.digest(mic),external=external,panel=panel))
                if (v,t) not in members:
                    name=f"{v}-{t.split()[0].lower()}.wav"
                    members[v,t]=dict(voice=v,carrier=t,command_at=len(cue),**uplink.write_member(pack,name,np.r_[cue,cmd]))
                print(i,external["delivered"],panel["opened"],flush=True)
        assert len(records)==16 and len(members)==8
        pack.writestr("manifest.json",json.dumps(dict(sample_rate=16000,channels=1,cue_sha256=ch.sha(CUE),
            cue_samples=len(cue),records=list(members.values())),indent=2)+"\n")
        for failed in sorted(uplink.WORK.glob("*-attempt*.gz")):
            z.write(failed,failed.name)
    uart.with_suffix(".tmp").replace(uart)
    playback.with_suffix(".tmp").replace(playback)
    ch.save(ch.RESULTS/f"{STEM}-p4.json",dict(records=records,source_sha256=ch.sha(SOURCE_INFO),plan_sha256=ch.sha(ch.PLAN),
        asr_files=identity,**fw,uart_sha256=ch.sha(uart),playback_sha256=ch.sha(playback),
        failures=[p.name for p in sorted(uplink.WORK.glob("*-attempt*.gz"))]))


def rooms():
    tail,cue=cues()
    assert ch.digest(cue)==ch.read(SOURCE_INFO)["readings"]["advance"]["pcm_sha256"]
    base=sf.read(SOURCE,dtype="int16")[0]
    original=sf.read(WAV,dtype="int16")[0]
    inventory=ch.RESULTS/"4090-mit-rir-audit.json"
    native=ch.c161_binding()
    rows=[]
    with TemporaryDirectory() as tmp:
        detector=ch.binding(Path(tmp)/"full.so",coded="v5")
        for i,r in enumerate(ch.read(inventory)["records"]):
            path=Path(".local/wakeword/mit-rirs")/r["path"]
            assert r["split"]=="training" and ch.sha(path)==r["sha256"]
            h,rate=sf.read(path)
            assert rate==16000
            h=h/np.linalg.norm(h)
            gain=min(.65,.78*32767/max(abs(fftconvolve(base,h)).max(),abs(fftconvolve(original,h)).max()))
            lead=14537
            count=lead+len(original)+len(h)-1+8000
            noise=lfilter([1],[1,-.85],np.random.default_rng(52631+i).normal(size=count))
            noise*=gain*np.sqrt(np.mean(base.astype(float)**2)/np.mean(noise**2))/10
            for mode,wave in (("tail",tail),("advance",cue)):
                signal=np.r_[np.zeros(lead),gain*fftconvolve(wave,h),np.zeros(8000)]
                for noisy in (False,True):
                    q=np.rint(signal+noise[:len(signal)] if noisy else signal)
                    mic=q.clip(-32768,32767).astype("<i2")
                    scored=ch.routes(detector,native,mic)
                    event=next(iter(scored["merged"]),None)
                    rows.append(dict(**scored,room=r["path"],noisy=noisy,mode=mode,start=lead,
                         end=lead+len(wave),peak_rir_sample=int(abs(h).argmax()),clipped=int((q!=mic).sum()),
                         delivery=None if event is None else (event+255)//256*256))
    assert len(rows)==1080
    ch.save(ch.RESULTS/f"{STEM}-rooms.json",dict(records=rows,source_sha256=ch.sha(SOURCE_INFO),
        inventory_sha256=ch.sha(inventory),c161_sha256=ch.matched.MODEL_SHA,
        policy="Isolated cue followed by room decay and 500 ms tail. First merged noise/cue event rounded to 256 samples; judge against source command start at the dry cue end. Report every C161/fixed event. Same 270 consumed room responses and noise seeds."))


if __name__=="__main__":
    import sys
    {"generate":generate,"p4":main,"rooms":rooms}[sys.argv[1]]()
