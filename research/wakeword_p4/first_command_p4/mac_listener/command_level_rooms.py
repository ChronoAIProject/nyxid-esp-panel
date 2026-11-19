"""Cross two newly synthesized commands with room, voice and fixed/ordinary/silent onsets."""
import argparse
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_command_playback_level import CANDIDATE, RESULTS, STEM, binding, c161_binding, digest, levels, limited
from check_c161_command_playback_level import load_reader, read, recognition, routes, save, sha
from transition_inputs import commands, read_noise, read_room

VOICES = ("af_heart","am_onyx","bf_isabella","bm_lewis")
TEXTS = ("Raise the bedroom blinds halfway.", "Read tomorrow's first calendar entry.")


def render(bundle):
    requests = [(v,t) for v in VOICES for t in ("Hey Nyx.",*TEXTS)]
    dry, _ = commands(requests)
    with ZipFile(bundle,"x",ZIP_DEFLATED) as z:
        for (voice,text),wave in zip(requests,dry,strict=True):
            stream=io.BytesIO()
            sf.write(stream,wave,16000,format="WAV",subtype="PCM_16")
            z.writestr(f"{voice}-{('Hey Nyx.',*TEXTS).index(text)}.wav",stream.getvalue())


def main(bundle, output, match_level):
    bank = {}
    files = {}
    inventory=read(RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    for path in (".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx",".local/wakeword/kokoro-models/voices-v1.0.bin"):
        files[path]=sha(Path(path))
        assert files[path]==inventory["files"][path]
    with ZipFile(bundle) as z:
        for voice in VOICES:
            for i,text in enumerate(("Hey Nyx.",*TEXTS)):
                audio,rate=sf.read(io.BytesIO(z.read(f"{voice}-{i}.wav")),dtype="int16")
                assert rate==16000 and audio.ndim==1
                bank[voice,text]=audio
    cue,rate = sf.read(CANDIDATE,dtype="int16")
    assert rate==16000 and sha(CANDIDATE)==read(RESULTS/f"{STEM}.json")["cue_sha256"]
    reader,identity=load_reader()
    native=c161_binding()
    rows=[]
    with TemporaryDirectory() as tmp:
        detector=binding(Path(tmp)/"fixed.so",coded="v5")
        for rir,background in (("h012_","air_conditioner"),("h004_","music")):
            h,room,files[rir]=read_room(rir)
            noise,noise_path,files[background]=read_noise(background)
            for iv,voice in enumerate(VOICES):
                natural=bank[voice,"Hey Nyx."]
                for it,text in enumerate(TEXTS):
                    command=bank[voice,text]
                    start=32000
                    onset=start+max(len(cue),len(natural))+320
                    wet=fftconvolve(command.astype(float),h)
                    length=onset+len(wet)+8000
                    power=np.mean(np.pad(wet**2,(0,-len(wet)%320)).reshape(-1,320),axis=1)
                    active=np.repeat(power >= power.max()*.001,320)[:len(wet)]
                    assert len(noise) >= length
                    offset=((27+3*iv+it)*16000) % (len(noise)-length+1)
                    background_pcm=noise[offset:offset+length]
                    assert len(background_pcm)==length
                    noise_gain=np.linalg.norm(wet[active])/np.linalg.norm(background_pcm[onset:onset+len(wet)][active])
                    conditions={}
                    for role,w in (("fixed",cue),("ordinary",natural),("silence",np.zeros_like(cue))):
                        wake=fftconvolve(w.astype(float),h)
                        for name,utterance in (("baseline",command),("limited",limited(command))):
                            cmd=fftconvolve(utterance.astype(float),h)
                            mixed=noise_gain*background_pcm.copy()
                            mixed[start:start+len(wake)]+=wake
                            mixed[onset:onset+len(cmd)]+=cmd
                            conditions[role,name]=(mixed,cmd)
                    gain=min(1.,28000/max(np.max(np.abs(m)) for m,_ in conditions.values()))
                    if match_level:
                        shaped,boosted=conditions["fixed","limited"]
                        ratio=np.linalg.norm(wet[active])/np.linalg.norm(boosted[active])
                        matched_command=np.rint(limited(command)*ratio).astype("<i2")
                        matched_wet=fftconvolve(matched_command.astype(float),h)
                        adjusted=shaped.copy()
                        adjusted[onset:onset+len(boosted)] += matched_wet-boosted
                        assert np.max(np.abs(gain*adjusted)) < 32767
                        conditions["fixed","matched"]=(adjusted,matched_wet)
                    for (role,name),(mixed,cmd) in conditions.items():
                        if match_level and name!="matched":
                            continue
                        pcm=np.rint(gain*mixed).astype("<i2")
                        outcome=routes(detector,native,pcm)
                        events=outcome["merged"]
                        hit=bool(events and start <= events[0] < onset)
                        capture=None
                        if hit:
                            begin=max(0,((events[0]+255)//256*256)//160*160-32000)
                            capture=recognition(reader,pcm[begin:],text)
                        base=conditions[role,"baseline"][0]
                        np.testing.assert_array_equal(mixed[:onset],base[:onset])
                        dry_source=command if name=="baseline" else limited(command)
                        if name=="matched":
                            dry_source=matched_command
                        rows.append(dict(room=room,noise_source=noise_path,noise_offset=offset,noise_gain=noise_gain,
                            common_gain=gain,voice=voice,text=text,role=role,mode=name,source_command_sha256=digest(command),
                            cue_at=start,command_at=onset,command_snr_db=float(20*np.log10(np.linalg.norm(cmd[active])/np.linalg.norm(wet[active]))),
                            source_level=levels(dry_source), correct_wake=hit,
                            capture=capture,**outcome))
    assert len(rows)==(16 if match_level else 96)
    save(output,dict(records=rows,source_files=files,asr_files=identity,
        cue_sha256=sha(CANDIDATE),dry_zip_sha256=sha(bundle),
        snr="Baseline command active frames at 0 dB. Each paired replay uses the same absolute ambient noise and receiver gain."))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--match-level",action="store_true")
    parser.add_argument("--bundle",type=Path,default=RESULTS/f"{STEM}-new-dry.zip")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.render:
        render(args.bundle)
    else:
        group="new-rooms-matched" if args.match_level else "new-rooms"
        main(args.bundle,args.out or RESULTS/f"{STEM}-{group}.json",args.match_level)
