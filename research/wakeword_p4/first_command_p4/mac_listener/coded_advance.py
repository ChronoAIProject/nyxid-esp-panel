"""Advance the three source-side codes by one 80 ms segment while keeping both speech clocks fixed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve

import coded_close_join as challenge
import coded_command_uplink as uplink
from code_gain_guard import score as self_score
from prefix_guard import BUILD
from reference_signature import qualify

STEM = "4090-c161-coded-advance"
PACK = challenge.RESULTS / f"{STEM}-playback.zip"
UART = challenge.RESULTS / f"{STEM}-uart.zip"
SHIFT = 1280


def cue_pair(nyx):
    code = challenge.codes()
    start, clips = challenge.variants(nyx, code)
    end = start+len(code)
    assert start>=SHIFT and end==challenge.active(nyx)[1]
    early = nyx[:end].astype(np.int32)
    early[start-SHIFT:end-SHIFT] += code
    assert np.abs(early).max() <= 32767
    return clips["coded"][:end], early.astype("<i2")


def materials():
    bank,_=challenge.dry_bank()
    plan=challenge.read(challenge.PLAN)
    for rir,kind in (("h012_","air_conditioner"),("h004_","music")):
        h,room,room_hash=challenge.read_room(rir)
        noise,_,noise_hash=challenge.read_noise(kind)
        for i,s in enumerate(plan["records"]):
            if (s["target_id"],s["room"])!=("coded",room):
                continue
            assert (room_hash,noise_hash)==(s["rir_sha256"],s["noise_sha256"])
            v,t=s["voice"],s["carrier"]
            cmd=challenge.limited(bank[v,t])[s["command_trim_samples"]:]
            assert challenge.digest(cmd)==s["command_pcm_sha256"]
            wet=fftconvolve(cmd.astype(float),h)
            common=noise[s["noise_offset"]:s["noise_offset"]+s["samples"]]*s["noise_gain"]
            at=s["end"]
            common[at:at+len(wet)] += wet
            cues=cue_pair(bank[v,"Hey Nyx."])
            for mode,cue in zip(("tail","advance"),cues,strict=True):
                signal=common.copy()
                wake=fftconvolve(cue.astype(float),h)
                signal[s["start"]:s["start"]+len(wake)]+=wake
                q=np.rint(s["common_gain"]*signal)
                assert np.abs(q).max()<=32767
                mic=q.astype("<i2")
                if mode=="tail":
                    assert challenge.digest(mic)==s["pcm_sha256"] and challenge.digest(cue)==s["cue_pcm_sha256"]
                yield i,s,mode,mic,cue,cmd


def main():
    out = challenge.RUNS/"advance"
    out.mkdir(exist_ok=True)
    uplink.WORK = out
    fw = uplink.firmware()
    reader,identity=challenge.load_reader()
    native=challenge.c161_binding()
    decoder=uplink.decoder_api()
    rows, members=[],{}
    with TemporaryDirectory() as tmp, ZipFile(UART.with_suffix(".tmp"),"w",ZIP_DEFLATED) as uart, \
         ZipFile(PACK.with_suffix(".tmp"),"w",ZIP_DEFLATED) as pack:
        temp=Path(tmp)
        detector=challenge.binding(temp/"v5.so",coded="v5")
        pair=challenge.binding(temp/"pair.so",coded="aligned")
        for i,s,mode,mic,cue,cmd in materials():
            j=len(rows)
            expected=challenge.routes(detector,native,mic)
            ref=dict(c161=expected["c161"],fixed=expected["fixed"],events=dict(nyx=expected["merged"]))
            meta=dict(voice=s["voice"],carrier=s["carrier"],room=s["room"],cue_at=s["start"],
                      command_at=s["end"],index=i,mode=mode)
            if expected["merged"]:
                external=uplink.p4(meta,mic,detector,j,decoder,reader,ref,uart,require_timely=False)
            else:
                external=self_score(temp,meta,mic,np.zeros_like(mic),detector,j,build=BUILD,prefix=pair,qualify=qualify)
                assert not external["opened"]
            digital=np.zeros_like(mic)
            digital[s["start"]:s["end"]]=cue
            panel=self_score(temp,meta,mic,digital,detector,j,build=BUILD,prefix=pair,qualify=qualify)
            assert panel["fixed"]==[e["t"] for e in expected["fixed"]] and panel["c161"]==expected["c161"]
            uart.writestr(f"{j:02}-self.log","\n".join(panel["uart_events"])+"\n")
            rows.append(dict(**meta,mic_sha256=challenge.digest(mic),fixed=expected["fixed"],c161=expected["c161"],
                             external=external,panel=panel))
            key=(s["voice"],s["carrier"],mode)
            track=np.r_[cue,cmd]
            if key not in members:
                member=f'{s["voice"]}-{s["carrier"].split()[0].lower()}-{mode}.wav'
                members[key]=dict(**uplink.write_member(pack,member,track),voice=s["voice"],carrier=s["carrier"],
                                  mode=mode,command_at=len(cue),cue_sha256=challenge.digest(cue),
                                  command_sha256=challenge.digest(cmd))
            assert members[key]["pcm_sha256"]==challenge.digest(track)
            print(j,mode,external.get("delivered"),panel["opened"],flush=True)
        assert len(rows)==32 and len(members)==16
        pack.writestr("manifest.json",json.dumps(dict(sample_rate=16000,channels=1,format="PCM16",shift_samples=SHIFT,
            dry_zip_sha256=challenge.sha(challenge.DRY),records=list(members.values())),indent=2)+"\n")
        for failed in sorted(out.glob("*-attempt*.gz")):
            uart.write(failed,failed.name)
    UART.with_suffix(".tmp").replace(UART)
    PACK.with_suffix(".tmp").replace(PACK)
    challenge.save(challenge.RESULTS/f"{STEM}.json",dict(records=rows, shift_samples=SHIFT, asr_files=identity,**fw,
        code_sha256=challenge.sha(Path(__file__)),library_script_sha256=challenge.sha(Path(uplink.__file__)),
        plan_sha256=challenge.sha(challenge.PLAN),uart_sha256=challenge.sha(UART),playback_sha256=challenge.sha(PACK),
        failures=[p.name for p in sorted(out.glob("*-attempt*.gz"))],policy="Same speech samples, code values, wake duration, room, speaker gain and first active command frame. Move the three code segments 80 ms earlier within the synthetic wake utterance. First external OPEN versus source command at 57600. Self condition puts the same wake audio on the playback reference, external command follows."))


if __name__=="__main__":
    main()
