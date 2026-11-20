"""Change the relative source clocks by 400 ms and score native 077 events and sends."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import wavfile
import soundfile as sf

import diagnose_aula_077_command as diagnosis

trial, P = diagnosis.trial, diagnosis.P
shifts = trial.source.shifts
OUT = trial.OUT.with_name("4090-aula-077-command-gap.json")
PROTO = trial.PROTOCOL.with_name("aula_077_command_gap.md")
WORK = trial.WORK/"aula-gap"
PACK = WORK/"hye-nyx-aula-077-gap-v1.zip"
SAMPLES = 6400


def programs(blank,cue,end):
    wake = np.zeros(len(blank))
    wake[end-len(cue):end]=cue.astype(float)/131072
    for name,w,c,b in (("original",0,0,0),("command",0,1,0),
                       ("command-and-interruption",0,1,1),("whole-target",1,1,0)):
        left=shifts.shift(wake,SAMPLES*w)+shifts.shift(blank[:,0],SAMPLES*c)
        right=shifts.shift(blank[:,1],SAMPLES*b)
        yield name,np.column_stack((left,right)),dict(wake=w*SAMPLES,command=c*SAMPLES,interruption=b*SAMPLES)


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    stereo,air=diagnosis.stereo,trial.local.voices.transfer.air
    prior=trial.prior
    measured=P.read(trial.OUT)
    base,=(r for r in measured["records"] if (r["voice"],r["mode"])==("af_bella","aula-original"))
    plan,=(r for r in P.read(trial.local.PLAN)["records"] if r["label"]==base["label"])
    report=P.read(diagnosis.OUT)
    assert P.sha(stereo.PACK)==P.read(stereo.OUT)["archive_sha256"] and P.sha(trial.local.PACK)==P.read(trial.local.OUT)["sources"][str(trial.local.PACK)]
    with ZipFile(stereo.PACK) as z:
        name="077/absent-baseline.wav"
        blank=shifts.member_wave(z,name,P.read(stereo.OUT)["files"][name],"float64")
    with ZipFile(trial.local.PACK) as z:
        meta=P.read(trial.local.OUT)["voices"]["af_bella"]
        cue=shifts.member_wave(z,meta["file"],meta["sha256"],"int16")
    responses=[sf.read(air.response_file(entry))[0][:,1] for entry in air.RESPONSES["aula"]]
    filters=[v/np.linalg.norm(responses[0]) for v in responses]
    noise=np.random.Generator(np.random.PCG64(209473)).normal(0,12,len(blank)+20000)
    dry_end=report["dry_command_start"]
    build=P.read(trial.local.OUT)["build"]
    flash=bytearray((shifts.source.receiver.WORK/"guest.flash").read_bytes())
    assert shifts.fixed.digest(flash)==build["flash_sha256"] and P.sha(shifts.fixed.QEMU)==build["qemu_sha256"]
    trial.scoring.sender.BUILD=prior.SENDER
    images=(trial.scoring.clock.BUILD/"nyx_mac_listener.bin",trial.scoring.prior.CODEC/"nyx_first_command.bin",prior.SENDER/"board.bin")
    assert all(P.sha(p)==measured["sources"][str(p)] for p in images)
    proc,model,qid=trial.scoring.prior.load()
    decoder=trial.scoring.prior.decoder_api()
    assert qid==measured["qwen"] and decoder.opus_get_version_string().decode()==measured["decoder"]
    readers=proc,model,trial.scoring.prior.reader(measured["turbo"]),decoder
    native_cache=WORK/build["flash_sha256"]
    send_cache=WORK/("sends-"+P.sha(diagnosis.OUT)[:12]+"-"+P.sha(images[-1])[:12])
    for folder in (native_cache,send_cache):
        folder.mkdir(exist_ok=True)
    trial.scoring.clock.reference.WORK=send_cache
    rows,files,member_hashes=[],{},{}
    with TemporaryDirectory(dir=WORK) as tmp:
        work=Path(tmp)
        with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as logs, ZipFile(work/PACK.name,"w") as pack:
            for i,(name,wave,offset) in enumerate(programs(blank,cue,dry_end)):
                buf=BytesIO()
                wavfile.write(buf,16000,wave)
                raw=buf.getvalue()
                member=name+".wav"
                stereo.write_member(pack,member,raw)
                member_hashes[member]=sha256(raw).hexdigest()
                rendered=shifts.member_wave(pack,member,member_hashes[member],"float64")
                mic,clip=air.receive(32768*rendered,*filters,noise)
                assert not clip.any()
                ref=np.zeros_like(mic)
                block=shifts.fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                ident=shifts.fixed.digest(block)
                cue_end=base["end"]+offset["wake"]
                cmd=base["end"]+offset["command"]
                row=dict(label=name,offset_samples=offset,cue_end=cue_end,command_at=cmd,pcm_sha256=P.digest(mic),
                         reference_sha256=P.digest(ref),input_sha256=ident)
                if name=="original":
                    assert ident==base["input_sha256"] and row["pcm_sha256"]==base["pcm_sha256"]
                    row.update(events=base["events"],uplink=base["uplink"],crop=base["crop"],source=base["label"])
                else:
                    for attempt in range(2):
                        try:
                            runs,log,faults=shifts.fixed.boot(work,flash,native_cache,block,1)
                            break
                        except AssertionError:
                            if attempt:
                                raise
                    row["events"]=runs[0]["events"]
                    for path in (log,*faults):
                        member=name+("/wake.log" if path==log else "/"+path.name)
                        files[member]=P.sha(path)
                        logs.write(path,member)
                    if row["events"]["O"]:
                        path=work/"mic.wav"
                        sf.write(path,mic,16000,subtype="PCM_16")
                        record=dict(row,path=str(path),reference_path=plan["reference_path"],command=base["command"],end=cue_end)
                        opened,=row["events"]["O"]
                        uplink,shard=trial.scoring.score_case(record,i,opened,readers,work,send_cache)
                        assert all(uplink[k]==0 for k in ("capture_watchdog","opus_watchdog","sender_watchdog"))
                        row["uplink"]=uplink
                        row["crop"]=prior.cropped_readings(record,i,uplink,shard,readers,work,cue,
                                                            locate=prior.anchor.alignment_tool.locate_available_prefix)
                        with ZipFile(shard) as src:
                            for member in src.namelist():
                                if member.endswith(".log"):
                                    raw=src.read(member)
                                    files[member]=sha256(raw).hexdigest()
                                    logs.writestr(member,raw)
                row["raw"]=prior.read_words(readers,mic[cmd-1600:],base["command"])
                opened=next(iter(row["events"]["O"]),None)
                row["timely"]=opened is not None and plan["start"]+offset["wake"]<=opened<cmd
                print(name,opened,row["raw"],flush=True)
                rows.append(row)
            manifest=dict(files=member_hashes.copy(),dry_command_at=dry_end,offsets={r["label"]:r["offset_samples"] for r in rows},
                          sample_rate=16000,command=base["command"],playback="Play one float stereo file at a time. Left is the target "
                          "and right the other speaker. Keep positions and level fixed.")
            raw=(json.dumps(manifest,indent=2)+"\n").encode()
            stereo.write_member(pack,"manifest.json",raw)
            member_hashes["manifest.json"]=sha256(raw).hexdigest()
        deps=(Path(__file__).resolve().relative_to(Path.cwd()),PROTO,diagnosis.OUT,trial.OUT,trial.local.PLAN,stereo.OUT,trial.local.PACK,*images)
        P.save(work/OUT.name,dict(records=rows,files=files,archive_sha256=P.sha(work/"uart.zip"),qwen=qid,turbo=measured["turbo"],
                                  pack=PACK.name,pack_sha256=P.sha(work/PACK.name),members=member_hashes,
                                  sources={str(p):P.sha(p) for p in deps}))
        (work/"uart.zip").replace(OUT.with_suffix(".zip"))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
