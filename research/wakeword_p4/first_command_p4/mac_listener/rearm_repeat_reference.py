"""Rearm the playback matcher after each triple and test self playback alongside external interruptions."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve

import check_repeat_p4 as replay
from code_gain_guard import activity_pulses
from self_guard import expected
from replay_c161_pcm_cues_v1 import detect

repeat, phase, grid, ch = replay.repeat, replay.phase, replay.grid, replay.ch
BUILD = phase.listener.WORK/"build-repeat-reference"
OUT = ch.RESULTS/(repeat.source.STEM+"-reference.json")
BOARD = ch.RESULTS/(repeat.source.STEM+"-reference-p4.json")


def fresh_hits(detector, signal):
    # Each completed reference triple finishes on the four-sample phase boundary.
    # Independent replay from the next input sample mirrors constructing fresh state.
    start, hits = 0, []
    while start < len(signal):
        events = detect(detector,[signal[start:]])
        if not events:
            break
        start += events[0]["t"]
        assert start%4==0 and (not hits or start>hits[-1])
        hits.append(start)
    return hits


def refs(s,name,dry):
    wave=repeat.drift.project(dry[name],grid.Fraction(1000000,1000000+s["player_ppm"]))
    ref=np.zeros(s["samples"],dtype="<i2")
    ref[s["start"]:s["command_at"]]=grid.room.quantize(wave.astype(float)*10**(s["level_db"]/20))
    return ref


def revised(full, route, old, ref):
    hits=fresh_hits(full,ref)
    previous=old["reference_events"]
    assert hits[:1]==previous[:1]
    signed=old["qualified_prefix"]
    tentative=sorted(set(old["prefix_reference"])-set(signed))
    allowed=sorted(expected(old["fixed"]+route["c161"],hits,activity_pulses(ref),signed,tentative))
    return dict(reference_events=hits,allowed=allowed,opened=[(allowed[0]+255)//256*256] if allowed else [])


def interruptions(dry):
    # Panel plays the candidate. A second source plays the same candidate and a command.
    # Shift only the second source. Noise, gain and the local electrical reference are fixed.
    bg,_,bh=grid.room.rooms.read_noise("music")
    bank=grid.room.rooms.source_bank()
    cases=((1,3),(7,9),(18,20),(24,26))
    local_at=32000
    cue=dry["repeat_trained"]
    for i,(a,b) in enumerate(cases):
        hp,rp,ph=grid.room.rooms.read_room(f"h{a:03}_")
        hx,rx,xh=grid.room.rooms.read_room(f"h{b:03}_")
        voice,cmd=sorted(k for k in bank if k[1].startswith(("Raise ","Read ")))[i]
        source=np.r_[cue,bank[voice,cmd]]
        local=fftconvolve(cue.astype(float),hp)
        ext=fftconvolve(source.astype(float),hx)
        for offset in (0,4000,8000,16000,40000):
            at=local_at+offset
            end=at+len(cue)
            length=at+len(ext)+4800
            noise=bg[(39413*i+np.arange(length))%len(bg)]
            ref=np.zeros(length,dtype="<i2")
            ref[local_at:local_at+len(cue)]=cue
            common=noise * (np.sqrt(np.mean(local[:len(cue)]**2))*.35/np.sqrt(np.mean(noise**2)))
            panel=common.copy()
            panel[local_at:local_at+len(local)]+=local
            both=panel.copy()
            both[at:at+len(ext)]+=ext
            external=common.copy()
            external[at:at+len(ext)]+=ext
            gain=min(1.,28000/max(np.max(np.abs(v)) for v in (both,panel,external)))
            meta=dict(room_id=a,panel_room=rp,external_room=rx,panel_rir_sha256=ph,external_rir_sha256=xh,noise_sha256=bh,
                      offset=offset,start=at,command_at=end,command=cmd,voice=voice,gain=float(gain))
            yield meta,ref,{k:grid.room.quantize(x*gain) for k,x in (("local_only",panel),("external_only",external),("both",both))}


def host():
    dry={"v5":grid.fit.original()[1],**repeat.tracks()}
    corpus=ch.read(repeat.OUT)
    records,barge=[],[]
    counts=defaultdict(Counter)
    host_native=grid.room.rooms.c161_binding()
    with TemporaryDirectory() as tmp:
        full=ch.binding(Path(tmp)/"v5.so",coded="v5")
        pair=ch.binding(Path(tmp)/"pair.so",coded="aligned")
        cache={}
        for i,r in enumerate(corpus["records"]):
            s=r["source"]
            arms={}
            for name,route in r["routes"].items():
                old=route["self"]
                ref_key=(name,s["samples"],s["player_ppm"],s["level_db"])
                if ref_key not in cache:
                    cache[ref_key]=refs(s,name,dry)
                ref=cache[ref_key]
                assert ch.digest(ref)==old["reference_sha256"]
                updated=revised(full,route,old,ref)
                counts[name].update(trials=1,old=bool(old["opened"]),new=bool(updated["opened"]))
                arms[name]=updated
            records.append(dict(index=i,routes=arms))
        for meta,ref,signals in interruptions(dry):
            row=dict(source=meta,routes={})
            for role,x in signals.items():
                reference=np.zeros_like(ref) if role=="external_only" else ref
                raw=grid.outcome(full,host_native,x,meta)
                old=repeat.drift.guarded(raw,full,pair,reference)
                new=revised(full,raw,old,reference)
                row["routes"][role]=dict(mic_sha256=ch.digest(x),reference_sha256=ch.digest(reference),old=old,new=new)
            barge.append(row)
    interruption_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for row in barge:
        source = row["source"]
        for role, routes in row["routes"].items():
            for version in ("old", "new"):
                opened = routes[version]["opened"]
                timely = bool(opened and source["start"] <= opened[0] < source["command_at"])
                interruption_counts[source["offset"]][role][version].update(
                    trials=1, opened=bool(opened), between_external_start_and_command=timely)
    assert len(records)==672 and len(barge)==20
    ch.save(OUT,dict(records=records,interruptions=barge,summary=counts,interruption_counts=interruption_counts,
        sources={p.name:ch.sha(p) for p in (repeat.OUT,repeat.source.PLAN,Path(__file__))},
        policy="Change only the digital playback matcher reset after a confirmed triple. Original microphone "
        "cooldown, veto windows and thresholds remain. External interruptions include matched self-only and "
        "external-only controls; 4 synthetic RIR pairs, five offsets, music and fixed levels."))


def board():
    dry={"v5":grid.fit.original()[1],**repeat.tracks()}
    document=ch.read(OUT)
    corpus=ch.read(repeat.OUT)
    firmware=BUILD/"nyx_mac_listener.bin"
    assert firmware.exists()
    original_build = phase.handoff.test.BUILD
    known = original_build/"nyx_mac_listener.bin"
    assert ch.sha(known) == ch.read(replay.OUT)["firmware_sha256"]
    assert ch.sha(firmware)!=ch.sha(known)
    rows=[]
    archive=BOARD.with_suffix(".zip")
    tmp=archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        full=ch.binding(work/"full.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        def run(s,x,ref,expected_result,build,label):
            j=len(rows)
            alt=build==BUILD
            for attempt in range(2):
                try:
                    result=phase.handoff.probe.score(work,s,x,ref,full,j,build=build,prefix=pair,
                        qualify=phase.handoff.probe.qualify,wait_for_third=True,frame_activity=True,qualify_capture_frame=True,
                        reference_hits=(lambda pcm:fresh_hits(full,pcm)) if alt else None)
                    break
                except (AssertionError,TimeoutError):
                    z.write(work/"serial.log",f"retries/{j:03}-{attempt}.log")
                    if attempt: raise
            assert result["opened"]==expected_result["opened"] and result["allowed"]==expected_result["allowed"]
            assert result["reference_events"]==expected_result["reference_events"]
            lines=result["uart_events"]
            z.writestr(f"events/{j:03}.txt","\n".join(lines)+"\n")
            rows.append(dict(label=label,build="rearmed" if alt else "original",source=s,mic_sha256=ch.digest(x),
                             ref_sha256=ch.digest(ref),result=result))
            print(j,label,alt,result["opened"],flush=True)
        for i, ((_,s,signals,_),old,row) in enumerate(zip(repeat.materials(),corpus["records"],document["records"],strict=True)):
            assert s==old["source"] and i==row["index"]
            for name,r in old["routes"].items():
                if not r["self"]["opened"]: continue
                ref=refs(s,name,dry)
                assert ch.digest(signals[name]) == r["pcm_sha256"]
                assert ch.digest(ref) == r["self"]["reference_sha256"]
                for b,want in ((original_build,r["self"]),(BUILD,row["routes"][name])):
                    run(s,signals[name],ref,want,b,name)
        for (s,ref,signals),old in zip(interruptions(dry),document["interruptions"],strict=True):
            assert s==old["source"]
            # Replay every mixed case plus controls for the first geometry.
            for role,x in signals.items():
                if role!="both" and s["room_id"]!=1: continue
                reference=np.zeros_like(ref) if role=="external_only" else ref
                assert ch.digest(x) == old["routes"][role]["mic_sha256"]
                assert ch.digest(reference) == old["routes"][role]["reference_sha256"]
                for b,key in ((original_build,"old"),(BUILD,"new")):
                    run(s,x,reference,old["routes"][role][key],b,role)
    with ZipFile(tmp) as z: assert z.testzip() is None
    tmp.replace(archive)
    ch.save(BOARD,dict(records=rows,host_sha256=ch.sha(OUT),original_firmware=ch.sha(known),
        rearmed_firmware=ch.sha(firmware),archive_sha256=ch.sha(archive),script_sha256=ch.sha(Path(__file__)),
        policy="Paired original/modified QEMU builds on every observed self opening, each mixed interruption, "
        "and first-geometry controls. Confirm host and guest allowed event and OPEN clocks."))


if __name__=="__main__":
    {"host":host,"p4":board}[sys.argv[1]]()
