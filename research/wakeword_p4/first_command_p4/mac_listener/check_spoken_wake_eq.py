"""Evaluate the selected causal mic EQ on speech, control utterances and unfiltered command Opus."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import fftconvolve

import check_speech_mask as diagnostic
import compare_selected_room_transcripts as readers
import fit_spoken_wake_eq as fit

ch, room, gap = fit.ch, fit.room, fit.selection.gap
OUT = ch.RESULTS/(fit.STEM+"-replay.json")


def main():
    spec = ch.read(fit.FIT)
    coeff = np.array(spec["taps"][spec["selected"]])
    frozen = ch.read(fit.selection.FIT)
    assert frozen["pool_sha256"]==ch.sha(fit.selection.PACK)
    bank = {r["index"]:x for r,x in fit.selection.pool()}
    prior = ch.read(readers.replay.OUT)
    known = ch.read(readers.OUT)
    controls = ch.read(diagnostic.INTERFERENCE)
    small,asr = ch.load_reader()
    info = known["turbo"]
    assert asr==prior["asr"]==known["small_files"] and all(ch.sha(readers.audit.TURBO/n)==h for n,h in info["files"].items())
    turbo = WhisperModel(str(readers.audit.TURBO),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder = gap.uplink.decoder_api()
    native = room.rooms.c161_binding()
    block = gap.study.voices.timing.block
    records, noise_rows, captures = [],[],[]
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    pack = OUT.with_suffix(".zip")
    tmp = pack.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for (s,_,kind,h,bg,cmd),prev in zip(room.room_inputs(),prior["records"],strict=True):
            for voice,choices in frozen["choices"].items():
                for role in ("target","nick"):
                    index = choices["selected"][role]
                    cue = bank[index]
                    end = s["end"]
                    start = end-len(cue)
                    wet = fftconvolve(cue.astype(float),h)
                    mix = bg+cmd
                    mix[start:start+len(wet)] += wet
                    pcm = room.quantize(mix*s["gain"])
                    base = prev["cues"][str(index)]
                    assert ch.digest(pcm)==base["pcm_sha256"]
                    filtered,clips = fit.process(pcm,coeff)
                    raw,_ = diagnostic.trace(native,pcm)
                    eq,codes = diagnostic.trace(native,filtered)
                    assert raw==base["c161"]
                    arms = {"raw":raw,"eq":eq,"union":sorted(set(raw+eq))}
                    row = dict(kind=kind,role=role,voice=voice,command=s["carrier"],command_voice=s["voice"],
                               pcm_sha256=ch.digest(pcm),eq_sha256=ch.digest(filtered),eq_codes_sha256=codes,
                               clips=clips, start=start,command_at=end,routes={})
                    cache = {}
                    for name,events in arms.items():
                        opened = block(events[0]) if events else None
                        timely = opened is not None and start<=opened<end
                        result = dict(events=events,opened=opened,timely=timely)
                        c = summary[kind][role][name]
                        c.update(trials=1,hits=opened is not None,timely=timely)
                        if role=="target" and opened is not None:
                            if opened not in cache:
                                old = base["routes"]["c161"]
                                if old.get("hits") and opened==old["hits"][0]:
                                    k=old["capture"]
                                    old_score=known["natural"][str(k)]
                                    cap=prior["captures"][k]
                                    assert cap["lanes"]["selected"]["decoded_sha256"]==old_score["decoded_sha256"]
                                    cache[opened]=dict(old_capture=k, small=old_score["small"]["complete_suffix"],
                                                       turbo=old_score["turbo"]["complete_suffix"])
                                else:
                                    k=len(captures)
                                    cap=gap.replay.capture(small,decoder,work,logs,k,"eq",pcm,opened,s["carrier"])
                                    text=logs.read(f"opus/{k:03}-eq.log").decode().replace("\r\n","\n")
                                    for lane,label in enumerate(("selected","raw")):
                                        packets=gap.uplink.packets(text,k,cap["emitted"]-cap["first"],lane)
                                        wave=gap.uplink.decode(decoder,packets)
                                        saved=cap["lanes"][label]
                                        assert len(packets)==saved["packets"] and ch.digest(wave)==saved["decoded_sha256"]
                                        if lane==0: large=gap.uplink.recognition(turbo,wave,s["carrier"])
                                    captures.append(dict(id=k,capture=cap,turbo=large,source=ch.digest(pcm),opened=opened))
                                    cache[opened]=dict(new_capture=k,small=cap["lanes"]["selected"]["asr"]["complete_suffix"],
                                                       turbo=large["complete_suffix"])
                            result.update(cache[opened])
                            for model in ("small","turbo"):
                                c[model+"_complete"]+=result[model]
                                c[model+"_timely_complete"]+=result[model] and timely
                        row["routes"][name]=result
                    records.append(row)
            print(len(records),kind,flush=True)
        for p in sorted(work.glob("*-attempt*.gz")):
            logs.write(p,"retry/"+p.name)
    for group in ("daily","bot"):
        for (meta,x,old,_,_),saved in zip(diagnostic.everyday.sources(group),controls["records"][group],strict=True):
            assert ch.digest(x)==saved["pcm_sha256"] and old==saved["independent"]
            y,clips=fit.process(x,coeff)
            events,_=diagnostic.trace(native,y)
            row=dict(group=group,index=saved["index"],pcm_sha256=ch.digest(x),eq_sha256=ch.digest(y),clipped=clips,
                     raw=old,eq=events,spans=[])
            for span in meta["source"]["spans"]:
                item=dict(role=span["address"],text=span["text"])
                for name,ev in (("raw",old),("eq",events),("union",sorted(set(old+events)))):
                    hits=[block(t) for t in ev if span["start"]<=block(t)<span["end"]+6400]
                    item[name]=hits
                    summary[group][span["address"]][name].update(trials=1,hits=bool(hits),events=len(hits))
                row["spans"].append(item)
            noise_rows.append(row)
    assert len(records)==256 and len(noise_rows)==100
    with ZipFile(tmp) as z: assert z.testzip() is None
    tmp.replace(pack)
    ch.save(OUT,dict(records=records,interference=noise_rows,captures=captures,summary=summary,small=asr,turbo=info,
                     archive_sha256=ch.sha(pack),sources={p.name:ch.sha(p) for p in (fit.FIT,readers.replay.OUT,
                     readers.OUT,diagnostic.INTERFERENCE,Path(__file__))},
                     policy="Frozen C161 on raw vs one EQ stream; union is the earliest event of two independent "
                     "instances, with no measured dual-instance device cost. All commands uploaded from raw mic "
                     "at actual block OPEN. Common baseline captures reused on matching source and clock, others "
                     "encoded in P4 QEMU. Daily and bot playback judged on pre-existing spans; no self-reference."))


if __name__=="__main__":
    main()
