"""Export the isolated +2 dB Bella cue and replay it with fixed-source and self-play controls."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_address_gain_wakes as wakes
import check_bella_daily_overlap as daily
import check_bella_marker_only_self as selfplay
import package_weak_gain_room_audio as sources

gain, fixed, P = wakes.gain, wakes.fixed, daily.P
OUT = gain.RESULTS/"4090-bella-plus2-playback.json"
CUE = gain.RESULTS/"hye-nyx-af-bella-plus2db.wav"
TRACE = OUT.with_suffix(".zip")


def external(original, candidate):
    historical = {(r["index"],r["role"]):r for r in P.read(daily.OUT)["records"]}
    rooms,taps = daily.rooms()
    by_file = {item[0]["file"]:item for region in rooms.values() for item in region}
    for src,key,_,speech in daily.event_inputs("daily"):
        i = int(key)
        before = historical[i,"original"]
        assert P.digest(speech)==before["dry_speech_sha256"] and src["text"]==before["text"]
        _,s,render,ref,active,tracks = by_file[before["file"]]
        talk,levels = daily.tx.balanced.match_level(active,speech)
        assert levels["gain_db"]==before["speech_gain_db"] and P.digest(ref)==before["reference_sha256"]
        start,stop,end = before["speech_start"],before["speech_end"],before["command_at"]
        removed = tracks["absent"].astype(np.int32)
        removed[start:stop] += talk.astype(np.int32)
        for role,pcm in (("original",original),("absent",None),("plus2",candidate)):
            dry = removed.copy()
            if pcm is not None:
                dry[end-len(pcm):end] += pcm
            assert np.array_equal(dry[end:],removed[end:])
            mic,clips = render(daily.board.bella.rear_filter(dry,taps))
            if role in ("original","absent"):
                assert P.digest(mic)==historical[i,role]["mic_sha256"]
            if role=="original":
                continue
            row = dict(group="external",scene=i,db=before["db"],file=before["file"],role=role,
                       text=src["text"],command=s["command"],command_at=end,start=before["cue_start"],
                       dry_suffix_sha256=P.digest(dry[end:]),clipped=clips,
                       original_marker=before["events"]["C"],selection=i in (77,79,90))
            yield f"external/{i:03}/{role}",row,mic,ref


def panel(original, candidate):
    previous = [r for r in P.read(selfplay.OUT)["records"] if r["version"]=="original"]
    for row in previous:
        g = row["geometry"]
        (ir,path,checksum),_ = daily.tx.diag.room(g)
        assert path==row["room"] and checksum==row["room_sha256"]
        for version,wav in (("original",original),("plus2",candidate)):
            wet = np.rint(fftconvolve(wav.astype(float),ir)*row["scale"])
            mic = np.pad(wet.clip(-32768,32767).astype("<i2"),(32000,32000))
            ref = np.zeros_like(mic)
            if row["reference"]=="synchronous":
                ref[32000:32000+len(wav)] = wav
            if version=="original":
                assert (P.digest(mic),P.digest(ref))==(row["mic_sha256"],row["reference_sha256"])
                continue
            yield f"panel/{g}/{row['reference']}",dict(group="panel",geometry=g,room=path,role=row["reference"],
                  clipped=int(np.count_nonzero((wet<-32768)|(wet>32767)))),mic,ref


def main():
    original,sr = sf.read(daily.board.bella.CUE,dtype="int16")
    assert sr==16000 and P.sha(daily.board.bella.CUE)==daily.board.bella.CUE_SHA
    scaled = np.rint(original.astype(float)*10**.1)
    assert np.abs(scaled).max()<32768
    candidate = scaled.astype("<i2")
    plan = P.read(gain.PLAN)
    assert P.sha(gain.PACK)==plan["archive_sha256"]
    with ZipFile(gain.PACK) as z:
        for i in (77,79,90):
            names = [f"{i:03}/{k}.wav" for k in ("nyx/plus2dB","absent/erased")]
            target,absent = [sf.read(BytesIO(z.read(n)),dtype="int32")[0]//16384 for n in names]
            end, = [r["end"] for r in plan["records"] if r["member"]==names[0]]
            extracted = target-absent
            assert np.array_equal(extracted[end-len(candidate):end],candidate)
            assert not np.any(extracted[:end-len(candidate)]) and not np.any(extracted[end:])
    previous = P.read(wakes.OUT)
    assert P.sha(wakes.TRACE)==previous["archive_sha256"]
    known = {r["nyx"]["input_sha256"]:r for r in previous["records"]}
    flash = wakes.fixed_image(P.read(fixed.REPORT),"fixed")
    cache = gain.WORK/"plus2-fixed"/wakes.digest(bytes.fromhex(previous["qemu_sha256"])+flash[:2<<20])
    cache.mkdir(parents=True,exist_ok=True)
    rows,counts=[],defaultdict(Counter)
    with TemporaryDirectory(dir=gain.WORK) as tmp,ZipFile(wakes.TRACE) as imported:
        temp = Path(tmp)
        (temp/CUE.name).write_bytes(sources.wav(candidate,"PCM_16"))
        with ZipFile(temp/TRACE.name,"w",ZIP_DEFLATED) as z:
            for group in (external,panel):
                for label,row,mic,ref in group(original,candidate):
                    block = fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                    key = wakes.digest(block)
                    prior = known.get(key)
                    failed = {}
                    if prior:
                        events = prior["nyx"]["events"]
                        raw = imported.read("p4/fixed/"+prior["label"]+".log")
                        assert wakes.digest(raw)==prior["nyx"]["uart_sha256"]
                    else:
                        for attempt in range(2):
                            try:
                                observations,log,faults = fixed.boot(cache,flash,cache,block,1)
                                break
                            except AssertionError:
                                if attempt: raise
                        raw = log.read_bytes()
                        events = observations[0]["events"]
                        for fault in faults:
                            name = f"failed/{label}/{fault.name}"
                            z.write(fault,name)
                            failed[name]=P.sha(fault)
                    z.writestr(label+".log",raw)
                    row.update(label=label,mic_sha256=P.digest(mic),reference_sha256=P.digest(ref),
                               input_sha256=key,events=events,uart_sha256=wakes.digest(raw),reused=bool(prior),failed=failed)
                    rows.append(row)
                    count = counts[row["group"]+"/"+row["role"]]
                    count.update(cases=1,opened=int(bool(events["O"])))
                    if row["group"]=="external":
                        ts = events["O"]
                        count.update(timely=int(bool(ts) and row["start"]<=ts[0]<row["command_at"]))
                    if len(rows)%24==0: print(len(rows),flush=True)
        assert len(rows)==200 and len({r["label"] for r in rows})==len(rows)
        sources_used = (Path(__file__),gain.PLAN,wakes.OUT,wakes.TRACE,daily.OUT,selfplay.OUT,Path(daily.__file__),
                        daily.board.bella.CUE,gain.ROOT/"experiments/bella_plus2_playback_protocol.md")
        P.save(temp/OUT.name,dict(records=rows,counts=counts,cue=CUE.name,cue_sha256=P.sha(temp/CUE.name),
               pcm_sha256=P.digest(candidate),samples=len(candidate),rate=16000,bytes=(temp/CUE.name).stat().st_size,
               sample_peak=int(np.abs(candidate).max()),qemu_sha256=previous["qemu_sha256"],guest_sha256=previous["guest_sha256"]["fixed"],
               sources={p.name:P.sha(p) for p in sources_used},archive_sha256=P.sha(temp/TRACE.name),
               scope="2 dB selected on three previous misses. 96 historical interruption scenes, 96 erased "
                     "controls, four panel playback paths with two electrical reference conditions. Digital exposure only."))
        for dest in (CUE,OUT,TRACE):
            (temp/dest.name).replace(dest)


if __name__=="__main__":
    main()
