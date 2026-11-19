"""Play the original and enhanced isolated cues with the same M and either silent or synchronous R."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve

import check_connected_marker_gains as gain
import score_connected_overlap as ordinary

ch, tx, study = gain.ch, gain.tx, gain.study
OUT = ch.RESULTS/"4090-c161-eight-voice-self-play.json"


def speech(matcher,native):
    original = ch.read(ordinary.OUT)
    known = {(r["group"],r["pcm_sha256"]): r for r in original["controls"]}
    rows, summary = [], defaultdict(Counter)
    for group in ("daily","bot"):
        for meta,pcm,events,_,_ in ordinary.base.sources(group):
            row = known.pop((group,ch.digest(pcm)))
            assert row["source"] == meta["source"]
            result = study.playback.host_result(matcher,native,pcm,0)
            assert result["c161"] == events == row["events"]["baseline"]
            decisions = sorted(result["allowed_fixed"] + result["allowed_model"])
            used = set()
            for span in row["source"]["spans"]:
                times = [t for t in decisions if span["start"] <= t < span["end"]+6400]
                used.update(times)
                summary[group+"."+span["address"]].update(trials=1,triggered=int(bool(times)),events=len(times))
            summary[group+".outside"].update(events=len(set(decisions)-used))
            rows.append(dict(group=group, index=meta["index"], result=result))
    assert not known and len(rows)==100
    return rows,summary


def events(work,build,archive,i,mic,ref):
    phase = tx.panel.phase
    # The generated external mark has no B,C,A panel-demo TX event.
    payload = struct.pack("<3I",0x5854594e,i,len(mic)) + mic.tobytes() + ref.tobytes() + struct.pack("<I",0)
    serial = work/"serial.log"
    for attempt in range(3):
        try:
            phase.listener.boot(build,serial,work/"image.bin",b"LISTENER PASS",(2<<20,payload))
            log = serial.read_text()
            assert phase.listener.values(log,"READY") == [i,len(mic),256,81600]
            finish = phase.listener.values(log,"FINISH")
            blocks = (len(mic)+255)//256
            assert finish[3] == blocks and finish[0] == blocks*256//160*160
            break
        except (AssertionError,TimeoutError,ValueError):
            if serial.exists():
                archive.write(serial,f"retry/{i:03}-{attempt}.log")
            if attempt==2:
                raise
    lines = [x for x in log.splitlines() if x.startswith(("READY ","OPEN ","HITS ","V5_ALLOW ","C161_ALLOW ",
                 "CUE_ROUTE ","FINISH ","LISTENER PASS"))]
    archive.writestr(f"events/{i:03}.txt","\n".join(lines)+"\n")
    opened = [int(x.split()[1]) for x in lines if x.startswith("OPEN ")]
    return dict(index=i,opened=opened,finish=finish,mic_sha256=ch.digest(mic),reference_sha256=ch.digest(ref),
                uart_sha256=ch.sha(serial))


def main():
    baseline = ch.read(gain.direction.OUT)
    enhanced = ch.read(gain.OUT)
    assert ch.sha(gain.PACK)==enhanced["pack_sha256"] and enhanced["baseline_sha256"]==ch.sha(gain.direction.OUT)
    known = {(r["batch"],r["source"]["index"]):r["source"] for r in baseline["records"]
             if r["role"]=="marked" and r["profile"]=="flat" and r["source"]["external_db"]==0}
    build = gain.direction.mixed.panel.middle.WORK/"build"
    firmware = ch.sha(build/"nyx_mac_listener.bin")
    assert firmware==ch.read(study.GUEST)["firmware"]
    rows = []
    archive = OUT.with_suffix(".zip")
    native = tx.panel.grid.room.rooms.c161_binding()
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        regressions, counts = speech(matcher,native)
        with ZipFile(gain.PACK) as pack, ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            manifest = json.loads(pack.read("manifest.json"))
            for r in manifest["records"]:
                if "wake_only" not in r:
                    continue
                s = known[r["batch"],r["index"]]
                (h,room,room_sha),_ = tx.diag.room(s["geometry"])
                assert room_sha==s["panel_rir_sha256"]
                source = study.prefix.balanced.PLAY if r["batch"]=="first" else study.PACK
                with ZipFile(source) as prior:
                    original = study.playback.read_wav(prior,r["original"])[32000:r["command_at"]]
                new = study.playback.read_wav(pack,r["wake_only"])
                waves = {key:fftconvolve(x.astype(float),h) for key,x in (("original",original),("enhanced",new))}
                # One path and microphone gain per pair. Reference retains the original digital samples.
                scale = 16000/np.max(np.abs(waves["original"]))
                for version,dry in (("original",original),("enhanced",new)):
                    wet = np.rint(waves[version]*scale)
                    assert wet.min()>=-32768 and wet.max()<=32767
                    mic = np.pad(wet.astype("<i2"),(32000,32000))
                    ref = np.zeros_like(mic)
                    ref[32000:32000+len(dry)]=dry
                    expected = study.playback.host_result(matcher,native,mic,0)
                    for lane,reference in (("silent",np.zeros_like(ref)),("synchronous",ref)):
                        result = events(work,build,logs,len(rows),mic,reference)
                        if lane=="silent":
                            assert result["opened"]==expected["opened"]
                        rows.append(dict(result,batch=r["batch"],voice=r["voice"],version=version,lane=lane,
                                         dry_sha256=ch.digest(dry),room=room,room_sha256=room_sha,gain=scale))
                print("self",r["voice"],flush=True)
        assert len(rows)==32
        with ZipFile(work/archive.name) as z:
            assert z.testzip() is None
        ch.save(work/OUT.name,dict(records=rows,ordinary=regressions,ordinary_summary=counts,firmware=firmware,
                model=baseline["model"],archive_sha256=ch.sha(work/archive.name),
                sources={p.name:ch.sha(p) for p in (gain.PACK,gain.OUT,gain.direction.OUT,ordinary.OUT,Path(__file__))},
                policy="Use only the isolated external cue and the assigned panel room path. Compare the same M "
                       "with ideal digital R or zero R on the same P4 firmware. No software BCA demo event, "
                       "background, panel speech or new FAR hours. Ordinary speech replays are complete original "
                       "host input streams with silent R; report all marker and C161 allows including outside spans."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
