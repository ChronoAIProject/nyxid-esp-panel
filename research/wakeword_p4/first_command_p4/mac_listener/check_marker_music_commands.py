"""Replay the normalized candidates on music, then apply the frozen server subtraction to the same packets."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from faster_whisper import WhisperModel

import subtract_fixed_cue_uplink as server
import package_marker_rms_ab as package

study, normal, prior, grid, ch = server.strong.study, server.normal, server.prior, server.grid, server.ch
OUT = ch.RESULTS/(study.fit.STEM+"-music-commands.json")


def main():
    dry=normal.test.tracks()
    normalized=[r for r in ch.read(normal.test.OUT)["records"] if r["group"]=="other_music"]
    base=[r for r in ch.read(study.OUT)["records"] if r["group"]=="other_music"]
    handoff={r["index"] for r in ch.read(package.REPORT)["cases"]}
    subtraction=server.CueSubtractor(dry["v5"])
    small,identity=ch.load_reader()
    settings=ch.read(server.OUT)
    info=settings["turbo"]
    model_dir=prior.audio.prior.context.prior.audit.TURBO
    assert identity==settings["small"] and all(ch.sha(model_dir/n)==h for n,h in info["files"].items())
    turbo=WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder=grid.gap.uplink.decoder_api()
    archive=OUT.with_suffix(".zip")
    tmp=archive.with_suffix(".tmp")
    rows,stats=[],defaultdict(lambda:defaultdict(Counter))
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        matcher=ch.binding(work/"fixed.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        for i,(s,waves,_) in enumerate(study.more_music(dry)):
            ri,rem=divmod(i,22)
            song,level=divmod(rem,2)
            assigned=ri==(3*song+1)%28 and level==song%2
            groups=(["assigned"] if assigned else [])+(["handoff"] if i in handoff else [])
            if not groups: continue
            assert s==base[i]["source"] and normalized[i]["index"]==i
            names=["rms","peak"] if assigned else ["rms"]
            if i in handoff: names.append("v5")
            for name in names:
                expected=(base if name=="v5" else normalized)[i]["routes"][name]
                pcm=waves[name]
                assert ch.digest(pcm)==expected["pcm_sha256"]
                k=len(rows)
                meta=dict(grid_index=i,carrier=s["command"],cue_at=s["start"],command_at=s["command_at"],track=s["track"])
                ref=np.zeros_like(pcm)
                for attempt in range(3):
                    try:
                        board=prior.p4.phase.probe(work,meta,pcm,ref,matcher,pair,k)
                        break
                    except (AssertionError,TimeoutError):
                        z.write(work/"serial.log",f"retry/{k:03}-{attempt}.txt")
                        if attempt==2: raise
                want=[] if expected["opened"] is None else [expected["opened"]]
                assert board["opened"]==want and board["c161"]==expected["c161"] and board["fixed"]==[e["t"] for e in expected["fixed"]]
                z.writestr(f"events/{k:03}.txt","\n".join(board["uart_events"])+"\n")
                row=dict(index=k,groups=groups,name=name,board=board)
                if want:
                    cap=prior.p4.phase.handoff.bench.fifo(work,z,k,pcm,ref,board,small,decoder,True,True)
                    x,count=prior.audio.decoded(z,f"{k:02}-opus.log",decoder,k,cap["samples"],0,cap["pcm_sha256"])
                    assert count==cap["packets"]
                    first=max(0,want[0]//160*160-32000)
                    y,detail=subtraction.apply(x,want[0]-first,matcher)
                    original_turbo=grid.gap.uplink.recognition(turbo,x,s["command"])
                    views={}
                    for size,model,old in (("small",small,cap["recognition"]),("turbo",turbo,original_turbo)):
                        new=grid.gap.uplink.recognition(model,y,s["command"])
                        views[size]=dict(original=old,subtracted=new)
                    row.update(uplink=cap,output_sha256=ch.digest(y),fit=detail,views=views)
                for group in groups:
                    if group=="assigned" and name=="v5" or group=="handoff" and name=="peak": continue
                    c=stats[group][name]
                    c.update(trials=1,opens=bool(want),timely=expected["timely"])
                    for size in ("small","turbo"):
                        for view in ("original","subtracted"):
                            c[size+"_"+view]+=bool(want and row["views"][size][view]["complete_suffix"])
                rows.append(row)
                print(k,i,name,want,flush=True)
    with ZipFile(tmp) as z:
        assert stats["assigned"]["rms"]["trials"]==11 and stats["handoff"]["rms"]["trials"]==5 and z.testzip() is None
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=stats,archive_sha256=ch.sha(archive),small=identity,turbo=info,
                     sources={p.name:ch.sha(p) for p in (server.OUT,Path(server.__file__),normal.test.OUT,study.OUT,
                                                         package.REPORT,Path(__file__))},
                     policy="Previously assigned eleven music excerpts, plus five cases selected for the Mac "
                     "package. Use the original QEMU listener and codec. Compare the exact archived Opus with "
                     "unchanged regularized subtraction, retaining misses in the denominator. Report the two "
                     "groups separately; the handoff selection is purposefully enriched for improved wakes."))


if __name__=="__main__":
    main()
