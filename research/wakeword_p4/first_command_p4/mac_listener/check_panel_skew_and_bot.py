"""Verify signed clock shifts, ordinary speech and natural Bot demonstrations on the middle-marker P4 build."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import check_panel_middle_p4 as middle
import music_proposal_replay as everyday

ch, tx, pair, single = middle.ch, middle.tx, middle.pair, middle.single
OUT=ch.RESULTS/"4090-c161-panel-clock-skew-bot-p4.json"


def main():
    raw_history=ch.read(tx.OUT)
    raw_by_id={r["host_index"]:r for r in raw_history["records"]}
    baseline=ch.read(single.OUT)
    build=middle.WORK/"build"
    firmware=ch.sha(build/"nyx_mac_listener.bin")
    model=tx.panel.phase.listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model)==tx.panel.phase.listener.C161 and (build/"nyx_mac_listener.bin").read_bytes().count(model.read_bytes())==1
    rows,stats=[],defaultdict(Counter)
    archive=OUT.with_suffix(".zip")
    tmp=archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        detector=tx.single.timed.binding(work)
        native=tx.panel.grid.room.rooms.c161_binding()
        def run(group,idx,s,x,raw,nominal,delta):
            clock=nominal+delta if nominal else 0
            hits=single.hits(x,clock)
            ref=np.zeros_like(x)
            known=pair.admit(raw,clock,hits,ref)
            # These are veto/negative controls. Record OPEN and timing without transcribing a false call.
            row=tx.panel.capture(work,z,len(rows),(idx,s,"bca","panel",x,ref,known),
                                 None,None,None,dict(bca=build),tx_pair=clock)
            row.update(group=group,transmit_event=clock,nominal=nominal,delta=delta,c_hits=hits)
            rows.append(row)
            stats[f"{group}.{delta}"].update(trials=1,opens=int(bool(row["opened"])))
        for i,(s,n,role,x,_) in enumerate(tx.study.simultaneous()):
            if n=="bca" and role=="panel" and i in raw_by_id and raw_by_id[i]["clocks"]["None"]:
                for lag in tx.LAGS: run("panel",i,s,x,raw_by_id[i]["raw"],s["local_at"]+raw_history["pair_at_in_dry"],lag)
                break
        for i,((s,x,raw,c),hist) in enumerate(zip(pair.room_trials(),baseline["rooms"],strict=True)):
            if hist["bc_open"]!=hist["single_open"]:
                for lag in tx.LAGS: run("room_residual",i,s,x,raw,c,lag)
        bot_history=[r for r in ch.read(tx.single.timed.OUT)["conversations"] if r["group"]=="bot"]
        for (meta,pcm,c161,_,_),old in zip(everyday.sources("bot"),bot_history,strict=True):
            assert ch.digest(pcm)==old["pcm_sha256"] and old["c161"]==c161
            spans=[s for s in old["spans"] if s["address"]=="explicit_mention"]
            assert len(spans)==2
            begin=(spans[0]["start"]-16000)//256*256
            x=pcm[begin:min(len(pcm),spans[-1]["end"]+16000)].copy()
            raw=tx.panel.grid.fit.fixed.ch.routes(detector,native,x)
            s=dict(command="Bot demonstration",start=spans[0]["start"]-begin,command_at=spans[-1]["end"]-begin,
                   spans=spans,source_offset=begin,full_pcm_sha256=old["pcm_sha256"])
            run("bot",old["index"],s,x,raw,0,0)
            run("bot_clock_stress",old["index"],s,x,raw,spans[-1]["start"]-begin,0)
        s,_,_,x,ref,_,_=tx.single.false_alert()
        s.update(command=s["carrier"],start=0)
        raw=tx.panel.grid.fit.fixed.ch.routes(detector,native,x)
        for tag,c in (("across",0),("across_clock_stress",64136)):
            run(tag,0,s,x,raw,c,0)
    with ZipFile(tmp) as z: assert z.testzip() is None
    tmp.replace(archive)
    assert len(rows)==37
    ch.save(OUT,dict(records=rows,summary=stats,archive_sha256=ch.sha(archive),firmware=firmware,
        sources={p.name:ch.sha(p) for p in (single.OUT,tx.single.timed.OUT,tx.OUT,middle.HEADER,Path(__file__))},
        policy="All nine predeclared signed offsets for the first self-triggering overlap panel and both old "
               "air-conditioner residuals. Replay the two contiguous natural Bot demonstrations in each of four "
               "archived streams with one second of preceding context; also inject a synthetic clock event for "
               "stress. The exact across excerpt has absent and stress-clock controls. Physical R is zero."))


if __name__=="__main__":
    main()
