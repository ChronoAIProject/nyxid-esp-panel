"""Replay the residual self opens, overlap controls and connected names on the single-marker P4 guard."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_panel_single_echo_evidence as single

pair,tx,ch = single.pair,single.tx,single.ch
OUT=ch.RESULTS/"4090-c161-panel-middle-confirmation-p4.json"
WORK=Path(".local/wakeword/c161-panel-middle-confirmation")
HEADER=tx.HEADER.with_name("c161_panel_middle_marker.h")


def main():
    history=ch.read(single.OUT)
    bc=ch.read(pair.OUT)
    spec=ch.read(tx.OUT)
    v=ch.read(tx.diag.BOARD)["versions"]
    baseline=ch.read(tx.diag.BOARD)["records"]
    firmware=WORK/"build/nyx_mac_listener.bin"
    model=tx.panel.phase.listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model)==v["model"] and firmware.read_bytes().count(model.read_bytes())==1
    compiled=WORK/"src/research/wakeword_p4/first_command_p4/mac_listener/main"
    assert (compiled/HEADER.name).read_bytes()==HEADER.read_bytes()
    assert tx.panel.grid.gap.uplink.firmware()==v["codec"]
    reader,info=ch.load_reader()
    assert info==v["small"]
    turbo=tx.diag.reader(v["turbo"])
    decoder=tx.panel.grid.gap.uplink.decoder_api()
    tmp=OUT.with_suffix(".tmp")
    rows,summary=[],defaultdict(Counter)
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        m=tx.single.timed.binding(work)
        native=tx.panel.grid.room.rooms.c161_binding()
        strict=ch.binding(work/"strict.so",coded="v5")
        prefix=ch.binding(work/"prefix.so",coded="aligned")
        def run(group,i,s,role,mic,ref,raw,clock,events,expect=None,physical=None):
            predicted=pair.admit(raw,clock,events,np.zeros_like(ref))
            if physical is not None:
                # The physical path includes delayed admissions, so apply the extra
                # solo marker veto across the pending-to-admitted interval as well.
                full=tx.study.grid.drift.guarded(raw,strict,prefix,ref)
                from self_guard import expected
                from code_gain_guard import activity_pulses
                partial=sorted(set(full["prefix_reference"])-set(full["qualified_prefix"]))
                def accepted(h):
                    return expected(h,full["reference_events"],activity_pulses(ref),full["qualified_prefix"],partial)
                windows=[(a,a+tx.study.VETO) for a in physical]
                windows.extend((max(clock,h),h+tx.study.VETO) for h in events if clock and abs(h-clock)<=pair.SKEW)
                model_ok=[out for t in raw["c161"] for out in accepted([t]) if not any(a<=out and b>=t for a,b in windows)]
                fixed_ok=accepted([h["t"] for h in raw["fixed"]])
                allows=sorted(fixed_ok+model_ok)
                predicted.update(reference_sha256=ch.digest(ref),allowed_model=model_ok,allowed_fixed=fixed_ok,
                                 opened=[(allows[0]+255)//256*256] if allows else [])
            if expect is not None: assert predicted["opened"]==expect
            r=tx.panel.capture(work,z,len(rows),(i,s,"bca",role,mic,ref,predicted),reader,turbo,decoder,
                               dict(bca=WORK/"build"),tx_pair=clock)
            r.update(group=group,transmit_event=clock,c_hits=events)
            rows.append(r)
            ok=bool(role!="panel" and r["opened"] and s["start"]<=r["opened"][0]<s["command_at"])
            summary[group].update(trials=1,opens=int(bool(r["opened"])),timely=int(ok),small=int(r["small"]),turbo=int(r["turbo"]))
            print(len(rows),group,i,r["opened"],flush=True)

        paired={r["host_index"]:r for r in spec["records"]}
        known={r["host_index"]:r for r in bc["records"]}
        # Use all 60 sender-balanced overlaps with zero R; check real-R on all
        # twelve simultaneous onsets, sharing the earlier firmware's exact input.
        for b in baseline:
            if b["name"]!="bca" or b["role"]!="both": continue
            i=b["host_index"]
            s=b["source"]
            _,mic,ref,_,_=next(tx.balanced.inputs(b))
            raw=tx.panel.grid.fit.fixed.ch.routes(m,native,mic)
            clock=known[i]["pair"]
            events=single.hits(mic,clock)
            run("balanced."+str(s["external_db"]),i,s,"both",mic,np.zeros_like(ref),raw,clock,events)
            if not s["offset"]:
                run("physical."+str(s["external_db"]),i,s,"both",mic,ref,raw,clock,events,physical=paired[i]["raw"]["panel_pairs"])
        for i,(s,n,role,x,ref) in enumerate(tx.study.simultaneous()):
            if i not in paired or role!="panel" or not known[i]["unprotected"]: continue
            clock=known[i]["pair"]
            run("panel",i,s,role,x,np.zeros_like(ref),paired[i]["raw"],clock,single.hits(x,clock),[])
        for i,((s,x,raw,clock),old) in enumerate(zip(pair.room_trials(),history["rooms"],strict=True)):
            if old["bc_open"]!=old["single_open"]:
                run("room_residual",i,s,"panel",x,np.zeros_like(x),raw,clock,old["c_hits"],old["single_open"])
        for i,((s,x),old) in enumerate(zip(tx.spoken_inputs(),history["isolated"],strict=True)):
            if spec["spoken"][i]["with_event"]!=old["single_open"]:
                run("isolated_spoken",i,s,"external",x,np.zeros_like(x),spec["spoken"][i]["raw"],old["tx"],old["c_hits"],old["single_open"])
        plan=ch.read(single.CONNECTED)["records"]
        for old in history["connected"]:
            if not old["raw_hits"] or old["event_to_hit"]!=1600: continue
            i=old["index"]
            s=plan[i]
            x,rate=sf.read(s["path"],dtype="int16")
            assert rate==16000 and ch.digest(x)==old["pcm_sha256"]
            raw=tx.panel.grid.fit.fixed.ch.routes(m,native,x)
            meta=dict(command=s["carrier"],start=s["start"],command_at=s["end"],voice=s["voice"],target=s["target_id"])
            run("connected",i,meta,"external",x,np.zeros_like(x),raw,old["tx"],old["c_hits"],old["single_open"])
        s,_,_,x,ref,_,_=tx.single.false_alert()
        s.update(command=s["carrier"],start=0)
        raw=tx.panel.grid.fit.fixed.ch.routes(m,native,x)
        run("across",0,s,"external",x,ref,raw,0,[],[])
    with ZipFile(tmp) as z: assert z.testzip() is None
    archive=OUT.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=summary,archive_sha256=ch.sha(archive),firmware=ch.sha(firmware),
        model=v["model"],small=info,turbo=v["turbo"],sources={p.name:ch.sha(p) for p in (single.OUT,tx.balanced.OUT,
            HEADER,Path(__file__),Path(__file__).with_name("build_panel_marker_order.sh"))},
        policy="Causal firmware verification with microphone C-template evidence and actual playout clock. "
               "All balanced overlaps with missing physical R, simultaneous starts with real R, all original "
               "15 self-open inputs, both previous room misses, timer-only isolated speech losses, all connected "
               "Nyx/Nicks C161 positives at one prespecified lag, original across speech. Full selected Opus."))


if __name__=="__main__":
    main()
