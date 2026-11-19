"""Require acoustic evidence of the panel cue for a software-only self-play veto."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import check_panel_playout_event as tx

ch, panel, study = tx.ch, tx.panel, tx.study
OUT = ch.RESULTS/"4090-c161-panel-echo-confirmation.json"
BOARD = OUT.with_name(OUT.stem+"-p4.json")
WORK = Path(".local/wakeword/c161-panel-echo-confirmation")
HEADER = tx.HEADER.with_name("c161_panel_echo_confirmation.h")
SKEW = 1280


def admit(raw, pair, echoes, ref):
    result = tx.apply(raw,0,ref)
    intervals = [(max(pair,m),m+study.VETO) for m in echoes if pair and abs(pair-m)<=SKEW]
    result["allowed_model"] = [t for t in raw["c161"] if all(t<start or t>end for start,end in intervals)]
    hits = sorted(result["allowed_model"]+result["allowed_fixed"])
    result["opened"] = [(hits[0]+255)//256*256] if hits else []
    return result


def room_trials():
    v5,bca = study.panel_wave()
    rows = ch.read(tx.HOST)["single_panel"]
    for (s,signals,_),old in zip(study.host.study.established({"v5":v5,"bca":bca}),rows,strict=True):
        x = signals["bca"]
        raw = old["result"]
        assert old["source"]==s and raw["pcm_sha256"]==ch.digest(x)
        # The paired ideal digital playback track includes the scene's clock skew.
        ref = study.host.study.prior.level.guard.refs(s,"bca",{"bca":bca})
        assert raw["reference_sha256"]==ch.digest(ref)
        pair, = raw["panel_pairs"]
        yield s,x,raw,pair


def host_main():
    prev = ch.read(tx.OUT)
    known = {r["host_index"]:r for r in prev["records"]}
    rows, rooms, spoken, stats = [], [], [], defaultdict(Counter)
    with TemporaryDirectory() as folder:
        mic_pair = study.reference_binding(Path(folder)/"pair")
        def measure(group,mic,raw,pair):
            echoes = [e["t"] for e in study.host.detect(mic_pair,[mic])]
            ref = np.zeros_like(mic)
            none = tx.apply(raw,0,ref)["opened"]
            timer = tx.apply(raw,pair,ref)["opened"]
            r = admit(raw,pair,echoes,ref)
            stats[group].update(trials=1,unprotected=int(bool(none)),timer=int(bool(timer)),
                                confirmed=int(bool(r["opened"])),mic_pair=int(bool(echoes)))
            return dict(pair=pair,echoes=echoes,unprotected=none,timer=timer,confirmed=r["opened"])
        for i,(s,name,role,mic,real) in enumerate(study.simultaneous()):
            if i not in known: continue
            old = known[i]
            raw = old["raw"]
            assert raw["pcm_sha256"]==ch.digest(mic)
            pair = 0 if role=="external" else s["local_at"]+prev["pair_at_in_dry"]
            row = measure("overlap."+role,mic,raw,pair)
            assert (row["timer"],row["unprotected"])==(old["clocks"]["0"],old["clocks"]["None"])
            shifts = {}
            for lag in tx.LAGS:
                shifted = pair+lag if pair else 0
                shifted_open = admit(raw,shifted,row["echoes"],np.zeros_like(mic))["opened"]
                shifts[str(lag)] = shifted_open
                stats[f"shift.{role}.{lag}"].update(trials=1,opens=int(bool(shifted_open)),timely=int(bool(
                    role!="panel" and shifted_open and s["start"]<=shifted_open[0]<s["command_at"])))
            rows.append(dict(host_index=i,shifts=shifts,**row))
        for i,(s,mic,raw,pair) in enumerate(room_trials()):
            row = measure("rooms."+s["kind"],mic,raw,pair)
            rooms.append(dict(index=i,source=s,pcm_sha256=raw["pcm_sha256"],**row))
        for i,((s,mic),old) in enumerate(zip(tx.spoken_inputs(),prev["spoken"],strict=True)):
            assert ch.digest(mic)==old["raw"]["pcm_sha256"]
            row = measure("isolated_spoken",mic,old["raw"],s["local_at"]+prev["pair_at_in_dry"])
            spoken.append(dict(index=i,source=s,**row))
    assert (len(rows),len(rooms),len(spoken))==(180,672,24)
    ch.save(OUT,dict(records=rows,rooms=rooms,spoken=spoken,summary=stats,
        sources={p.name:ch.sha(p) for p in (tx.OUT,tx.HOST,HEADER,Path(__file__))},
        policy="Missing R. Panel event and microphone B,C pair must be within 80 ms. The software veto begins "
               "when both observations are available, ends 5432 samples after the mic match. Preserve all raw "
               "fixed cue events. Synthetic isolated-speaker natural controls contain actual external C161 speech."))


def guest_main():
    evidence = ch.read(OUT)
    host = ch.read(tx.OUT)
    history = ch.read(tx.diag.BOARD)
    balanced = ch.read(tx.balanced.OUT)
    versions = history["versions"]
    build = WORK/"build"
    binary = build/"nyx_mac_listener.bin"
    model = panel.phase.listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model)==versions["model"] and binary.read_bytes().count(model.read_bytes())==1
    assert panel.grid.gap.uplink.firmware()==versions["codec"]
    small, sid = ch.load_reader()
    assert sid==versions["small"]
    turbo = tx.diag.reader(versions["turbo"])
    decoder = panel.grid.gap.uplink.decoder_api()
    lookup = {r["baseline_index"]:r for r in balanced["records"] if r["mode"]=="balanced"}
    both = {r["host_index"]:r for r in history["records"] if r["name"]=="bca" and r["role"]=="both"}
    known = {r["host_index"]:r for r in host["records"]}
    echo_rows = {r["host_index"]:r for r in evidence["records"]}
    summary, rows = defaultdict(Counter), []
    tmp = BOARD.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = panel.grid.room.rooms.c161_binding()
        mic_detector = study.reference_binding(work/"echo")
        def run(group,idx,s,role,mic,raw,pair,echoes,expected_open=None):
            ref = np.zeros_like(mic)
            pred = admit(raw,pair,echoes,ref)
            if expected_open is not None:
                assert pred["opened"]==expected_open
            row = panel.capture(work,z,len(rows),(idx,s,"bca",role,mic,ref,pred),
                                small,turbo,decoder,dict(bca=build),tx_pair=pair)
            row.update(group=group,transmit_event=pair,mic_pairs=echoes)
            rows.append(row)
            timely = bool(role!="panel" and row["opened"] and s["start"]<=row["opened"][0]<s["command_at"])
            summary[group].update(trials=1,opens=int(bool(row["opened"])),timely=int(timely),small=int(row["small"]),turbo=int(row["turbo"]))
            print(len(rows),group,idx,row["opened"],flush=True)
        for i,(s,name,role,mic,real) in enumerate(study.simultaneous()):
            if i not in known: continue
            old = echo_rows[i]
            raw = known[i]["raw"]
            pair, echoes = old["pair"],old["echoes"]
            if role=="both":
                base = both[i]
                mode,x,_,_,_ = next(tx.balanced.inputs(base))
                assert mode=="balanced" and ch.digest(x)==lookup[base["index"]]["mic_sha256"]
                updated = panel.grid.fit.fixed.ch.routes(matcher,native,x)
                mic_pairs = [e["t"] for e in study.host.detect(mic_detector,[x])]
                run("balanced."+str(s["external_db"]),i,s,role,x,updated,pair,mic_pairs)
            elif (role=="panel" and old["unprotected"]) or (role=="external" and s["offset"]==0):
                run(role,i,s,role,mic,raw,pair,echoes,old["confirmed"])
        # Preserve each broader-room failure. Add the first successful suppression in each background type.
        success = set()
        for i,((s,mic,raw,pair),old) in enumerate(zip(room_trials(),evidence["rooms"],strict=True)):
            if old["confirmed"] or (old["unprotected"] and s["kind"] not in success):
                if not old["confirmed"]: success.add(s["kind"])
                run("room."+s["kind"],i,s,"panel",mic,raw,pair,old["echoes"],old["confirmed"])
        for i,((s,mic),old) in enumerate(zip(tx.spoken_inputs(),evidence["spoken"],strict=True)):
            run("isolated_spoken",i,s,"external",mic,host["spoken"][i]["raw"],old["pair"],old["echoes"],old["confirmed"])
        s,_,_,mic,_,_,_=tx.single.false_alert()
        s.update(command=s["carrier"],start=0)
        raw = panel.grid.fit.fixed.ch.routes(matcher,native,mic)
        run("across",0,s,"external",mic,raw,0,[],[])
    assert sum(summary["balanced."+str(db)]["trials"] for db in (-6,0,6))==60
    with ZipFile(tmp) as z: assert z.testzip() is None
    archive=BOARD.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(BOARD,dict(records=rows,summary=summary,firmware=ch.sha(binary),model=versions["model"],
        small=sid,turbo=versions["turbo"],archive_sha256=ch.sha(archive),
        sources={p.name:ch.sha(p) for p in (OUT,tx.balanced.OUT,HEADER,Path(__file__),Path(panel.__file__),
               Path(__file__).with_name("build_panel_marker_order.sh"))},
        policy="Physical R zero for all trials. Replay all sixty sender-balanced overlaps, zero-reference panel "
               "false wakes, simultaneous external controls, every surviving false open in 672 rooms, the isolated "
               "external spoken controls and the original speech regression on the corroborated firmware."))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--p4",action="store_true")
    if p.parse_args().p4: guest_main()
    else: host_main()
