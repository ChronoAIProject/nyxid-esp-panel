"""Test a causal sender-side panel-demo event when the physical reference channel disappears."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_panel_balanced_playback as balanced
import check_timed_final_p4 as single

diag, study, panel, ch = balanced.diag, balanced.study, balanced.panel, balanced.ch
HOST = ch.RESULTS/"4090-c161-panel-marker-order-timed.json"
OUT = ch.RESULTS/"4090-c161-panel-playout-event.json"
BOARD = OUT.with_name(OUT.stem+"-p4.json")
WORK = Path(".local/wakeword/c161-panel-playout-event")
HEADER = single.timed.HEADER.with_name("c161_panel_demo_guard.h")
LAGS = (-6400,-3200,-1280,-256,0,256,1280,3200,6400)
SPOKEN = ch.RESULTS/"hye-nyx-c161-lewis-200ms.wav"


def apply(raw, pair, ref):
    fixed = [e["t"] for e in raw["fixed"]]
    c161 = raw["c161"]
    model = [t for t in c161 if not pair or t<pair or t>pair+study.VETO]
    allows = sorted(fixed+model)
    return dict(c161=c161,fixed=raw["fixed"],pcm_sha256=raw["pcm_sha256"],reference_sha256=ch.digest(ref),
                allowed_fixed=fixed,allowed_model=model,opened=[(allows[0]+255)//256*256] if allows else [])


def event_time():
    with TemporaryDirectory() as folder:
        pair = study.reference_binding(Path(folder)/"ref")
        dry = study.panel_wave()[1]
        event, = study.host.detect(pair,[dry])
        return event["t"]


def spoken_inputs():
    pcm, rate = sf.read(SPOKEN,dtype="int16")
    assert rate==16000
    # Synthetic address and command on a quiet raw microphone. Panel audio is
    # acoustically isolated, e.g. headphones; only its actual playback clock remains.
    bank = diag.sources()[-1]
    commands = sorted(k for k in bank if k[1].startswith(("Raise ","Read ")))[:4]
    for voice,cmd in commands:
        for delta in (-8000,0,4000,8000,16000,40000):
            start = 32000+delta
            x = np.r_[np.zeros(start,np.int16),pcm,bank[voice,cmd],np.zeros(8000,np.int16)]
            yield dict(source="isolated_speaker",voice=voice,command=cmd,local_at=32000,start=start,
                       command_at=start+len(pcm),offset=delta),x


def host_main():
    baseline = ch.read(HOST)
    e = event_time()
    rows, spoken, totals = [], [], defaultdict(Counter)
    for i,((s,name,role,mic,real),known) in enumerate(zip(study.simultaneous(),baseline["records"],strict=True)):
        if name!="bca":
            continue
        old = known["result"]
        assert s==known["source"] and ch.digest(mic)==old["pcm_sha256"] and ch.digest(real)==old["reference_sha256"]
        active = role!="external"
        assert old["panel_pairs"]==([s["local_at"]+e] if active else [])
        zero = np.zeros_like(real)
        vals = {}
        for lag in (None,*LAGS):
            pair = s["local_at"]+e+lag if lag is not None and active else 0
            result = apply(old,pair,zero)
            vals[str(lag)] = result["opened"]
            opened = result["opened"]
            timely = bool(opened and role!="panel" and s["start"]<=opened[0]<s["command_at"])
            totals[f"{role}.{lag}"].update(trials=1,opens=int(bool(opened)),timely=int(timely),
                before_external=int(bool(role!="panel" and opened and opened[0]<s["start"])))
        rows.append(dict(host_index=i,source=s,role=role,clocks=vals,raw=old))
    with TemporaryDirectory() as folder:
        work = Path(folder)
        detector = single.timed.binding(work)
        native = panel.grid.room.rooms.c161_binding()
        for s,x in spoken_inputs():
            raw = panel.grid.fit.fixed.ch.routes(detector,native,x)
            zero = np.zeros_like(x)
            quiet = apply(raw,0,zero)
            guarded = apply(raw,s["local_at"]+e,zero)
            spoken.append(dict(source=s,raw=raw,without_event=quiet["opened"],with_event=guarded["opened"]))
    ch.save(OUT,dict(records=rows,spoken=spoken,summary=totals,pair_at_in_dry=e,guard_samples=study.VETO,
        sources={p.name:ch.sha(p) for p in (HOST,SPOKEN,HEADER,Path(__file__))},
        policy="Physical R zero. Known B,C marker event from played dry panel PCM only; absent for external-only. "
               "Shift the event on the capture clock by fixed sample offsets. Natural synthetic diagnostic uses "
               "an acoustically isolated playing panel to attribute any suppressed raw C161 hits to the real external voice."))


def guest_main():
    report = ch.read(OUT)
    original = ch.read(diag.BOARD)
    versions = original["versions"]
    base_build = balanced.BUILD
    build = WORK/"build"
    assert ch.sha(base_build/"nyx_mac_listener.bin")==versions["panel_order"]
    model = panel.phase.listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model)==versions["model"] and (build/"nyx_mac_listener.bin").read_bytes().count(model.read_bytes())==1
    assert panel.grid.gap.uplink.firmware()==versions["codec"]
    small, sid = ch.load_reader()
    assert sid==versions["small"]
    turbo = diag.reader(versions["turbo"])
    decoder = panel.grid.gap.uplink.decoder_api()
    summary, rows = defaultdict(Counter), []
    cases = {r["host_index"]:r for r in report["records"]}
    e = report["pair_at_in_dry"]
    tmp = BOARD.with_suffix(".tmp")
    def run(work,z,group,idx,s,role,mic,ref,raw,pair):
        expected = apply(raw,pair,ref)
        row = panel.capture(work,z,len(rows),(idx,s,"bca",role,mic,ref,expected),
                      small,turbo,decoder,dict(bca=build),tx_pair=pair)
        row.update(group=group,transmit_event=pair)
        rows.append(row)
        stats = summary[group]
        timely = bool(row["opened"] and s["start"]<=row["opened"][0]<s["command_at"] and role!="panel")
        stats.update(trials=1,opens=int(bool(row["opened"])),timely=int(timely),small=int(row["small"]),turbo=int(row["turbo"]))
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        for i,(s,name,role,mic,real) in enumerate(study.simultaneous()):
            if i not in cases:
                continue
            old = cases[i]
            assert ch.digest(mic)==old["raw"]["pcm_sha256"]
            zero = np.zeros_like(real)
            pair = 0 if role=="external" else s["local_at"]+e
            if role=="both" or (role=="external" and s["offset"]==0) or (role=="panel" and old["clocks"]["None"]):
                run(work,z,"missing."+role,i,s,role,mic,zero,old["raw"],pair)
            if role=="panel" and old["clocks"]["None"] and s["offset"]==0:
                run(work,z,"missing.no_event_fault",i,s,role,mic,zero,old["raw"],0)
            if role=="both" and old["clocks"]["0"]!=old["clocks"]["None"]:
                run(work,z,"missing.both_no_event",i,s,role,mic,zero,old["raw"],0)
        for i,((s,mic),old) in enumerate(zip(spoken_inputs(),report["spoken"],strict=True)):
            if old["with_event"]!=old["without_event"] or (s["offset"]==40000 and i==5):
                run(work,z,"natural.with_event",i,s,"external",mic,np.zeros_like(mic),old["raw"],s["local_at"]+e)
                run(work,z,"natural.without_event",i,s,"external",mic,np.zeros_like(mic),old["raw"],0)
        s,_,_,mic,ref,_,_ = single.false_alert()
        s.update(command=s["carrier"],start=0)
        raw = dict(c161=[],fixed=[],pcm_sha256=ch.digest(mic))
        run(work,z,"original_across",0,s,"external",mic,ref,raw,0)
    assert summary["missing.both"]["trials"]==60
    with ZipFile(tmp) as z:
        assert z.testzip() is None
    archive = BOARD.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(BOARD,dict(records=rows,summary=summary,archive_sha256=ch.sha(archive),model=versions["model"],
        firmware=ch.sha(build/"nyx_mac_listener.bin"),small=sid,turbo=versions["turbo"],sources={p.name:ch.sha(p) for p in
            (OUT,diag.BOARD,HEADER,Path(__file__),Path(panel.__file__),Path(__file__).with_name("build_panel_marker_order.sh"))},
        policy="QEMU combined receiver uses actual zero R for AEC, capture and standard reference guard. Only the "
               "C161 veto receives the independent event at the indicated capture sample. Replay all original 60 "
               "dual-source inputs, all reference-fault panel-only opens, twelve external controls, targeted raw "
               "synthetic C161 losses and the ordinary speech regression. Score the full received Opus."))


if __name__=="__main__":
    import sys
    if sys.argv[1:] == ["--p4"]:
        guest_main()
    else:
        host_main()
