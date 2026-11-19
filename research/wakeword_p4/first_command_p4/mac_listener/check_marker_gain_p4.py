"""Verify a fixed sample of music inputs and paired kitchen/living-room commands in P4 firmware."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import fftconvolve

import check_marker_gain_music as study

ch, prior, grid = study.ch,study.prior,study.grid
OUT = ch.RESULTS / (study.fit.STEM + "-p4.json")


def restore(archive, output, idx, source, mic, reference, expected, small, decoder, native, full):
    event_file = f"events/{idx:03}.txt"
    if event_file not in archive.namelist():
        return None
    uart = archive.read(event_file).decode()
    opened = [int(x) for x in re.findall(r"^OPEN (\d+) ",uart,re.M)]
    assert opened == expected
    packet_file = f"{idx:02}-opus.log"
    if opened and packet_file not in archive.namelist():
        return None
    assert f"READY {idx} {len(mic)} 256 81600\n" in uart and "LISTENER PASS\n" in uart
    actual = [list(map(int,line.split())) for line in re.findall(r"^HITS (.+)$",uart,re.M)]
    c161 = [t for _,t,_ in actual if t>=0]
    fixed = [t for _,_,t in actual if t>=0]
    host = grid.outcome(full,native,mic,dict(start=source["cue_at"],command_at=source["command_at"]))
    assert c161==host["c161"] and fixed==[h["t"] for h in host["fixed"]]
    board = dict(source=source,opened=opened,c161=c161,fixed=fixed,uart_events=uart.splitlines(),
                 mic_sha256=host["pcm_sha256"],reference_sha256=ch.digest(reference),
                 allowed=sorted(int(v) for v in re.findall(r"^(?:V5|C161)_ALLOW (\d+)$",uart,re.M)))
    cap = None
    if opened:
        assert not reference.any()
        finish, = re.findall(r"^FINISH (.+)$",uart,re.M)
        emitted,first,count,blocks,*_ = map(int,finish.split())
        assert count==emitted-first and first==max(0,opened[0]//160*160-32000)
        buffered = np.pad(mic,(0,blocks*256-len(mic)))
        selected = np.pad(buffered,(1440,0))[first:emitted]
        assert len(selected)==count
        data = archive.read(packet_file)
        log = data.decode().replace("\r\n","\n")
        for lane in (0,1):
            packets = grid.gap.uplink.packets(log,idx,count,lane)
            decoded = grid.gap.uplink.decode(decoder,packets)
            if lane==0:
                cap=dict(samples=count,packets=len(packets),pcm_sha256=ch.digest(decoded),fifo_sha256=ch.digest(selected),
                         processed_samples=0,delayed_raw_samples=count,
                         recognition=grid.gap.uplink.recognition(small,decoded,source["carrier"]))
        output.writestr(packet_file,data)
    output.writestr(event_file,uart)
    return board,cap


def main(resume_archive=None):
    dry = study.tracks()
    previous = ch.read(prior.OUT)
    small, identity = ch.load_reader()
    info = previous["turbo"]
    model_dir = prior.audio.prior.context.prior.audit.TURBO
    assert identity == previous["small"] and all(ch.sha(model_dir/k)==h for k,h in info["files"].items())
    turbo = WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder = grid.gap.uplink.decoder_api()
    native = grid.room.rooms.c161_binding()
    rooms = ch.read(study.OUT)
    music = [r for r in rooms["records"] if r["group"]=="other_music"]
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    rows,comparison = [],[]
    totals = defaultdict(Counter)
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z, (
            ZipFile(resume_archive) if resume_archive else nullcontext()) as resume:
        if resume:
            assert resume.testzip() is None
            for path in resume.namelist():
                if "attempt" in path or path.startswith("retries/"):
                    z.writestr("previous-run/"+path,resume.read(path))
        work = Path(folder)
        full = ch.binding(work/"full.so",coded="v5")
        pair = ch.binding(work/"pair.so",coded="aligned")

        def run(cohort,name,source,pcm,ref,expected,local):
            idx = len(rows)
            saved = restore(resume,z,idx,source,pcm,ref,expected,small,decoder,native,full) if resume else None
            if saved is not None:
                board,cap = saved
            else:
                for attempt in range(3):
                    try:
                        board = prior.p4.phase.probe(work,source,pcm,ref,full,pair,idx)
                        break
                    except (AssertionError,TimeoutError):
                        z.write(work/"serial.log",f"retries/{idx}-{attempt}.txt")
                        if attempt==2: raise
                assert board["opened"] == expected
                z.writestr(f"events/{idx:03}.txt","\n".join(board["uart_events"])+"\n")
                cap = None
                if expected and not local:
                    cap = prior.p4.phase.handoff.bench.fifo(work,z,idx,pcm,ref,board,small,decoder,True,True)
            if cap:
                out,count = prior.audio.decoded(z,f"{idx:02}-opus.log",decoder,idx,cap["samples"],0,cap["pcm_sha256"])
                assert count == cap["packets"]
                cap["turbo"] = grid.gap.uplink.recognition(turbo,out,source["carrier"])
            timely = bool(expected and source["cue_at"] <= expected[0] < source["command_at"])
            totals[f"{cohort}.{name}.{'self' if local else 'external'}"].update(trials=1,opens=bool(expected),
                     timely=timely,small_complete=bool(cap and cap["recognition"]["complete_suffix"]),
                     turbo_complete=bool(cap and cap["turbo"]["complete_suffix"]))
            rows.append(dict(group=cohort,name=name,local=local,board=board,uplink=cap,restored=saved is not None))
            print(idx,cohort,name,local,expected,flush=True)
            return rows[-1]

        for i, (s,old,kind,h,bg,cmd) in enumerate(prior.audio.room.room_inputs()):
            end = s["end"]
            start = end-len(dry["v5"])
            baseline = [r for r in previous["records"] if r["group"]=="common" and r["board"]["source"]["scene"]==i]
            assert len(baseline)==2
            meta = dict(cue_at=start,command_at=end,carrier=s["carrier"],scene=i,voice=s["voice"])
            scores = {}
            for name,wav in dry.items():
                scene = bg+cmd
                wet = fftconvolve(wav.astype(float),h)
                scene[start:start+len(wet)] += wet
                pcm = grid.room.quantize(scene*s["gain"])
                route = grid.outcome(full,native,pcm,dict(start=start,command_at=end))
                if name == "markers":
                    expected = [] if route["opened"] is None else [route["opened"]]
                    row = run("common",name,meta,pcm,np.zeros_like(pcm),expected,False)
                else:
                    other = "v5" if name=="v5" else "boosted_v5"
                    row, = [r for r in baseline if r["name"]==other]
                result = row["board"]
                assert result["source"]==meta and result["mic_sha256"]==route["pcm_sha256"]
                assert result["fixed"]==[v["t"] for v in route["fixed"]] and result["c161"]==route["c161"]
                cap = row["uplink"]
                scores[name] = dict(opened=result["opened"],small=bool(cap and cap["recognition"]["complete_suffix"]),
                                    turbo=bool(cap and cap["turbo"]["complete_suffix"]))
            comparison.append(dict(scene=i,kind=kind,voice=s["voice"],command=s["carrier"],scores=scores))
        for i, ((s,signals,_), host) in enumerate(zip(study.more_music(dry),music,strict=True)):
            # One preassigned room and one level per recording, without choosing by outcome.
            ri, rem = divmod(i,22)
            song, level = divmod(rem,2)
            if ri!=(3*song+1)%28 or level!=song%2: continue
            assert host["source"]==s
            meta = dict(cue_at=s["start"],command_at=s["command_at"],carrier=s["command"],room=s["room"],
                        song=s["track"],level_db=s["level_db"],player_ppm=s["player_ppm"],index=i)
            for name,x in signals.items():
                r = host["routes"][name]
                ref = prior.level.guard.refs(s,name,dry)
                assert ch.digest(x)==r["pcm_sha256"] and ch.digest(ref)==r["reference_sha256"]
                ext = [] if r["opened"] is None else [r["opened"]]
                for local,reference,opens in ((False,np.zeros_like(x),ext),(True,ref,r["self_opened"])):
                    row = run("music",name,meta,x,reference,opens,local)
                    assert row["board"]["fixed"]==[v["t"] for v in r["fixed"]] and row["board"]["c161"]==r["c161"]
    with ZipFile(tmp) as z:
        assert len(rows)==16+11*6 and z.testzip() is None
    tmp.replace(archive)
    fw = prior.p4.phase.handoff.test.BUILD/"nyx_mac_listener.bin"
    assert ch.sha(fw)==previous["firmware_sha256"]
    ch.save(OUT,dict(records=rows,comparison=comparison,summary=totals,firmware_sha256=ch.sha(fw),
                     archive_sha256=ch.sha(archive),small=identity,turbo=info,
                     partial_run_sha256=ch.sha(resume_archive) if resume_archive else None,
                     sources={p.name:ch.sha(p) for p in (study.OUT,prior.OUT,study.fit.FIT,Path(__file__))},
                     policy="Current unmodified simulated P4 listener and FIFO/Opus. Original/boosted common "
                     "rows are reidentified from the published gain-control run. Eleven music scenes are "
                     "assigned by song number before reading events; each has external and aligned self playback."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-archive",type=Path)
    main(parser.parse_args().resume_archive)
