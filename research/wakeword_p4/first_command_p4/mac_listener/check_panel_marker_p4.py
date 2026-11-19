"""Replay original and panel-only marker orders in the digital P4 listener and its Opus encoder."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

from faster_whisper import WhisperModel

import check_panel_marker_order as study
import check_final_marker_p4 as prior

ch, grid, phase = study.ch, study.grid, prior.phase
WORK = Path(".local/wakeword/c161-panel-marker-order")
OUT = ch.RESULTS/"4090-c161-panel-marker-order-p4.json"


def cases():
    known = ch.read(study.OUT)["records"]
    for i, (trial, old) in enumerate(zip(study.simultaneous(), known, strict=True)):
        meta, name, role, pcm, ref = trial
        assert (meta, name, role) == (old["source"], old["name"], old["role"])
        if role == "both" or (meta["offset"] == 0 and (role != "external" or name == "abc")):
            yield i, meta, name, role, pcm, ref, old["result"]


def capture(work, z, idx, case, small, turbo, decoder):
    source_idx, meta, name, role, mic, ref, known = case
    assert ch.digest(mic) == known["pcm_sha256"] and ch.digest(ref) == known["reference_sha256"]
    build = prior.BUILD if name == "abc" else WORK/"build"
    source = dict(**meta, carrier=meta["command"], cue_at=meta["start"])
    packet = struct.pack("<3I", 0x4652594e, idx, len(mic)) + mic.tobytes() + ref.tobytes()
    serial = work/"serial.log"
    for attempt in range(3):
        try:
            phase.listener.boot(build, serial, work/"image.bin", b"LISTENER PASS", (2<<20, packet))
            log = serial.read_text()
            assert phase.listener.values(log,"READY") == [idx,len(mic),256,81600]
            hits = [tuple(map(int,x.split())) for x in re.findall(r"^HITS (.+)$", log, re.M)]
            assert [c for _,c,_ in hits if c >= 0] == known["c161"]
            assert [f for _,_,f in hits if f >= 0] == [e["t"] for e in known["fixed"]]
            get = lambda key: [int(x) for x in re.findall("^"+key+r" (\d+)(?: |$)", log, re.M)]
            assert get("V5_ALLOW") == known["allowed_fixed"]
            assert get("C161_ALLOW") == known["allowed_model"]
            assert get("OPEN") == known["opened"]
            break
        except (AssertionError, TimeoutError, ValueError):
            if serial.exists():
                z.write(serial, f"retry/{idx:03}-{attempt}.txt")
            if attempt == 2: raise
    events = [s for s in log.splitlines() if s.startswith(("READY ","HITS ","V5_ALLOW ","C161_ALLOW ","OPEN ","CUE_ROUTE ","FINISH ","LISTENER PASS"))]
    z.writestr(f"events/{idx:03}.txt","\n".join(events)+"\n")
    board = dict(opened=known["opened"], source=source)
    timing = phase.listener.values(log,"FINISH")
    result = dict(index=idx, host_index=source_idx, name=name, role=role, source=meta,
                  opened=board["opened"], mic_sha256=known["pcm_sha256"], reference_sha256=known["reference_sha256"],
                  allowed_fixed=known["allowed_fixed"], allowed_model=known["allowed_model"],
                  finish=timing, full_uart_sha256=ch.sha(serial), small=False, turbo=False)
    if board["opened"] and role != "panel":
        cap = phase.handoff.bench.fifo(work,z,idx,mic,ref,board,small,decoder,True,True)
        audio,count = prior.prior.prior.audio.decoded(z, f"{idx:02}-opus.log", decoder,idx,cap["samples"],0,cap["pcm_sha256"])
        assert count == cap["packets"]
        cap["turbo"] = grid.gap.uplink.recognition(turbo,audio,meta["command"])
        result.update(uplink=cap,small=cap["recognition"]["complete_suffix"],turbo=cap["turbo"]["complete_suffix"])
    return result


def main():
    previous = ch.read(prior.OUT)
    versions = previous["versions"]
    binaries = [b/"nyx_mac_listener.bin" for b in (prior.BUILD,WORK/"build")]
    embedded = (phase.listener.WORK/"model/hey_nyx_model.bin").read_bytes()
    assert ch.sha(binaries[0]) == versions["candidate"] and all(b.read_bytes().count(embedded)==1 for b in binaries)
    assert grid.gap.uplink.firmware() == versions["codec"]
    small, identity = ch.load_reader()
    assert identity == versions["small"]
    info = versions["turbo"]
    model_dir = prior.prior.prior.audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n,h in info["files"].items())
    turbo = WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder = grid.gap.uplink.decoder_api()
    versions = dict(original=ch.sha(binaries[0]), panel_order=ch.sha(binaries[1]), model=sha256(embedded).hexdigest(),
                    codec=versions["codec"], small=identity, turbo=info,
                    sources={p.name:ch.sha(p) for p in (study.OUT,Path(__file__),Path(__file__).with_name("build_panel_marker_order.sh"))})
    key = sha256(json.dumps(versions,sort_keys=True).encode()).hexdigest()[:16]
    work = WORK/"runs"/key
    work.mkdir(exist_ok=True,parents=True)
    rows, summary = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as logs:
        for idx,case in enumerate(cases()):
            shard = work/f"{idx:03}.zip"
            if not shard.exists():
                partial = shard.with_suffix(".tmp")
                with ZipFile(partial,"w",ZIP_DEFLATED) as z:
                    row = capture(work,z,idx,case,small,turbo,decoder)
                    z.writestr("case.json",json.dumps(row))
                partial.replace(shard)
            with ZipFile(shard) as z:
                row = json.loads(z.read("case.json"))
                assert row["host_index"]==case[0] and row["mic_sha256"]==ch.digest(case[4]) and z.testzip() is None
                for member in z.namelist():
                    if member != "case.json": logs.writestr(member,z.read(member))
            rows.append(row)
            opens = row["opened"]
            meta = row["source"]
            timely = bool(opens and meta["start"]<=opens[0]<meta["command_at"] and row["role"]!="panel")
            summary[row["name"]+"."+row["role"]].update(trials=1,opens=bool(opens),timely=timely,
                                                       small=row["small"],turbo=row["turbo"],
                                                       timely_turbo=timely and row["turbo"])
            print(idx,case[0],row["name"],row["role"],opens,flush=True)
    assert len(rows)==156 and all(summary[n+".both"]["trials"]==60 for n in ("abc","bca"))
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=summary,versions=versions,archive_sha256=ch.sha(archive),
                     policy="Both complete paired 60-condition overlaps, plus the panel-only and external-only "
                     "controls at simultaneous onset across all four geometries and three levels. External-only controls "
                     "are identical for both panels. Use P4 ESP-SR FIFO with the actual M and R, verify each admitted "
                     "notification and encode complete selected audio. Never-opened streams stay in command denominators."))


if __name__ == "__main__":
    main()
