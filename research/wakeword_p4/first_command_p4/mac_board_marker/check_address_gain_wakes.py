"""Collect native fixed-sound and official Willow decisions for the registered address levels."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_address_gain as gain
import check_fixed_sound_runtime as fixed
import willow_room_reference as room

OUT = gain.RESULTS / (gain.STEM+"-wake.json")
TRACE = OUT.with_suffix(".zip")
OLD = Path(".local/wakeword/c161-fixed-sound-runtime/replay")
WILLOW_BOARD = Path(".local/wakeword/c161-willow-board/src")
digest = gain.digest


def fixed_image(prior, which):
    app = (OLD/which/"board.bin").read_bytes()
    assert digest(app)==prior["guest_sha256"][which] and digest(fixed.QEMU.read_bytes())==prior["qemu_sha256"]
    flash = bytearray(b"\xff"*(4<<20))
    config = room.read(fixed.SIM/"flasher_args.json")
    for key,member in config["flash_files"].items():
        offset = int(key,16)
        data = app if member==config["app"]["file"] else (fixed.SIM/member).read_bytes()
        assert offset+len(data)<=2<<20
        flash[offset:offset+len(data)]=data
    return flash


def p4_cases():
    meta = room.read(fixed.REPORT)
    plan = room.read(gain.PLAN)
    known = {r["label"]:r for r in meta["records"]}
    with TemporaryDirectory(dir=gain.WORK) as tmp:
        stage = Path(tmp)
        rows = []
        with ZipFile(stage/"p4.zip","w",ZIP_DEFLATED) as trace:
            for kind in ("fixed","dual"):
                flash = fixed_image(meta,kind)
                cache = gain.WORK/kind/digest(bytes.fromhex(meta["qemu_sha256"])+flash[:2<<20])
                cache.mkdir(parents=True,exist_ok=True)
                for r in plan["records"]:
                    mic,sr = sf.read(r["path"],dtype="int16")
                    ref,rr = sf.read(r["reference_path"],dtype="int16")
                    assert sr==rr==16000 and digest(mic.tobytes())==r["pcm_sha256"] and digest(ref.tobytes())==r["reference_sha256"]
                    block = fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                    label = r["label"]
                    old = known.get(f"{r['scene']:03}/original/{r['role']}") if r["gain_db"]==0 else None
                    if old and kind in old:
                        assert old["serialized_input_sha256"]==digest(block)
                        with ZipFile(fixed.REPORT.with_suffix(".zip")) as prior:
                            raw = prior.read(f"{kind}/{r['scene']:03}/original/{r['role']}.log")
                        assert digest(raw)==old[kind]["uart_sha256"]
                        events = old[kind]["epochs"][0]["events"]
                        faults = {}
                    else:
                        for attempt in range(2):
                            try:
                                result,log,failed = fixed.boot(cache,flash,cache,block,1)
                                break
                            except AssertionError:
                                if attempt:
                                    raise
                        raw = log.read_bytes()
                        events = result[0]["events"]
                        faults = {}
                        for log in failed:
                            member = f"failed/{kind}/{label}/{log.name}"
                            trace.write(log,member)
                            faults[member]=digest(log.read_bytes())
                    assert b"BOARD PASS\r\n" in raw
                    trace.writestr(f"{kind}/{label}.log",raw)
                    rows.append(dict(kind=kind,label=label,input_sha256=digest(block),events=events,
                                     uart_sha256=digest(raw),reused=bool(old and kind in old),failed=faults))
                print(kind,len(rows),flush=True)
        room.save(stage/"p4.json",dict(records=rows,guest_sha256=meta["guest_sha256"],qemu_sha256=meta["qemu_sha256"],
                   sources={p.name:digest(p.read_bytes()) for p in (gain.PLAN,fixed.REPORT,fixed.REPORT.with_suffix(".zip"),
                            Path(fixed.__file__),Path(__file__))},archive_sha256=digest((stage/"p4.zip").read_bytes())))
        for name in ("p4.json","p4.zip"):
            (stage/name).replace(gain.WORK/name)


def willow_cases(plan_path=gain.PLAN, work_root=gain.WORK, runner=gain.SCRIPT):
    plan = room.read(plan_path)
    for path, sha in plan["files"].items():
        assert digest(Path(path).read_bytes())==sha,path
    old = room.read(room.OUT)
    assert digest(room.TRACE.read_bytes())==old["archive_sha256"]
    previous = {r["mic_sha256"]:r for r in old["records"]}
    sha_plan = digest(plan_path.read_bytes())
    with ZipFile(room.TRACE) as history, TemporaryDirectory(dir=work_root) as tmp:
        stage = Path(tmp)
        feeder = room.build_feeder(WILLOW_BOARD,stage)
        rows = []
        with ZipFile(stage/"willow.zip","w",ZIP_DEFLATED) as trace:
            for n,r in enumerate(plan["records"]):
                key = r["pcm_sha256"]
                old_row = previous.get(key)
                names = ("report.json","serial.log","raw-returns.bin")
                if old_row:
                    data = {name:history.read(old_row["label"]+"/"+name) for name in names}
                else:
                    run = work_root/"willow"/sha_plan/str(n)
                    if not run.exists():
                        with TemporaryDirectory(prefix="address-gain-") as tmp_run:
                            output = Path(tmp_run)/"run"
                            cmd = [sys.executable,runner,"case","--record-index",str(n),"--model","willow",
                                   "--threshold","DEFAULT","--output",output,"--timeout-seconds","70"]
                            for name,path in plan["runtime"].items():
                                cmd.extend(["--"+name,path])
                            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
                            (output/"input.pcm").unlink()
                            run.parent.mkdir(parents=True,exist_ok=True)
                            shutil.move(str(output),run)
                    data = {name:(run/name).read_bytes() for name in names}
                rpt = json.loads(data["report.json"])
                assert rpt["source_pcm_sha256"]==key and rpt["model"]=="wn9_heywillow_tts" and rpt["mode"]=="DEFAULT"
                assert rpt["stage"]==104 and rpt["total_prime_samples"]==81600 and not rpt["additional_priming_events"]
                assert rpt["actual_float32_bits"]==rpt["post_prime_getter_bits"]=="3f24dd2f"
                assert all(v<=0 for _,v in rpt["prime_returns"])
                if not old_row:
                    assert rpt["execution_sha256"]==sha_plan and rpt["record"]==r
                for tag in ("qemu","elf","flash"):
                    assert rpt[tag+"_sha256"]==plan["files"][plan["runtime"][tag]]
                assert digest(data["serial.log"])==rpt["serial_sha256"] and digest(data["raw-returns.bin"])==rpt["raw_returns_sha256"]
                mic,sr = sf.read(r["path"],dtype="int16")
                assert sr==16000 and digest(mic.tobytes())==key
                pcm = mic.tobytes()+bytes(2*rpt["final_padding_samples"])
                sequence = bytes(v+1 for _,v in rpt["prime_returns"])+data["raw-returns.bin"]
                lines = subprocess.check_output([feeder,str(len(pcm)//2),"256"],input=pcm+sequence)
                events = [tuple(map(int,line.split())) for line in lines.splitlines()]
                assert [e[0] for e in events]==[e["relative_sample"] for e in rpt["real_events"]]
                for name,raw in data.items():
                    trace.writestr(r["label"]+"/"+name,raw)
                rows.append(dict(label=r["label"],pcm_sha256=key,events=events,reused=bool(old_row),
                                 members={name:digest(raw) for name,raw in data.items()}))
                print(n,r["label"],events,flush=True)
        room.save(stage/"willow.json",dict(records=rows,archive_sha256=digest((stage/"willow.zip").read_bytes()),
                  plan_sha256=sha_plan,prior_trace_sha256=old["archive_sha256"]))
        for name in ("willow.json","willow.zip"):
            (stage/name).replace(work_root/name)


def combine():
    rows = room.read(gain.PLAN)["records"]
    p4 = room.read(gain.WORK/"p4.json")
    willow = room.read(gain.WORK/"willow.json")
    native = {(r["kind"],r["label"]):r for r in p4["records"]}
    refs = {r["label"]:r for r in willow["records"]}
    assert len(rows)==len(refs)==45 and len(native)==90
    for r in rows:
        r["nyx"]=native["fixed",r["label"]]
        r["c161_marker"]=native["dual",r["label"]]
        r["willow"]=refs[r["label"]]
    with TemporaryDirectory(dir=gain.WORK) as tmp:
        stage = Path(tmp)
        with ZipFile(stage/TRACE.name,"w",ZIP_DEFLATED) as trace:
            for name,meta in (("p4",p4),("willow",willow)):
                path = gain.WORK/(name+".zip")
                assert digest(path.read_bytes())==meta["archive_sha256"]
                with ZipFile(path) as z:
                    for member in z.namelist():
                        trace.writestr(name+"/"+member,z.read(member))
        room.save(stage/OUT.name,dict(records=rows,archive_sha256=digest((stage/TRACE.name).read_bytes()),
                  sources={p.name:digest(p.read_bytes()) for p in (Path(__file__),gain.PLAN,fixed.REPORT,room.OUT)},
                  guest_sha256=p4["guest_sha256"],qemu_sha256=p4["qemu_sha256"],scope="Three consumed interruption "
                  "scenes with native P4 fixed-sound and C161+marker events and official Willow classic ESP32 decisions."))
        (stage/TRACE.name).replace(TRACE)
        (stage/OUT.name).replace(OUT)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("p4","willow","combine"))
    {"p4":p4_cases,"willow":willow_cases,"combine":combine}[parser.parse_args().stage]()
