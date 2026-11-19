"""Compare verified official returns, frozen Nyx route events and real Opus for the rear command grid."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import json

import numpy as np
import soundfile as sf

import check_connected_marker_gains as gain
import eight_voice_willow as setup
from score_handoff_willow import willow_events
from score_paired_piper_willow import capture, uplink

ch = gain.ch
OUT = setup.PLAN.with_name(setup.STEM+".json")


def timing(s, opened):
    return dict(opened=opened, timely=opened is not None and s["start"]<=opened<s["end"],
                within_500ms=opened is not None and s["start"]<=opened<s["end"]+8000,
                relative_to_end_ms=(opened-s["end"])/16 if opened is not None else None)


def recognize(work, logs, i, mic, s, opened, readers):
    small, turbo, decoder, processor, model = readers
    count = capture(small,decoder,work,logs,i,"wake",mic,opened,s["carrier"])
    raw = logs.read(f"opus/{i:03}-wake.log").decode().replace("\r\n","\n")
    n = count["emitted"]-count["first"]
    streams = {}
    for lane,name in enumerate(("selected","raw")):
        packets = uplink.packets(raw,i,n,lane)
        pcm = uplink.decode(decoder,packets)
        assert len(packets)==count["lanes"][name]["packets"] and ch.digest(pcm)==count["lanes"][name]["decoded_sha256"]
        streams[name] = pcm
    pcm = streams["selected"]
    qwen = gain.study.prefix.balanced.evaluate(processor,model,pcm,s["carrier"])
    assert qwen["pcm_sha256"]==count["lanes"]["selected"]["decoded_sha256"]
    return dict(fifo=count,qwen=qwen,turbo=uplink.recognition(turbo,pcm,s["carrier"]))


def main():
    plan, plan_sha = ch.read(setup.PLAN), ch.sha(setup.PLAN)
    baseline = ch.read(gain.GUEST)
    nyx = {(r["batch"],r["source"]["index"]):r for r in baseline["records"] if r.get("paired_back_grid")}
    assert len(nyx)==24 and baseline["codec"]==uplink.firmware()
    old_archive = gain.GUEST.with_suffix(".zip")
    assert ch.sha(old_archive)==baseline["archive_sha256"]
    small, sid = ch.load_reader()
    proc, model, qid = gain.study.prefix.balanced.original.asr.load()
    assert sid==baseline["small"] and qid==baseline["qwen"]
    decoder = uplink.decoder_api()
    readers = small,gain.tx.diag.reader(baseline["turbo"]),decoder,proc,model
    counts, rows = defaultdict(Counter), []
    archive = OUT.with_suffix(".zip")
    cache = setup.WORK/"scored"/(plan_sha[:16]+"-"+ch.sha(Path(__file__))[:16])
    cache.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory(dir=cache) as folder:
        work = Path(folder)
        uplink.WORK = work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs, ZipFile(old_archive) as previous:
            for i,s in enumerate(plan["records"]):
                wav = Path(s["path"])
                mic, sr = sf.read(wav,dtype="int16")
                assert sr==16000 and ch.sha(wav)==s["sha256"] and ch.digest(mic)==s["pcm_sha256"]
                hits = willow_events(i,s,plan_sha,logs,setup.WORK/"runs")
                w = timing(s,(hits[0]+255)//256*256 if hits else None)
                n = timing(s,next(iter(s["nyx"]["opened"]),None))
                role = s["target_id"]
                row = dict(index=i,batch=s["batch"],track=s["track"],mode=s["mode"],role=role,
                           end=s["end"],voice=s["voice"],external_db=s["external_db"],mic_sha256=s["pcm_sha256"],
                           willow_raw_events=hits,willow=w,nyx=n,allowed_model=s["nyx"]["allowed_model"],
                           allowed_fixed=s["nyx"]["allowed_fixed"])
                if s["mode"]=="rear" and role=="nyx_marked":
                    old = nyx.pop((s["batch"],s["track"]))
                    assert old["mic_sha256"]==s["pcm_sha256"] and old["opened"]==s["nyx"]["opened"]
                    cap = old["uplink"]
                    uart = previous.read(f'{old["index"]:02}-opus.log').decode().replace("\r\n","\n")
                    packets = uplink.packets(uart,old["index"],cap["samples"],0)
                    assert len(packets)==cap["packets"]
                    assert ch.digest(uplink.decode(decoder,packets))==cap["pcm_sha256"]==old["qwen"]["pcm_sha256"]
                    row["capture"] = dict(from_report=gain.GUEST.name,case=old["index"],qwen=old["qwen"],turbo=cap["turbo"])
                route = w if role.startswith("willow_") else n
                new = s["mode"]=="rear" and (role.startswith("willow_") or role=="nyx_plain") and route["opened"] is not None
                if new:
                    shard = cache/f"{i:03}-{route['opened']}.zip"
                    if not shard.exists():
                        with ZipFile(work/shard.name,"w",ZIP_DEFLATED) as partial:
                            reading = recognize(work,partial,i,mic,s,route["opened"],readers)
                            partial.writestr("case.json",json.dumps(reading))
                        (work/shard.name).replace(shard)
                    with ZipFile(shard) as part:
                        assert part.testzip() is None
                        row["capture"] = json.loads(part.read("case.json"))
                        for member in part.namelist():
                            if member!="case.json":
                                logs.writestr(member,part.read(member))
                for system, decision in (("nyx",n),("willow",w)):
                    group = counts[".".join((s["mode"],role,system))]
                    group.update(trials=1,opened=int(decision["opened"] is not None),timely=int(decision["timely"]),
                                 within_500ms=int(decision["within_500ms"]))
                    if s["mode"]=="rear" and ((system=="willow" and role.startswith("willow_")) or
                                               (system=="nyx" and role.startswith("nyx_"))):
                        cap = row.get("capture")
                        for reader,field in (("qwen","complete"),("turbo","complete_suffix")):
                            complete = bool(cap and cap[reader][field])
                            group.update({reader:int(complete),"timely_"+reader:int(complete and decision["timely"])})
                rows.append(row)
        assert not nyx and len(rows)==160
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        ch.save(work/OUT.name,dict(records=rows,summary=counts,plan_sha256=plan_sha,pack_sha256=ch.sha(setup.PACK),
                nyx_report_sha256=ch.sha(gain.GUEST),nyx_archive_sha256=ch.sha(old_archive),nyx_model=baseline["model"],
                codec=baseline["codec"],qwen=qid,turbo=baseline["turbo"],small=sid,script_sha256=ch.sha(Path(__file__)),
                willow="wn9_heywillow_tts",willow_mode="DEFAULT",willow_threshold_float32_bits="3f24dd2f",
                archive_sha256=ch.sha(work/archive.name),policy="Official Xtensa QEMU raw 512-sample outputs after "
                    "81,600 priming samples, first delivery rounded to 256. Native Nyx runs both frozen routes. "
                    "For intended rear commands read the full P4 simulator Opus uplink at the corresponding OPEN, "
                    "including late openings; score the full exact normalized command suffix with two readers. "
                    "Marked Nyx reuses byte-verified captures of the actual P4 listener."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
