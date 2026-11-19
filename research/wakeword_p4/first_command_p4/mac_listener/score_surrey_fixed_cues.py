"""Recover native Willow returns and score complete P4-encoded commands from the measured Surrey room."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import score_eight_voice_channel_uplinks as uplinks
import score_eight_voice_willow as reader
from score_handoff_willow import willow_events
import surrey_fixed_cues as experiment

ch = reader.ch
OUT = experiment.PLAN.with_name("4090-c161-surrey-fixed-cues.json")


def main():
    plan = ch.read(experiment.PLAN)
    ident = ch.sha(experiment.PLAN)
    ref = ch.read(reader.OUT)
    small, sid = ch.load_reader()
    proc, model, qid = reader.gain.study.prefix.balanced.original.asr.load()
    assert (sid,qid,reader.uplink.firmware()) == (ref["small"],ref["qwen"],ref["codec"])
    readers = small,reader.gain.tx.diag.reader(ref["turbo"]),reader.uplink.decoder_api(),proc,model
    cache = experiment.WORK/"uplinks"/(ident+ch.sha(Path(__file__)))
    cache.mkdir(exist_ok=True,parents=True)
    archive = OUT.with_suffix(".zip")
    rows, summary = [], defaultdict(Counter)
    with TemporaryDirectory(dir=cache) as tmp:
        work = Path(tmp)
        reader.uplink.WORK = work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for i,src in enumerate(plan["records"]):
                hits = willow_events(i,src,ident,logs,experiment.WORK/"runs")
                w = reader.timing(src,(hits[0]+255)//256*256 if hits else None)
                n = reader.timing(src,next(iter(src["nyx"]["opened"]),None))
                role = src["target_id"]
                decision = w if role=="willow_matched" else n
                row = dict(index=i, role=role, mode=src["mode"], nyx=n, willow=w, decision=decision,
                           mic_sha256=src["pcm_sha256"], willow_samples=hits)
                stats = summary[src["mode"]+"."+role]
                stats.update(trials=1,opened=int(decision["opened"] is not None),timely=int(decision["timely"]))
                if src["mode"]!="dry" and role!="no_address" and decision["opened"] is not None:
                    path = Path(src["path"])
                    mic,rate = sf.read(path,dtype="int16")
                    assert rate==16000 and ch.sha(path)==src["sha256"] and ch.digest(mic)==src["pcm_sha256"]
                    cap = uplinks.capture(work,cache,logs,row,src,mic,readers)
                    assert cap["fifo"]["first"] == max(0,decision["opened"]//160*160-32000)
                    row["capture"] = cap
                    stats.update(qwen=int(cap["qwen"]["complete"]),turbo=int(cap["turbo"]["complete_suffix"]))
                rows.append(row)
            for path in sorted(work.glob("*-attempt*.gz")):
                logs.write(path,"codec-retries/"+path.name)
        assert len(rows)==68
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        ch.save(work/OUT.name,dict(records=rows,summary=summary,plan_sha256=ident,
            archive_sha256=ch.sha(work/archive.name),script_sha256=ch.sha(Path(__file__)),
            small=sid,turbo=ref["turbo"],qwen=qid,codec=ref["codec"],policy="Official DEFAULT Willow and INT8 C161 "
            "with marker guard on identical PCM. Decode complete two-channel P4 Opus for each own-target opening."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
