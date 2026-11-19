"""Recover the original and post120 routes' complete spoken commands from P4 Opus."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import replay_post120_overlap as replay
import score_eight_voice_willow as reader

ch = replay.ch
OUT = replay.OUT.with_name(replay.STEM + "-commands.json")


def main():
    run = ch.read(replay.OUT)
    ref = ch.read(reader.OUT)
    proc, model, qid = reader.gain.study.prefix.balanced.original.asr.load()
    small, sid = ch.load_reader()
    assert (qid,sid,reader.uplink.firmware()) == (ref["qwen"],ref["small"],ref["codec"])
    decoder = reader.uplink.decoder_api()
    readers = small,reader.gain.tx.diag.reader(ref["turbo"]),decoder,proc,model
    old = {}
    sources = {}
    for stem in ("4090-c161-connected-overlap-commands", "4090-c161-additional-connected-voices-unmarked"):
        path = ch.RESULTS / (stem + ".json")
        doc = ch.read(path)
        archive = path.with_suffix(".zip")
        assert ch.sha(archive)==doc["archive_sha256"] and doc["codec"]==ref["codec"] and doc["qwen"]==qid
        sources[path.name] = ch.sha(path)
        sources[archive.name] = doc["archive_sha256"]
        for index, entry in enumerate(doc["captures"]):
            old[entry["pcm_sha256"],entry["opened"]] = entry,archive,index
    captures, rows, stats = [], [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        reader.uplink.WORK = work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for (cohort,track,s,mic,*_), r in zip(replay.recordings(),(r for r in run["records"] if "cohort" in r),strict=True):
                assert (cohort,track,s,ch.digest(mic)) == (r["cohort"],r["track"],r["source"],r["pcm_sha256"])
                if track != "plain":
                    continue
                clocks = {}
                for name, hits in r["opened"].items():
                    g = stats[cohort+"."+name]
                    g.update(trials=1)
                    if not hits:
                        continue
                    at, = hits
                    if at not in clocks:
                        i = len(captures)
                        key = r["pcm_sha256"],at
                        if key in old:
                            entry,where,j = old[key]
                            with ZipFile(where) as previous:
                                uart = previous.read(f"opus/{j:03}-external.log").decode().replace("\r\n","\n")
                            fifo = entry["capture"]
                            assert fifo["first"]==max(0,at//160*160-32000)
                            for lane,label in enumerate(("selected","raw")):
                                pkts = reader.uplink.packets(uart,j,fifo["emitted"]-fifo["first"],lane)
                                pcm = reader.uplink.decode(decoder,pkts)
                                assert ch.digest(pcm)==fifo["lanes"][label]["decoded_sha256"]
                            cap = dict(fifo=fifo,qwen=entry["qwen"],turbo=entry["turbo"])
                            origin = dict(archive=where.name,case=j)
                        else:
                            cap = reader.recognize(work,logs,i,mic,dict(carrier=s["command"]),at,readers)
                            origin = dict(archive=archive.name,case=i)
                        assert cap["fifo"]["lanes"]["selected"]["decoded_sha256"]==cap["qwen"]["pcm_sha256"]
                        captures.append(dict(case=i,cohort=cohort,source=s,mic_sha256=key[0],opened=at,origin=origin,**cap))
                        clocks[at] = i
                    cap = captures[clocks[at]]
                    timely = s["start"]<=at<s["command_at"]
                    g.update(opened=1,timely=int(timely),qwen=int(cap["qwen"]["complete"]),turbo=int(cap["turbo"]["complete_suffix"]),
                             timely_qwen=int(timely and cap["qwen"]["complete"]),timely_turbo=int(timely and cap["turbo"]["complete_suffix"]))
                rows.append(dict(key=r["key"],captures={name:(clocks[h[0]] if h else None) for name,h in r["opened"].items()}))
            for p in sorted(work.glob("*-attempt*.gz")):
                logs.write(p,"codec-retries/"+p.name)
        assert len(rows)==72
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        ch.save(work/OUT.name,dict(records=rows,captures=captures,summary=stats,archive_sha256=ch.sha(work/archive.name),
                replay_sha256=ch.sha(replay.OUT),script_sha256=ch.sha(Path(__file__)),sources=sources,codec=ref["codec"],
                small=sid,turbo=ref["turbo"],qwen=qid,policy="Same two-second history and full selected 24 kb/s uplink. "
                "Decode verified prior captures at equal input and OPEN; encode uncovered clocks on simulated P4."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
