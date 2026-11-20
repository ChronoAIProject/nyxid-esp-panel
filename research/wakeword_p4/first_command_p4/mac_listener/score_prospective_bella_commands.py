"""Score the complete simulated uplink after each detector's first opening in twelve synthetic rooms."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import prospective_bella_rooms as study
import score_bella_weak_speaker_gain as scoring

P = scoring.P
OUT = study.OUT.with_name(study.STEM+"-commands.json")
TRACE = study.WORK/(OUT.stem+"-uart.zip")
PREVIOUS = study.PLAN.with_name("4090-bella-plus2-commands.json")


def clocks(row):
    if row["role"] == "nyx":
        return row["nyx"]["events"]["O"]
    return [read_end for _, read_end in row["willow"]["events"]]


def main():
    result = P.read(study.OUT)
    previous = P.read(PREVIOUS)
    plan = P.read(study.PLAN)
    assert P.sha(study.PLAN) == result["plan_sha256"] and P.sha(P.QEMU) == result["qemu_sha256"]
    sources = {p.name:P.sha(p) for p in (study.OUT,study.PLAN,PREVIOUS,Path(__file__),Path(scoring.__file__),
               scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin", scoring.sender.BUILD/"board.bin")}
    for name in ("nyx_mac_listener.bin","nyx_first_command.bin","board.bin"):
        assert sources[name] == previous["sources"][name]
    proc,model,qwen = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert qwen == previous["qwen"] and "4090" in qwen["device"]
    assert decoder.opus_get_version_string().decode() == previous["decoder"]
    readers = proc,model,scoring.prior.reader(previous["turbo"]),decoder
    cache = study.WORK/("sender-"+sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:20])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows,stats=[],defaultdict(Counter)
    with TemporaryDirectory(dir=study.WORK) as folder:
        temp=Path(folder)
        with ZipFile(temp/TRACE.name,"w",ZIP_DEFLATED) as uart:
            for i,r in enumerate(result["records"]):
                if r["role"]=="erased":
                    continue
                witness=plan["records"][i]
                assert all(r[k]==v for k,v in witness.items()) and P.sha(Path(r["path"]))==r["sha256"]
                ref=np.zeros(r["samples"],dtype="<i2")
                reference=temp/"silent-reference.wav"
                sf.write(reference,ref,16000,subtype="PCM_16")
                opened=next(iter(clocks(r)),None)
                row={k:r[k] for k in ("room","role","mode","level","label","command","start","end","pcm_sha256")}
                row.update(index=i, opened=opened, uart={}, readings={})
                c=stats[f"{r['mode']}/{r['role']}/{r['level']}"]
                c.update(cases=1,opened=int(opened is not None))
                if opened is not None:
                    record=dict(r,reference_path=reference,reference_sha256=P.digest(ref))
                    reading,shard=scoring.score_case(record,i,opened,readers,temp,cache)
                    row.update(first=reading["first"],stop=reading["stop"],samples=reading["samples"],
                               fifo_sha256=reading["fifo_sha256"],readings=reading["readings"])
                    with ZipFile(shard) as logs:
                        for stage in ("capture","opus","sender"):
                            member=f"{r['label']}/{stage}.log"
                            data=logs.read(member)
                            digest=sha256(data).hexdigest()
                            assert digest==reading[stage+"_uart_sha256"] and reading[stage+"_watchdog"]==0
                            uart.writestr(member,data)
                            row["uart"][member]=digest
                    for condition,asr in row["readings"].items():
                        c[condition+"_qwen"]+=int(asr["qwen"]["complete"])
                        c[condition+"_turbo"]+=int(asr["turbo"]["complete_suffix"])
                rows.append(row)
                if len(rows)%8==0:
                    print(len(rows),flush=True)
        assert len(rows)==96 and len(stats)==8 and all(c["cases"]==12 for c in stats.values())
        P.save(temp/OUT.name,dict(records=rows,counts=stats,archive=TRACE.name,archive_sha256=P.sha(temp/TRACE.name),
               sources=sources,qwen=qwen,turbo=previous["turbo"],decoder=previous["decoder"],
               scope="Each Nyx or matched Willow address at its own native first panel read. Panel playback "
                     "silent. Complete P4 capture, Opus, sender and two readiness schedules. All 12 rooms in each denominator."))
        (temp/TRACE.name).replace(TRACE)
        (temp/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
