"""Read ordinary Nyx and matched Willow commands after their own measured first-open times."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import score_bella_weak_speaker_gain as scoring
import replay_conversation_part_cutoffs as study

fit, P = study.fit, scoring.P
OUT = fit.RESULTS / (fit.STEM + "-first-commands.json")
TRACE = fit.WORK / (OUT.stem + "-uart.zip")
SPEECH = fit.RESULTS / (fit.STEM + "-conversation-speech.json")


def main():
    plan = study.replay.CONNECTED.with_name(study.replay.CONNECTED.stem+"-plan.json")
    readers_meta = P.read(scoring.OUT)
    sources = {p.name:P.sha(p) for p in (Path(__file__),SPEECH,plan,study.replay.CONNECTED,scoring.OUT,Path(scoring.__file__))}
    for p in (scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",scoring.sender.BUILD/"board.bin"):
        sources[p.name] = P.sha(p)
        assert sources[p.name] == readers_meta["sources"][p.name]
    assert P.sha(P.QEMU) == readers_meta["p4_qemu_sha256"]
    proc,model,qid = scoring.prior.load()
    assert qid==readers_meta["qwen"] and "4090" in qid["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==readers_meta["decoder"]
    readers = (proc,model,scoring.prior.reader(readers_meta["turbo"]),decoder)
    cache = fit.WORK/("sender-"+sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    planned = {str(r["index"]):r for r in P.read(plan)["records"]}
    rows,counts = [],defaultdict(Counter)
    with TemporaryDirectory(dir=fit.WORK) as folder:
        work = Path(folder)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as uart:
            for r in P.read(SPEECH)["records"]:
                if r["group"]!="connected" or r["role"] not in ("Hey Nyx","Hey Willow"):
                    continue
                source = planned[r["key"]]
                role = r["role"]
                mic,sr = sf.read(source["path"],dtype="int16")
                assert sr==16000 and source["pcm_sha256"]==r["pcm_sha256"]==P.digest(mic)
                ref = np.zeros_like(mic)
                ref_file = work/"reference.wav"
                sf.write(ref_file,ref,16000,subtype="PCM_16")
                routes = {"willow":r["willow"]} if role=="Hey Willow" else dict(c161=r["c161"],**r["union"])
                clocks = {k:(ts[0]+255)//256*256 if ts else None for k,ts in routes.items()}
                received = {}
                for opened in sorted({v for v in clocks.values() if v is not None}):
                    label = r["key"]+"/"+str(opened)
                    record = dict(path=source["path"],pcm_sha256=r["pcm_sha256"],reference_path=ref_file,
                                  reference_sha256=P.digest(ref),command=source["carrier"],label=label)
                    captured,shard = scoring.score_case(record,int(r["key"]),opened,readers,work,cache)
                    with ZipFile(shard) as z:
                        for name in z.namelist():
                            if name!="record.json":
                                uart.writestr(name,z.read(name))
                    received[opened] = captured
                for arm,opened in clocks.items():
                    c = counts[arm]
                    c.update(cases=1,opened=int(opened is not None))
                    if opened is None:
                        continue
                    timely = r["start"]<=opened<r["command_at"]
                    c.update(timely=int(timely))
                    for condition,read in received[opened]["readings"].items():
                        for reader,field in (("qwen","complete"),("turbo","complete_suffix")):
                            ok = read[reader][field]
                            c[condition+"_"+reader] += int(ok)
                            c[condition+"_"+reader+"_timely"] += int(ok and timely)
                rows.append(dict(index=int(r["key"]),role=role,voice=r["voice"],command=source["carrier"],
                            pcm_sha256=r["pcm_sha256"],command_at=r["command_at"],opens=clocks,received=received))
                print(role,r["key"],clocks,flush=True)
        assert all(v["cases"]==12 for v in counts.values()) and len(rows)==24
        with ZipFile(work/TRACE.name) as z:
            assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        P.save(work/OUT.name,dict(records=rows,counts=counts,qwen=qid,turbo=readers_meta["turbo"],
               decoder=readers_meta["decoder"],qemu_sha256=P.sha(P.QEMU),sources=sources,archive=TRACE.name,
               archive_sha256=P.sha(work/TRACE.name),scope="One set of 12 literal Nyx and 12 matched Willow dry "
               "continuous utterances. Use first host integer part / baseline C161 or native official Willow clocks. "
               "One P4 capture, Opus and sender for each distinct open per waveform. Zero playback reference. "
               "Read complete immediate and 500/1500 ms delayed simulated sends on RTX 4090. Missed opens "
               "remain in all denominators. Fixed synthetic cue is outside these natural-address samples."))
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
