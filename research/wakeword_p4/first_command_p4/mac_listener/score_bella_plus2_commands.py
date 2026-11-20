"""Compare complete received commands for the isolated two-decibel cue over all interruptions."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_plus2_playback as trial
import score_address_gain_commands as sender

P, scoring = sender.P,sender.scoring
OUT = trial.gain.RESULTS / "4090-bella-plus2-commands.json"
TRACE = trial.gain.WORK / (OUT.stem+"-uart.zip")


def main():
    checks = P.read(trial.OUT)
    wake = {r["label"]:r for r in checks["records"]}
    old = P.read(sender.OUT)
    anchors = {r["scene"]:r for r in old["records"] if r["role"]=="nyx" and r["level"]=="plus2dB"}
    assert len(anchors)==3 and P.sha(sender.TRACE)==old["archive_sha256"]
    baseline = sender.baseline.daily.OUT
    histories = {(r["index"],r["role"]):r for r in P.read(baseline)["records"]}
    original,rate = sf.read(trial.daily.board.bella.CUE,dtype="int16")
    cue,cr = sf.read(trial.CUE,dtype="int16")
    assert rate==cr==16000 and P.sha(trial.CUE)==checks["cue_sha256"] and P.digest(cue)==checks["pcm_sha256"]
    inputs = (Path(__file__),Path(scoring.__file__),trial.OUT,trial.CUE,sender.OUT,baseline,
              scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",scoring.sender.BUILD/"board.bin")
    sources = {p.name:P.sha(p) for p in inputs}
    for key in ("nyx_mac_listener.bin","nyx_first_command.bin","board.bin"):
        assert sources[key]==old["sources"][key]
    assert P.sha(P.QEMU)==checks["qemu_sha256"]
    proc,model,qid = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==old["decoder"] and qid==old["qwen"] and "4090" in qid["device"]
    readers = (proc,model,scoring.prior.reader(old["turbo"]),decoder)
    cache = trial.gain.WORK/("plus2-sender-"+P.sha(trial.OUT)[:20])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    summary, rows = defaultdict(lambda:defaultdict(Counter)), []
    with TemporaryDirectory(dir=trial.gain.WORK) as tmp, ZipFile(sender.TRACE) as previous_logs:
        temp = Path(tmp)
        with ZipFile(temp/TRACE.name,"w",ZIP_DEFLATED) as uart:
            for label,meta,mic,ref in trial.external(original,cue):
                if meta["role"]!="plus2":
                    continue
                native = wake[label]
                i = meta["scene"]
                assert native["mic_sha256"]==P.digest(mic) and native["reference_sha256"]==P.digest(ref)
                opened = next(iter(native["events"]["O"]),None)
                row = dict(index=i,label=label,level=meta["db"],file=meta["file"],command=meta["command"],
                           command_at=meta["command_at"],mic_sha256=native["mic_sha256"],reference_sha256=native["reference_sha256"],
                           opened=opened,selected=meta["selection"],readings={},uart={})
                if opened is not None:
                    if i in anchors:
                        old_row = anchors[i]
                        assert old_row["opened"]==opened and old_row["pcm_sha256"]==P.digest(mic)
                        assert old_row["reference_sha256"]==P.digest(ref)
                        reading = old_row
                        for source,digest in old_row["uart"].items():
                            name = label+"/"+Path(source).name
                            raw = previous_logs.read(source)
                            assert sha256(raw).hexdigest()==digest
                            uart.writestr(name,raw)
                            row["uart"][name]=digest
                    else:
                        mic_path, ref_path = temp/"mic.wav",temp/"ref.wav"
                        sf.write(mic_path,mic,16000,subtype="PCM_16")
                        sf.write(ref_path,ref,16000,subtype="PCM_16")
                        source = dict(label=label,path=mic_path,reference_path=ref_path,pcm_sha256=P.digest(mic),
                                      reference_sha256=P.digest(ref),command=meta["command"])
                        reading,shard = scoring.score_case(source,i,opened,readers,temp,cache)
                        with ZipFile(shard) as logs:
                            for stage in ("capture","opus","sender"):
                                name = f"{label}/{stage}.log"
                                raw = logs.read(name)
                                digest = sha256(raw).hexdigest()
                                assert digest==reading[stage+"_uart_sha256"] and reading[stage+"_watchdog"]==0
                                uart.writestr(name,raw)
                                row["uart"][name]=digest
                    row.update(first=reading["first"],stop=reading["stop"],samples=reading["samples"],
                               fifo_sha256=reading["fifo_sha256"],readings=reading["readings"],reused=i in anchors)
                for name,reference in (("plus2",row),("original",histories[i,"original"]),("peak",histories[i,"enhanced"])):
                    if name!="plus2":
                        assert (reference["file"],reference["command"],reference["command_at"])==(meta["file"],meta["command"],meta["command_at"])
                    for group in ("all",str(meta["db"]),"selected" if meta["selection"] else "other"):
                        c = summary[group][name]
                        c.update(cases=1,opened=int(reference["opened"] is not None))
                        for condition in ("immediate","waited"):
                            for reader,field in (("qwen","complete"),("turbo","complete_suffix")):
                                observed = reference["opened"] is not None
                                if observed:
                                    observed = (reference["readings"] if name=="plus2" else reference["sender"])[condition][reader][field]
                                c[condition+"_"+reader]+=int(observed)
                rows.append(row)
                if len(rows)%8==0: print(len(rows),flush=True)
        assert len(rows)==len({r["index"] for r in rows})==96
        P.save(temp/OUT.name,dict(records=rows,summary=summary,sources=sources,qwen=qid,turbo=old["turbo"],decoder=old["decoder"],
               archive=TRACE.name,archive_sha256=P.sha(temp/TRACE.name),scope="All 96 historical interruptions. "
               "Three gain-selection cases reuse the room comparison. Rounded native openings and complete "
               "P4 simulated uplinks for all other opens. Identical suffixes and electrical references."))
        (temp/TRACE.name).replace(TRACE)
        (temp/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
