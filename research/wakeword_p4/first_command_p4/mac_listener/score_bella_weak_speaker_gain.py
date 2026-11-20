"""Read full board-sent commands at measured request clocks in the weak speaker gain pairs."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf

import bella_weak_speaker_gain as trial
import check_bella_daily_sender as daily
from score_handoff_willow import willow_events

P, prior, sender, clock = daily.P, daily.prior, daily.sender, daily.daily.clock
OUT = trial.RESULTS / (trial.STEM + ".json")
TRACE = trial.WORK / (trial.STEM + "-uart.zip")


def score_case(record, number, opened, readers, work, cache):
    proc, model, turbo, decoder = readers
    mic, rate = sf.read(record["path"],dtype="int16")
    ref, ref_rate = sf.read(record["reference_path"],dtype="int16")
    assert rate == ref_rate == 16000 and P.digest(mic) == record["pcm_sha256"] and P.digest(ref) == record["reference_sha256"]
    name = record["label"]
    shard = cache / (str(number) + "-" + str(opened) + ".zip")
    if not shard.exists():
        tmp = shard.with_suffix(".tmp")
        with ZipFile(tmp,"w",ZIP_DEFLATED) as logs:
            pcm, meta = clock.capture(work,logs,number,name,mic,ref,opened)
            packets, immediate, codec = clock.opus(work,logs,number,name,pcm,decoder)
            payload = struct.pack("<4I",number,opened,meta["first"],len(pcm)) + pcm.tobytes()
            log,(_, (sent, waited)) = sender.run_case(work,payload,packets)
            logs.write(log,name+"/sender.log")
            assert waited["captured"]==len(packets) and waited["sent_silence"]==50
            readings = {}
            for label, audio in (("immediate",immediate),("waited",prior.decode(decoder,sent))):
                readings[label] = dict(qwen=prior.evaluate(proc,model,audio,record["command"]),
                                       turbo=prior.recognition(turbo,audio,record["command"]))
                assert readings[label]["qwen"]["pcm_sha256"] == P.digest(audio)
            item = dict(**meta,**codec,sender_uart_sha256=P.sha(log),sender_watchdog=log.read_text().count("Task watchdog got triggered"),
                        sender=waited,readings=readings,pcm_sha256=record["pcm_sha256"],reference_sha256=record["reference_sha256"])
            logs.writestr("record.json",json.dumps(item))
        tmp.replace(shard)
    with ZipFile(shard) as saved:
        item = json.loads(saved.read("record.json"))
        assert item["opened"]==opened and item["pcm_sha256"]==record["pcm_sha256"] and item["reference_sha256"]==record["reference_sha256"]
        for stage,key in (("capture","capture_uart_sha256"),("opus","opus_uart_sha256"),("sender","sender_uart_sha256")):
            assert sha256(saved.read(name+"/"+stage+".log")).hexdigest()==item[key]
    return item,shard


def main():
    plan, board, parent = (P.read(p) for p in (trial.PLAN,trial.WAKE,daily.OUT))
    previous = P.read(daily.daily.OUT)
    files = (Path(__file__),trial.PLAN,trial.WAKE,daily.OUT,daily.daily.OUT,trial.PACK,
             sender.BUILD/"board.bin",clock.BUILD/"nyx_mac_listener.bin",prior.CODEC/"nyx_first_command.bin")
    sources = {p.name:P.sha(p) for p in files}
    assert sources[trial.PLAN.name]==board["plan_sha256"] and sources[trial.PACK.name]==plan["pack_sha256"]
    assert sources["board.bin"]==parent["sources"]["board.bin"] and board["qemu_sha256"]==P.sha(P.QEMU)==parent["qemu_sha256"]
    assert sources["nyx_mac_listener.bin"]==previous["capture_firmware_sha256"] and sources["nyx_first_command.bin"]==previous["codec"]["opus_sha256"]
    proc,model,qid = prior.load()
    assert qid==parent["qwen"] and qid["device"].endswith("RTX 4090")
    turbo,decoder = prior.reader(parent["turbo"]),prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==parent["decoder"]
    cache = trial.WORK/("first-"+sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16])
    cache.mkdir(exist_ok=True)
    clock.reference.WORK = cache
    by_label = {r["label"]:r for r in board["records"]}
    previous_rows = {(r["index"],r["role"]):r for r in parent["records"]}
    rows, totals = [],defaultdict(Counter)
    with TemporaryDirectory(dir=trial.WORK) as folder:
        work = Path(folder)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as logs:
            for n,r in enumerate(plan["records"]):
                role,i,name = r["target_id"],r["index"],r["label"]
                # The board never redirects the Willow target to a Nyx call.
                # Score reciprocal official events as diagnostics on the original physical microphone.
                willow = willow_events(n,r,sources[trial.PLAN.name],logs,trial.WORK/"runs")
                w_open = (willow[0]+255)//256*256 if willow else None
                nyx = by_label[name]["events"] if name in by_label else None
                opened = w_open if role=="willow" else next(iter(nyx["O"]),None)
                row = dict(index=i,role=role,mode=r["mode"],file=r["file"],label=name,clipped=r["clipped"],
                           command=r["command"],mic_sha256=r["pcm_sha256"],ref_sha256=r["reference_sha256"],
                           start=r["start"],command_at=r["end"],opened=opened,willow_events=willow,nyx_events=nyx,
                           timely=opened is not None and r["start"]<=opened<r["end"])
                if nyx is not None:
                    cue = nyx["C"]
                    row["cue_only_open"] = (cue[0]+255)//256*256 if cue else None
                if opened is not None:
                    row["uplink"],shard = score_case(r,n,opened,(proc,model,turbo,decoder),work,cache)
                    with ZipFile(shard) as collected:
                        for member in collected.namelist():
                            if member!="record.json":
                                logs.writestr(member,collected.read(member))
                for subset in ("all", "three_misses" if i in (77,79,90) else "other_weak"):
                    acc = totals[".".join((r["mode"],role,subset))]
                    acc.update(cases=1,opened=int(opened is not None),timely=int(row["timely"]),
                               saturated_samples=r["clipped"],willow_open=int(w_open is not None))
                    if role not in ("nyx","willow"):
                        continue
                    old = previous_rows[i,"original" if role=="nyx" else "willow"]
                    for condition in ("immediate","waited"):
                        for reader,key in (("qwen","complete"),("turbo","complete_suffix")):
                            before = old["opened"] is not None and old["sender"][condition][reader][key]
                            after = opened is not None and row["uplink"]["readings"][condition][reader][key]
                            k = condition+"_"+reader
                            acc.update({k:int(after),k+"_original":int(before),k+"_gained":int(after and not before),
                                        k+"_lost":int(before and not after)})
                rows.append(row)
                if (n+1)%24==0:
                    print("Scored",n+1,flush=True)
        assert len(rows)==len({r["label"] for r in rows})==192
        with ZipFile(work/TRACE.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        P.save(work/OUT.name,dict(records=rows,summary=totals,sources=sources,qwen=qid,turbo=parent["turbo"],
                    c161=board["model"],board_guest_sha256=board["guest_sha256"],p4_qemu_sha256=board["qemu_sha256"],
                    willow="wn9_heywillow_tts DEFAULT",decoder=parent["decoder"],
                    archive=TRACE.name,archive_sha256=P.sha(work/TRACE.name),
                    policy="32 consumed weak-level interruptions; 8 distinct room/command playback files, 4 reuses each. "
                           "Same 81600-sample prime. Board C161 plus fixed marker on Nyx and erased controls. "
                           "Official Willow on all microphones including reciprocal inputs. Each target uses its "
                           "own first open and complete board FIFO. GPU ASR on immediate and delayed decoded sends. "
                           "All 32 enter each target denominator."))
        (work/OUT.name).replace(OUT)
        (work/TRACE.name).replace(TRACE)


if __name__=="__main__":
    main()
