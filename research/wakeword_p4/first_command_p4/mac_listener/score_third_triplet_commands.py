"""Read the complete sends for the surviving three-triplet cue at the original stereo level."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import check_third_triplet_collisions as chosen
import score_bella_headroom_commands as previous

triple, P, scoring = chosen.triple, chosen.P, previous.scoring
OUT = triple.OUT.with_name("4090-bella-third-triplet-commands.json")
ARCHIVE = triple.WORK/OUT.with_suffix(".zip").name
SENDER = triple.WORK / "sender"


def verify_full_history_sender(work, logs, files):
    sender = scoring.sender
    parent = P.read(sender.OUT)
    row, payload, packets = next(sender.load_cases())
    header = struct.unpack_from("<4I", payload)
    index, opened, first, _ = header
    assert index == row["index"] and first + 32000 == opened // 160 * 160
    original = sender.OUT.with_suffix(".zip")
    assert P.sha(original) == parent["archive_sha256"]
    with ZipFile(original) as saved:
        old_log = work / "full-history-original.log"
        old_log.write_bytes(saved.read(f"{index:02}.log"))
    assert P.sha(old_log) == parent["records"][index]["uart_sha256"]
    old = sender.compare(old_log, *header, packets)
    new_log, new = sender.run_case(work, payload, packets)
    assert old == new
    name = "control/full-history-sender.log"
    logs.write(new_log, name)
    files[name] = P.sha(new_log)
    return dict(opened=opened, first=first, original_uart_sha256=P.sha(old_log),
                matched_packets=[len(frames) for frames, _ in new])


def cases(work, measured):
    fit=triple.fit
    old,sr=sf.read(fit.early.WAV,dtype="int16")
    cue,rr=sf.read(triple.wave(chosen.NAME),dtype="int16")
    assert sr==rr==16000
    plan={r["label"]:r for r in P.read(fit.study.PLAN)["records"]}
    rows={r["label"]:r for r in measured["records"]}
    for label,group,mic,ref,_,_,_ in fit.scenes(old,cue):
        if group!="talk":
            continue
        row=rows[label]
        assert P.digest(mic)==row["mic_sha256"] and P.digest(ref)==row["ref_sha256"]
        path=work/(label+".wav")
        sf.write(path,mic,16000,subtype="PCM_16")
        yield dict(plan[label], pcm_sha256=row["mic_sha256"],path=str(path)),row
    more={r["label"]:r for r in P.read(triple.experiment.source.angles.PLAN)["records"]}
    for row in measured["records"]:
        if row["group"]=="new_angles":
            yield dict(more[row["label"]],pcm_sha256=row["mic_sha256"],path=row["path"]),row


def main():
    measured=P.read(chosen.REPLAY)
    parent=P.read(previous.OUT)
    images=(scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",scoring.sender.BUILD/"board.bin")
    expected=P.read(previous.baseline.earlier.OUT)["sources"]
    assert all(P.sha(p)==expected[str(p)] for p in images) and P.sha(P.QEMU)==measured["build"]["qemu_sha256"]
    proc,model,qid=scoring.prior.load()
    assert qid==parent["qwen"] and "4090" in qid["device"]
    decoder=scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==parent["decoder"]
    turbo=scoring.prior.reader(parent["turbo"])
    readers=proc,model,turbo,decoder
    dry,rate=sf.read(triple.wave(chosen.NAME),dtype="int16")
    assert rate==16000
    dry_text=dict(qwen=scoring.prior.evaluate(proc,model,dry,"Hey Nix")["text"],
                  whisper=scoring.prior.recognition(turbo,dry,"Hey Nix")["text"])
    assert (SENDER / "mic_wake.inc").read_bytes() == (scoring.sender.BUILD / "mic_wake.inc").read_bytes()
    assert (SENDER / "sender_constants.h").read_bytes() == (scoring.sender.BUILD / "sender_constants.h").read_bytes()
    scoring.sender.BUILD = SENDER
    cache=triple.WORK/("sends-"+P.sha(chosen.REPLAY)[:16]+"-"+P.sha(SENDER/"board.bin")[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK=cache
    rows,counts,files=[],defaultdict(Counter),{}
    with TemporaryDirectory(dir=triple.WORK) as temp:
        work=Path(temp)
        with ZipFile(work/ARCHIVE.name,"w",ZIP_DEFLATED) as logs:
            control = verify_full_history_sender(work, logs, files)
            for i,(r,obs) in enumerate(cases(work,measured)):
                opened=next(iter(obs["events"]["O"]),None)
                row=dict(label=r["label"],group=obs["group"],command=r["command"],opened=opened,parent_open=obs["before"],end=r["end"])
                c=counts[obs["group"]]
                c.update(cases=1,opened=int(opened is not None))
                if opened is not None:
                    u,shard=scoring.score_case(dict(r,reference_sha256=obs["ref_sha256"]),i,opened,readers,work,cache)
                    assert all(u[k]==0 for k in ("capture_watchdog","opus_watchdog","sender_watchdog"))
                    row["uplink"]=u
                    with ZipFile(shard) as z:
                        for name in z.namelist():
                            if name.endswith(".log"):
                                raw=z.read(name)
                                logs.writestr(name,raw)
                                files[name]=triple.fit.study.sha256(raw).hexdigest()
                    for schedule,v in u["readings"].items():
                        c[schedule+"_qwen"]+=int(v["qwen"]["complete"])
                        c[schedule+"_whisper"]+=int(v["turbo"]["complete_suffix"])
                rows.append(row)
                print(i,r["label"],opened,flush=True)
        primary = [name for name in files if not name.startswith(("control/", "retry/"))]
        assert len(rows)==24 and len(primary)==3*sum(r["opened"] is not None for r in rows)
        deps=(Path(__file__).resolve().relative_to(Path.cwd()),chosen.REPLAY,triple.PROTO,triple.OUT,previous.OUT,*images,
              SENDER/"board.bin",SENDER/"mic_wake.inc",scoring.sender.APP/"sender.c",scoring.sender.OUT)
        P.save(work/OUT.name,dict(records=rows,counts=counts,files=files,archive=ARCHIVE.name,archive_sha256=P.sha(work/ARCHIVE.name),
               sources={str(p):P.sha(p) for p in deps},full_history_control=control,isolated=dry_text,qwen=qid,turbo=parent["turbo"],decoder=parent["decoder"]))
        (work/ARCHIVE.name).replace(ARCHIVE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
