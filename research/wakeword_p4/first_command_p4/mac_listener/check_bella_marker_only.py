"""Apply the frozen whole-cue boost only to Bella's three code stems and replay the paired calls."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import check_bella_daily_overlap as overlap
import check_bella_daily_sender as daily

P, sender, prior, clock = daily.P, daily.sender, daily.prior, daily.daily.clock
gains = overlap.board.bella.gains
WORK = Path(".local/wakeword/c161-bella-marker-only")
OUT = P.RESULTS / "4090-c161-bella-marker-only.json"
GUEST = Path(".local/wakeword/c161-mac-board-marker/guest/board.bin")
CUE = P.RESULTS / "hye-nyx-af-bella-marker-only-v1.wav"
TRACE = WORK / "4090-c161-bella-marker-only-capture.zip"


def make_cue():
    base, rate = sf.read(overlap.board.bella.CUE,dtype="int16")
    full_gain = P.read(overlap.peak.OUT)
    assert rate==16000 and P.sha(overlap.board.bella.CUE)==full_gain["input_cue_sha256"]
    meta, v5, _, codes = gains.fit.grid.fit.original()
    with ZipFile(gains.PACK) as pack:
        spec = json.loads(pack.read("manifest.json"))
        item, = [r for r in spec["records"] if r.get("wake_only",{}).get("sha256")==P.sha(overlap.board.bella.CUE)]
    scale = item["prefix_scale"]
    boost = 10**(full_gain["gain_db"]/20)
    new = base.astype(float)
    intervals = []
    for pos, code, existing in zip(meta["code_positions"],codes,spec["gains"],strict=True):
        start = len(base)-len(v5)+pos
        assert 0<=start<=len(base)-len(code)
        new[start:start+len(code)] += (boost-1)*existing*scale*code
        intervals.append([start,start+len(code)])
    new = np.rint(new)
    assert np.max(np.abs(new))<=32767
    result = new.astype("<i2")
    return result, dict(original_sha256=P.sha(overlap.board.bella.CUE),code_boost_db=full_gain["gain_db"],
                        existing_marker_gains=spec["gains"],original_prefix_scale=scale,
                        marker_samples=intervals,changed_samples=int(np.count_nonzero(result!=base)),
                        pcm_peak=int(np.max(np.abs(new))),rms_change_db=float(20*np.log10(np.linalg.norm(new)/np.linalg.norm(base.astype(float)))))


def inputs(cue):
    baseline = P.read(overlap.OUT)
    by_id = {(r["index"],r["role"]):r for r in baseline["records"]}
    grid, taps = overlap.rooms()
    by_file = {v[0]["file"]:v for cases in grid.values() for v in cases}
    for _,key,_,speech in overlap.event_inputs("daily"):
        i = int(key)
        r = by_id[i,"original"]
        _, s, render, ref, active, tracks = by_file[r["file"]]
        end = s["command_at"]
        talk, levels = overlap.tx.balanced.match_level(active,speech)
        assert P.digest(speech)==r["dry_speech_sha256"] and levels["gain_db"]==r["speech_gain_db"]
        assert P.digest(ref)==r["reference_sha256"]
        new = tracks["original"].copy()
        new[end-len(cue):end] = cue
        for name, dry in [*tracks.items(),("marker_only",new)]:
            mixed = dry.astype(float)
            mixed[r["speech_start"]:r["speech_end"]] += talk
            mic, clips = render(overlap.board.bella.rear_filter(mixed,taps))
            if name!="marker_only":
                assert P.digest(mic)==by_id[i,name]["mic_sha256"]
            else:
                yield r, mic, ref, clips


def wake():
    expected = P.read(overlap.OUT)
    cue, info = make_cue()
    assert P.sha(GUEST)==expected["guest_sha256"] and P.sha(P.QEMU)==expected["qemu_sha256"]
    path = WORK/"wake.json"
    rows = []
    def cases():
        for old,mic,ref,clips in inputs(cue):
            i = old["index"]
            label = f"{i:03}"
            rows.append(dict(index=i,file=old["file"],label=label,text=old["text"],command=old["command"],
                             command_at=old["command_at"],cue_start=old["cue_start"],db=old["db"],
                             reference_sha256=P.digest(ref),mic_sha256=P.digest(mic),clipped=clips))
            yield label,mic.tobytes(),ref.tobytes(),None,old["command_at"]
            if len(rows)%24==0:
                print("P4 wake",len(rows),flush=True)
    result = overlap.board.run_guest.replay_cases(GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                  P.QEMU,WORK,cases())
    assert sorted(r["index"] for r in rows)==list(range(96))
    for r,t in zip(rows,result["records"],strict=True):
        assert r["label"]==t["label"]
        r.update(t)
    P.save(path,dict(records=rows,cue=info,pcm_sha256=P.digest(cue),guest_sha256=P.sha(GUEST),
                     qemu_sha256=P.sha(P.QEMU),source_sha256=P.sha(overlap.OUT)))


def score():
    wake_data = P.read(WORK/"wake.json")
    base = P.read(daily.OUT)
    previous = P.read(daily.daily.OUT)
    cue, info = make_cue()
    assert P.digest(cue)==wake_data["pcm_sha256"] and info==wake_data["cue"]
    assert P.sha(overlap.OUT)==wake_data["source_sha256"]
    assert P.sha(sender.BUILD/"board.bin")==base["sources"]["board.bin"]
    assert P.sha(clock.BUILD/"nyx_mac_listener.bin")==previous["capture_firmware_sha256"]
    assert P.sha(prior.CODEC/"nyx_first_command.bin")==previous["codec"]["opus_sha256"]
    assert P.sha(P.QEMU)==base["qemu_sha256"]==wake_data["qemu_sha256"]
    proc, model, qid = prior.load()
    assert qid==base["qwen"] and qid["device"].endswith("RTX 4090")
    turbo,decoder = prior.reader(base["turbo"]),prior.decoder_api()
    def readings(audio,expected):
        return dict(qwen=prior.evaluate(proc,model,audio,expected),turbo=prior.recognition(turbo,audio,expected))
    rows = []
    clock.reference.WORK=WORK
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp:
        work=Path(tmp)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as captures, ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for known, (src,mic,ref,clips) in zip(wake_data["records"],inputs(cue),strict=True):
                i = known["index"]
                assert i==src["index"] and P.digest(mic)==known["mic_sha256"] and P.digest(ref)==known["reference_sha256"]
                row = dict(known)
                label = row["label"]
                uart = WORK/"uart"/(row["input_sha256"]+".log")
                assert P.sha(uart)==row["uart_sha256"]
                logs.write(uart,label+"/wake.log")
                if row["events"]["O"]:
                    opened, = row["events"]["O"]
                    pcm, meta = clock.capture(work,captures,i,label,mic,ref,opened)
                    cold, immediate, identity = clock.opus(work,logs,i,label,pcm,decoder)
                    payload = struct.pack("<4I",i,opened,meta["first"],len(pcm))+pcm.tobytes()
                    log,(_, (sent,waited)) = sender.run_case(WORK,payload,cold)
                    logs.write(log,label+"/sender.log")
                    row.update(**meta,opus_uart_sha256=identity["opus_uart_sha256"],opus_watchdog=identity["opus_watchdog"],
                               sender_uart_sha256=P.sha(log),sender_watchdog=log.read_text().count("Task watchdog got triggered"),
                               waited_sender=waited,immediate=readings(immediate,row["command"]),
                               waited=readings(prior.decode(decoder,sent),row["command"]))
                rows.append(row)
                if len(rows)%24==0:
                    print("Scored",len(rows),flush=True)
        assert len(rows)==96
        by_id={(r["index"],r["role"]):r for r in base["records"]}
        counts={}
        for level in ("all",-6,0,6):
            selected=[r for r in rows if level=="all" or r["db"]==level]
            count=dict(cases=len(selected),timely=sum(bool(r["events"]["O"] and
                       r["cue_start"]<=r["opened"]<r["command_at"]) for r in selected))
            for treatment in ("immediate","waited"):
                for reader,key in (("qwen","complete"),("turbo","complete_suffix")):
                    outcomes=[bool(r.get(treatment,{}).get(reader,{}).get(key)) for r in selected]
                    comparison={}
                    for role in ("original","enhanced","willow"):
                        pairs=[(bool(by_id[r["index"],role].get("sender",{}).get(treatment,{}).get(reader,{}).get(key)),hit)
                               for r,hit in zip(selected,outcomes,strict=True)]
                        comparison[role]=dict(complete=sum(a for a,_ in pairs),gained=pairs.count((False,True)),lost=pairs.count((True,False)))
                    count[treatment+"_"+reader]=dict(complete=sum(outcomes),against=comparison)
            counts[str(level)]=count
        wave=work/CUE.name
        wave.write_bytes(overlap.peak.wav_bytes(cue))
        for p in (TRACE,archive):
            with ZipFile(work/p.name) as z:
                assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        P.save(work/OUT.name,dict(records=rows,counts=counts,cue=dict(info,wav_sha256=P.sha(wave),
               dry_readings=readings(cue,"Hey Nix")),model=base["model"],qwen=qid,turbo=base["turbo"],
               guest_sha256=P.sha(GUEST),qemu_sha256=P.sha(P.QEMU),sender_sha256=P.sha(sender.BUILD/"board.bin"),
               archive_sha256=P.sha(work/archive.name),capture_archive=TRACE.name,capture_archive_sha256=P.sha(work/TRACE.name),
               sources={p.name:P.sha(p) for p in (Path(__file__),daily.OUT,overlap.OUT,overlap.peak.OUT,
                       gains.PACK,gains.fit.FIT,overlap.board.bella.CUE,daily.daily.OUT)},
               policy="One candidate, frozen original-level af_bella PCM plus the existing 5.68 dB boost on the "
                      "three trained code stems alone. Verify original, whole-cue boost and erased mic hashes on "
                      "each of the 96 consumed development trials. Native board receiver, actual candidate open, "
                      "full history, board AEC/AGC and complete decoded immediate and delayed local sends. "
                      "Keep all non-opens in each denominator. No new FAR exposure or human speech."))
        (work/archive.name).replace(archive)
        (work/TRACE.name).replace(TRACE)
        wave.replace(CUE)
        (work/OUT.name).replace(OUT)
    print(counts["all"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["wake","score"])
    WORK.mkdir(exist_ok=True,parents=True)
    if parser.parse_args().action=="wake":
        wake()
    else:
        score()
