"""Mute panel playback and external music independently at the 24 original Nyx call times."""
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_board_reference as board
import score_bella_room_willow as clock

P, prior, study, tx = clock.P, clock.prior, board.study, board.tx
OUT = P.RESULTS / "4090-c161-bella-product-interference.json"
WORK = Path(".local/wakeword/c161-bella-product-interference")
TRACE = WORK / "4090-c161-bella-product-interference-uart.zip"


def rooms():
    history = P.read(board.bella.OUT)
    plan = P.read(clock.setup.PLAN)
    original = P.read(clock.reference.OUT)
    panel, rate = sf.read(study.mixed.panel.demo.WAVE,dtype="int16")
    assert rate == 16000 and P.sha(study.mixed.panel.demo.WAVE) == original["panel_sha256"]
    music, _, music_id = tx.study.grid.room.rooms.read_noise("music")
    v5, _ = tx.study.panel_wave()
    taps, _ = board.bella.gains.direction.filters()
    weak = {(r["file"], r["condition"]):r["result"]["pcm_sha256"] for r in P.read(board.weak.OUT)["records"]
            if r["condition"].startswith("back_") and r["role"] == "fixed"}
    with ZipFile(board.weak.PACK) as pack:
        files = {r["file"]:r for r in json.loads(pack.read("manifest.json"))["records"]}
        for i,(case, rec, old) in enumerate(zip(plan["cases"],history["captures"],original["records"],strict=True)):
            s = rec["source"]
            assert old["file"] == case["file"] and old["index"] == rec["index"] == i
            file = files[case["file"]]
            dry = study.playback.read_wav(pack,file)
            assert P.digest(dry) == s["dry_pcm_sha256"]
            ext = board.bella.rear_filter(dry,taps["back"])
            _, (hx, _, _) = tx.diag.room(s["geometry"])
            length = case["samples"] - 4800 - len(hx) + 1
            group = {"length":(file,range(length))}
            for p in (True,False):
                panel_signal = panel if p else np.zeros_like(panel)
                scene = study.mixed.room_scene(s,group,panel_signal,v5,music,music_id)
                ref = np.zeros(case["samples"],dtype="<i2")
                ref[s["panel_at"]:s["panel_at"]+len(panel)] = panel_signal
                if p:
                    assert P.digest(ref) == case["reference_sha256"]
                for m in (True,False):
                    mic, clipped = scene(ext,include_music=m)
                    assert len(mic) == case["samples"]
                    name = f"panel{int(p)}_music{int(m)}"
                    if s["external_db"] == -6:
                        assert P.digest(mic) == weak[file["file"],"back_"+name]
                    if p and m:
                        assert P.digest(mic) == old["mic_sha256"]
                    yield i,name,old,mic,ref,clipped


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    clock.reference.WORK = WORK
    base = P.read(clock.reference.OUT)
    compared = P.read(clock.OUT)
    assert P.sha(clock.BUILD/"nyx_mac_listener.bin") == compared["capture_firmware_sha256"]
    assert P.sha(prior.CODEC/"nyx_first_command.bin") == base["codec"]["opus_sha256"]
    assert P.sha(P.QEMU) == base["codec"]["qemu_sha256"]
    qproc, qmodel, qid = prior.load()
    assert qid == base["qwen"]
    turbo, decoder = prior.reader(base["turbo"]), prior.decoder_api()
    rows, counts = [],defaultdict(list)
    with TemporaryDirectory(dir=WORK) as folder:
        work = Path(folder)
        compact = OUT.with_suffix(".zip")
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as captures, ZipFile(work/compact.name,"w",ZIP_DEFLATED) as codecs:
            for i,name,old,mic,ref,clipped in rooms():
                opened, = old["continuous"]["opened"]
                row = dict(index=i,file=old["file"],condition=name,db=old["db"],command=old["command"],
                           command_at=old["command_at"],opened=opened,mic_sha256=P.digest(mic),
                           reference_sha256=P.digest(ref),clipped=clipped)
                if name == "panel1_music1":
                    product = old["product"]
                    row.update(fifo_sha256=product["fifo_sha256"],decoded_sha256=product["decoded_sha256"],
                               samples=product["samples"],qwen=product["qwen"],turbo=product["turbo"],
                               source=clock.reference.OUT.name)
                else:
                    label = f"{i:02}/{name}"
                    pcm, info = clock.capture(work,captures,i,label,mic,ref,opened)
                    assert info["samples"] == old["product"]["samples"] and info["first"] == old["continuous"]["finish"][1]
                    _, decoded, codec = clock.opus(work,codecs,i,label,pcm,decoder)
                    row.update(**info, **codec, qwen=prior.evaluate(qproc,qmodel,decoded,row["command"]),
                               turbo=prior.recognition(turbo,decoded,row["command"]))
                counts[name].append((row["qwen"]["complete"],row["turbo"]["complete_suffix"]))
                rows.append(row)
                if name == "panel0_music0":
                    print(i+1,"of 24",flush=True)
        for archive in (work/TRACE.name, work/compact.name):
            with ZipFile(archive) as z:
                assert len([n for n in z.namelist() if not n.startswith("retry/")]) == 72
                assert len(z.namelist()) == len(set(z.namelist())) and z.testzip() is None
        assert len(rows) == 96 and all(len(v)==24 for v in counts.values())
        summary = {k:dict(cases=len(v),qwen=sum(q for q,_ in v),turbo=sum(t for _,t in v)) for k,v in counts.items()}
        P.save(work/OUT.name,dict(records=rows,summary=summary,model=base["model"],qwen=qid,turbo=base["turbo"],
               codec=base["codec"],capture_firmware_sha256=compared["capture_firmware_sha256"],
               sources={p.name:P.sha(p) for p in (Path(__file__),clock.OUT,clock.reference.OUT,clock.setup.PLAN,
                                                    board.bella.OUT,board.weak.PACK,board.weak.OUT)},
               archive_sha256=P.sha(work/compact.name),capture_archive=TRACE.name,capture_archive_sha256=P.sha(work/TRACE.name),
               policy="Paired source mutes on the consumed 24-case rear grid. Hold the original full-mixture Nyx open, "
                      "room, external level, mic gain, two-second preroll, FIFO end, P4 PCM processing, Opus and GPU "
                      "readers fixed. Muting panel means acoustic panel and electrical R both silent. External music "
                      "has no electrical reference. These are conditional command readings, not wake counts."))
        (work/TRACE.name).replace(TRACE)
        (work/compact.name).replace(compact)
        (work/OUT.name).replace(OUT)
    print(summary)


if __name__ == "__main__":
    main()
