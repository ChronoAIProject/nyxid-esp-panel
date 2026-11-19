"""Score official default Willow detections and the complete uplinks on matched connected commands."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import connected_panel_willow as study
import check_balanced_connected_commands as balanced
from score_handoff_willow import willow_events
from score_paired_piper_willow import capture

ch, tx, panel = balanced.ch, balanced.tx, balanced.panel
OUT = ch.RESULTS / "4090-c161-connected-panel-willow.json"


def main():
    plan = ch.read(study.PLAN)
    plan_hash = ch.sha(study.PLAN)
    history = ch.read(balanced.OUT)
    nyx = {r["index"]:r for r in history["records"]}
    small, ident = ch.load_reader()
    assert ident == history["small"]
    turbo = tx.diag.reader(history["turbo"])
    proc, model, qid = balanced.original.asr.load()
    assert qid == history["qwen"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == history["codec"]
    decoder = uplink.decoder_api()
    rows, totals = [], defaultdict(Counter)
    tmp = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        uplink.WORK = work
        native = tx.panel.grid.room.rooms.c161_binding()
        matcher = tx.single.timed.binding(work)
        for i,s in enumerate(plan["records"]):
            wav = Path(s["path"])
            pcm, rate = sf.read(wav,dtype="int16")
            assert rate == 16000 and ch.digest(pcm) == s["pcm_sha256"] and ch.sha(wav) == s["sha256"]
            match = nyx[s["host_index"]]
            assert s["command_pcm_sha256"] == match["new_level"]["pcm_sha256"]
            hits = willow_events(i,s,plan_hash,z,study.WORK / "runs")
            first = (hits[0]+255)//256*256 if hits else None
            timely = first is not None and s["start"] <= first < s["end"]
            clock = s["transmit_event"]
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher,native,pcm)
            result = panel.middle.pair.admit(raw,clock,panel.middle.single.hits(pcm,clock),np.zeros_like(pcm))
            if s["target_id"] == "nyx_marked":
                assert result["opened"] == match["opened"] and s["pcm_sha256"] == match["mic_sha256"]
            if s["target_id"] == "no_address":
                assert result == match["no_address"]
            row = dict(index=i,source=s,events=hits,opened=first,timely=timely,nyx_opened=result["opened"],
                       small=False,turbo=False,qwen=False)
            if first is not None and s["target_id"].startswith("willow_"):
                cap = capture(small,decoder,work,z,i,"willow",pcm,first,s["carrier"])
                serial = z.read(f"opus/{i:03}-willow.log").decode().replace("\r\n","\n")
                for lane,name in enumerate(("selected","raw")):
                    packets = uplink.packets(serial,i,cap["emitted"]-cap["first"],lane)
                    audio = uplink.decode(decoder,packets)
                    assert ch.digest(audio) == cap["lanes"][name]["decoded_sha256"]
                    if lane == 0:
                        reading = balanced.evaluate(proc,model,audio,s["carrier"])
                        t = uplink.recognition(turbo,audio,s["carrier"])
                row.update(capture=cap, small=cap["lanes"]["selected"]["asr"]["complete_suffix"],
                           turbo=t["complete_suffix"],qwen=reading["complete"],qwen_reading=reading,turbo_reading=t)
            totals[f'{s["target_id"]}.{s["external_db"]}'].update(trials=1,
                opens=int(bool(hits)),timely=int(timely),small=int(row["small"]),turbo=int(row["turbo"]),qwen=int(row["qwen"]),
                timely_qwen=int(timely and row["qwen"]), nyx_opens=int(bool(result["opened"])))
            rows.append(row)
        for f in sorted(work.glob("*-attempt*.gz")):
            z.write(f,"retry/"+f.name)
    assert len(rows) == 96 and len(nyx) == 36
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    archive = OUT.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(OUT,dict(records=rows,summary=totals,paired_nyx_summary=history["summary"],model="wn9_heywillow_tts",
        firmware=history["codec"],nyx_firmware=history["firmware"],qwen=qid,small=ident,turbo=history["turbo"],
        archive_sha256=ch.sha(archive),sources={p.name:ch.sha(p) for p in (balanced.OUT,study.PLAN,Path(__file__))},
        policy="Verify all official raw frame returns at DEFAULT after 81600 priming samples. Round first "
        "hit to the 256-sample block boundary, then transcribe all selected P4 Opus packets. Closed "
        "trials count as incomplete. The same external command PCM follows same-voice generated addresses; "
        "the local panel plays the quieter Nyx demo in all cases."))


if __name__ == "__main__":
    main()
