"""Keep the transmitted Opus packets intact and vary the ASR input after decoding."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import check_bella_peak_playback as peak

ch = peak.ch
OUT = ch.RESULTS / "4090-c161-bella-decoded-history.json"


def main():
    baseline = ch.read(peak.bella.OUT)
    louder = ch.read(peak.GUEST)
    uplink = peak.tx.panel.grid.gap.uplink
    balanced = peak.study.prefix.balanced
    processor, model, qid = balanced.original.asr.load()
    assert qid == baseline["qwen"] == louder["qwen"] and uplink.firmware()==louder["codec"]==baseline["codec"]
    assert baseline["turbo"]==louder["turbo"]
    turbo = peak.tx.diag.reader(baseline["turbo"])
    decoder = uplink.decoder_api()
    rows = []
    for name,path,report,records in (("original",peak.bella.OUT,baseline,baseline["captures"]),
                                     ("louder",peak.GUEST,louder,louder["records"])):
        assert ch.sha(path.with_suffix(".zip")) == report["archive_sha256"]
        with ZipFile(path.with_suffix(".zip")) as logs:
            for row in records:
                s, cap, i = row["source"], row["uplink"], row["index"]
                raw = logs.read(f"{i:02}-opus.log").decode().replace("\r\n","\n")
                packets = uplink.packets(raw,i,cap["samples"],0)
                audio = uplink.decode(decoder,packets)
                assert ch.digest(audio)==cap["pcm_sha256"] and len(packets)==cap["packets"]
                # All paired opens are early, with the full two-second FIFO primed.
                opened, = row["opened"]
                emitted_at_open = opened//160*160
                assert emitted_at_open-row["finish"][1]==32000 and opened < s["command_at"]
                crop = audio[32000:]
                qwen = balanced.evaluate(processor,model,crop,s["command"])
                reading = uplink.recognition(turbo,crop,s["command"])
                file = row["file"] if name=="louder" else f'{row["batch"]}/{s["index"]:03}.wav'
                rows.append(dict(version=name,file=file,db=s["external_db"],paired=row.get("paired_grid",True),
                    opened=opened,command_at=s["command_at"],first_selected_sample=row["finish"][1],
                    packets=len(packets),full_pcm_sha256=cap["pcm_sha256"],asr_start_in_decoded_pcm=32000,
                    cropped_pcm_sha256=ch.digest(crop),command=s["command"],full_qwen=row["qwen"],
                    full_turbo=cap["turbo"],cropped_qwen=qwen,cropped_turbo=reading))
    assert len(rows)==50
    with TemporaryDirectory(dir=peak.WORK) as folder:
        temp = Path(folder)/OUT.name
        ch.save(temp,dict(records=rows,qwen=qid,turbo=baseline["turbo"],source_sha256={
            p.name:ch.sha(p) for p in (peak.bella.OUT,peak.GUEST,Path(__file__))},
            policy="On the receiver after decoding, compare the complete selected PCM with its suffix from "
                   "decoded sample index 32000. Read the same archived packet stream and the same end sample. "
                   "Use the original 24 paired cases at both gains and the two extra formerly missed weak cases "
                   "at the louder gain. Diagnostic on early fixed-code openings."))
        temp.replace(OUT)


if __name__=="__main__":
    main()
