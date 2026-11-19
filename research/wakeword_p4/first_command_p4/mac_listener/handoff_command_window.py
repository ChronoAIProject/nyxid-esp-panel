"""Transcribe the same decoded packets from the previously tested OPEN-minus-100-ms boundary."""
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile
import argparse

import handoff_barge_in as handoff
import handoff_fade_delay as delayed

listener = handoff.listener
uplink = handoff.probe.uplink


def records(history, mode):
    for j, row in enumerate(history["records"]):
        if mode == "delay":
            yield j, row["delay_ms"], row["result"], row["uplink"]
        else:
            for k, name in enumerate(("continue", "interrupt")):
                run = row[name]
                yield 2*j+k, (None if name == "continue" else 0), run, run["uplink"]


def main(mode):
    base = delayed.BASE if mode == "delay" else handoff.BASE
    report = base.with_suffix(".json")
    parent = listener.read(report)
    reader, identity = listener.load_reader()
    assert identity == parent["asr"]
    archive = base.with_suffix(".zip")
    assert parent["archive_sha256"] == listener.sha(archive)
    decoder = uplink.decoder_api()
    rows = []
    with ZipFile(archive) as z:
        for i, delay, run, full in records(parent, mode):
            opened, = run["opened"]
            info = "\n".join(run["uart_events"])
            _, start, count, *_ = listener.values(info, "FINISH")
            assert count == full["samples"] and start == max(0, opened//160*160-32000)
            with z.open(f"{i:02}-opus.log") as stream, TextIOWrapper(stream, encoding="utf-8") as uart:
                log = uart.read()
            packets = uplink.packets(log, i, count, 0)
            pcm = uplink.decode(decoder, packets)
            assert len(packets) == full["packets"] and listener.digest(pcm) == full["pcm_sha256"]
            offset = max(0, opened - 1600 - start)
            command = run["source"]["carrier"]
            short = uplink.recognition(reader, pcm[offset:], command)
            rows.append(dict(index=i, delay_ms=delay, source=run["source"], opened=opened,
                    capture_start=start, crop_offset=offset, complete=full["recognition"], from_wake=short,
                    pcm_sha256=full["pcm_sha256"], cropped_pcm_sha256=listener.digest(pcm[offset:])))
    out = base.with_name(base.name+"-window.json")
    assert len(rows) == 32
    listener.save(out, dict(parent_sha256=listener.sha(report), archive_sha256=listener.sha(archive), records=rows,
            asr=identity, code_sha256=listener.sha(Path(__file__)), decoder=decoder.opus_get_version_string().decode(),
            policy="Preserve P4 packets and decoder history. Begin Whisper input 100 ms before observed OPEN. "
                   "Offset uses the listener clock only. Compare the same recording's full and cropped readings."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["immediate", "delay"])
    main(p.parse_args().group)
