"""Compare recognizers and the pre-OPEN crop on matching P4 packets, including missed openings."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from check_c161_v5_far_command_noise import normalized
import check_panel_qwen_asr as asr
import check_panel_music_transfer as transfer
import score_panel_willow_reference as willow

ch, tx = asr.ch, asr.tx


def cases(source):
    if source == "willow":
        d = ch.read(willow.OUT)
        for r in d["records"]:
            if not r["role"].startswith("willow_"):
                continue
            cap = r.get("capture")
            item = None if cap is None else (
                f'opus/{r["index"]:03}-willow.log', cap["emitted"]-cap["first"],
                cap["lanes"]["selected"]["decoded_sha256"], r["opened"]-cap["first"])
            yield r["index"], f'{r["role"]}.{r["external_db"]}', r["carrier"], r["timely"], r["turbo"], item
    elif source == "music":
        for r in ch.read(transfer.OUT)["guest"]:
            if r["view"] != "both":
                continue
            cap = r.get("uplink")
            item = None if cap is None else (
                f'{r["index"]:02}-opus.log', cap["samples"], cap["pcm_sha256"], r["opened"][0]-r["finish"][1])
            yield r["index"], f'{r["panel"]}.{r["source"]["external_db"]}', r["source"]["command"], r["timely"], r["turbo"], item


def main(source):
    doc = willow.OUT if source == "willow" else transfer.OUT
    out = ch.RESULTS / f"4090-c161-qwen-{source}-paired.json"
    archive = doc.with_suffix(".zip")
    assert ch.sha(archive) == ch.read(doc)["archive_sha256"]
    proc, model, ident = asr.load()
    assert ident == ch.read(asr.OUT)["model"]
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    rows, summary = [], defaultdict(Counter)
    with ZipFile(archive) as z:
        for i, group, command, timely, turbo, item in cases(source):
            row = dict(index=i, group=group, command=command, timely=timely, turbo=turbo, full=False, cropped=False)
            if item:
                name, samples, digest, op = item
                uart = z.read(name).decode().replace("\r\n","\n")
                pkts = uplink.packets(uart, i, samples, 0)
                x = uplink.decode(decoder, pkts)
                assert ch.digest(x) == digest
                cut = max(0, op-1600)
                assert cut < len(x)
                row.update(pcm_sha256=digest, packets=len(pkts), cut=cut)
                want = normalized(command)
                for mode,audio in (("full",x),("cropped",x[cut:])):
                    text = asr.transcribe(proc, model, audio)
                    row[mode+"_text"] = text
                    row[mode] = normalized(text)[-len(want):] == want
            summary[group].update(trials=1, opened=int(item is not None), timely=int(timely), turbo=int(turbo),
                full=int(row["full"]), cropped=int(row["cropped"]), full_timely=int(row["full"] and timely),
                gained_vs_turbo=int(row["full"] and not turbo), lost_vs_turbo=int(turbo and not row["full"]))
            rows.append(row)
    assert len(rows) == (120 if source=="willow" else 66)
    ch.save(out, dict(records=rows, summary=summary, model=ident, sources={
        f.name:ch.sha(f) for f in (doc,archive,asr.OUT,Path(asr.__file__),Path(__file__))},
        policy="Same English-only greedy ASR on full selected Opus (primary) and from the fixed 100 ms "
               "before device OPEN (secondary). Prior complete-suffix metric. All missed opens count as incomplete."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("willow","music"))
    main(parser.parse_args().source)
