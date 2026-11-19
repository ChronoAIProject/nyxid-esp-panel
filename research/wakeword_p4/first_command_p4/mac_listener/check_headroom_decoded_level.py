"""Apply one post-decode peak normalization rule to the original and fixed-gain P4 uplinks."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import check_connected_mic_headroom as headroom

ch, balanced, tx = headroom.ch, headroom.balanced, headroom.tx
OUT = ch.RESULTS / "4090-c161-headroom-decoded-level.json"
TARGET_DBFS = -1.0


def main():
    fixed = ch.read(headroom.OUT)
    proc, model, qid = balanced.original.asr.load()
    assert qid == fixed["qwen"]
    turbo = tx.diag.reader(fixed["turbo"])
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    stats, records = defaultdict(Counter), []
    for name, path in (("previous", balanced.OUT), ("fixed", headroom.OUT)):
        report = ch.read(path)
        assert report["qwen"] == qid and report["turbo"] == fixed["turbo"]
        archive = path.with_suffix(".zip")
        assert ch.sha(archive) == report["archive_sha256"]
        with ZipFile(archive) as z:
            for row in report["records"]:
                if row["role"] != "both":
                    continue
                s = row["source"]
                tally = stats[f'{name}.{s["external_db"]}']
                tally.update(trials=1)
                record = dict(profile=name, index=row["index"], source=s, opened=row["opened"])
                if row["opened"]:
                    cap = row["uplink"]
                    log = z.read(f'{row["index"]:02}-opus.log').decode().replace("\r\n", "\n")
                    packets = uplink.packets(log, row["index"], cap["samples"], 0)
                    audio = uplink.decode(decoder, packets)
                    assert ch.digest(audio) == cap["pcm_sha256"] == row["qwen"]["pcm_sha256"]
                    x = audio.astype(float)
                    peak = float(np.abs(x).max())
                    gain = 32768 * 10**(TARGET_DBFS/20) / peak if peak else 1.
                    samples = np.rint(x * gain).astype("<i2")
                    reading = balanced.evaluate(proc, model, samples, s["command"])
                    whisper = uplink.recognition(turbo, samples, s["command"])
                    record.update(decoded_pcm_sha256=cap["pcm_sha256"], gain=gain, qwen=reading, turbo=whisper)
                    for key, before, after in (("qwen", row["qwen"]["complete"], reading["complete"]),
                                               ("turbo", row["turbo"], whisper["complete_suffix"])):
                        tally.update({key+"_before":int(before), key:int(after),
                                      key+"_gained":int(after and not before), key+"_lost":int(before and not after)})
                records.append(record)
    assert len(records) == 72 and all(c["trials"] == 12 for c in stats.values())
    tmp = OUT.with_suffix(".tmp")
    ch.save(tmp, dict(records=records, summary=stats, target_peak_dbfs=TARGET_DBFS, qwen=qid,
                      turbo=fixed["turbo"], sources={p.name: ch.sha(p) for p in (headroom.OUT, balanced.OUT,
                      headroom.OUT.with_suffix(".zip"), balanced.OUT.with_suffix(".zip"), Path(__file__))},
                      policy="Apply one constant gain to each complete decoded, already received utterance, based "
                             "only on its absolute peak. No crop, hint or command boundary. All original and fixed "
                             "gain trials included. This batch ASR rule requires the full utterance to be available."))
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
