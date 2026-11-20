"""Compare full and wake-clock-cropped decoded sends for all +2 dB room pairs."""
from collections import Counter, defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_c161_v5_far_command_noise import normalized
from crop_received_command import sample_offset
import score_prospective_bella_commands as baseline

scoring = baseline.scoring
P = scoring.P
OUT = baseline.OUT.with_name(baseline.study.STEM + "-wake-crop.json")
PROTOCOL = baseline.study.ROOT / "experiments/bella_prospective_command_protocol.md"


def metrics(readings, command):
    expected = normalized(command)
    result = {}
    for reader, key in (("qwen", "complete"), ("turbo", "complete_suffix")):
        reading = readings[reader]
        words = normalized(reading["text"])
        assert expected and reading[key] == (words[-len(expected):] == expected)
        result[reader + "_suffix"] = reading[key]
        result[reader + "_exact"] = words == expected
    return result


def score_opened(work, logs, row, readers, counts, conditions, expected_index):
    proc, model, turbo, decoder = readers
    opus_log = row["label"] + "/opus.log"
    opus = scoring.sender.archived(logs, opus_log, row["uart"][opus_log])
    n, = {int(k) for k in re.findall(r"^UPLINK (\d+) [01] ", opus, re.M)}
    if expected_index is not None:
        assert n == expected_index
    original = scoring.prior.packets(opus, n, row["samples"], 0)
    sender_log = row["label"] + "/sender.log"
    log = work / "sender.log"
    log.write_text(scoring.sender.archived(logs, sender_log, row["uart"][sender_log]))
    lanes = scoring.sender.compare(log, n, row["opened"], row["first"], row["samples"], original)
    for condition, (packets, timing) in zip(("immediate", "waited"), lanes, strict=True):
        pcm = scoring.prior.decode(decoder, packets)
        full = row["readings"][condition]
        assert P.digest(pcm) == full["qwen"]["pcm_sha256"]
        offset = sample_offset(row["opened"], row["first"], timing["sent_silence"] * 320)
        assert 0 <= offset < len(pcm)
        crop = pcm[offset:]
        readings = dict(qwen=scoring.prior.evaluate(proc, model, crop, row["command"]),
                        turbo=scoring.prior.recognition(turbo, crop, row["command"]))
        before, after = metrics(full, row["command"]), metrics(readings, row["command"])
        for metric in before:
            prefix = condition + "_" + metric
            counts.update({prefix + "_full": before[metric], prefix + "_crop": after[metric],
                                  prefix + "_gained": after[metric] and not before[metric],
                                  prefix + "_lost": before[metric] and not after[metric]})
        conditions[condition] = dict(offset=offset, full_pcm_sha256=P.digest(pcm),
                                              full=before, crop=after, readings=readings)


def main():
    source = P.read(baseline.OUT)
    assert P.sha(baseline.TRACE) == source["archive_sha256"]
    proc, model, qwen = scoring.prior.load()
    assert qwen == source["qwen"] and "4090" in qwen["device"]
    turbo = scoring.prior.reader(source["turbo"])
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == source["decoder"]
    records = []
    counts = defaultdict(Counter)
    with TemporaryDirectory(dir=baseline.study.WORK) as folder, ZipFile(baseline.TRACE) as logs:
        work = Path(folder)
        for row in source["records"]:
            if row["level"] != 2:
                continue
            group = row["mode"] + "/" + row["role"]
            counts[group]["cases"] += 1
            item = {k: row[k] for k in ("room", "role", "mode", "label", "command", "opened")}
            item["conditions"] = {}
            records.append(item)
            if row["opened"] is None:
                continue
            counts[group]["opened"] += 1
            score_opened(work, logs, row, (proc, model, turbo, decoder), counts[group],
                         item["conditions"], expected_index=row["index"])
        assert len(records) == 48 and len(counts) == 4 and all(v["cases"] == 12 for v in counts.values())
        paths = (Path(__file__), baseline.OUT, PROTOCOL)
        P.save(work / OUT.name, dict(records=records, counts=counts, qwen=qwen, turbo=source["turbo"],
               decoder=source["decoder"], archive_sha256=source["archive_sha256"],
               sources={p.name: P.sha(p) for p in paths}, margin_samples=1600))
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
