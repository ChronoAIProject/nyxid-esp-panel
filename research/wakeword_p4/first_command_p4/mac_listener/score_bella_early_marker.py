"""Read the actual first commands sent from all 15 dual-marker weak-room presentations."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import bella_early_marker as early
import score_bella_gain_candidate as previous

P, scoring = previous.P, previous.scoring
OUT = early.BASE.with_name(early.BASE.name + "-commands.json")


def main():
    baseline = P.read(previous.OUT)
    named = {r["label"]: r for r in P.read(early.PLAN)["records"]}
    native = P.read(early.REPORT)
    assert native["plan_sha256"] == P.sha(early.PLAN)
    proc, model, qwen = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert "4090" in qwen["device"] and qwen == baseline["qwen"]
    assert decoder.opus_get_version_string().decode() == baseline["decoder"]
    images = (scoring.clock.BUILD / "nyx_mac_listener.bin", scoring.prior.CODEC / "nyx_first_command.bin",
              scoring.sender.BUILD / "board.bin")
    assert all(P.sha(p) == baseline["sources"][p.name] for p in images)
    assert P.sha(P.QEMU) == native["qemu_sha256"]
    readers = proc, model, scoring.prior.reader(baseline["turbo"]), decoder
    cache = early.WORK / ("sender-" + P.sha(early.REPORT)[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    gain_inputs = {r["label"]: r for r in P.read(early.fitted.gain_grid.PLAN)["records"]}
    rows, files, counts = [], {}, defaultdict(Counter)
    with TemporaryDirectory(dir=early.WORK) as tmp:
        temp = Path(tmp)
        archive = OUT.with_suffix(".zip")
        with ZipFile(temp/archive.name, "w", ZIP_DEFLATED) as dest:
            for n, r in enumerate(native["records"]):
                src = named[r["label"]]
                gain = gain_inputs[src["baseline"]]
                opened = r["candidate_read"]["sample"]
                row = dict(label=r["label"], scene=src["scene"], rate=src["rate"], command=gain["command"], opened=opened,
                           command_at=src["end"], pcm_sha256=src["pcm_sha256"], reference_sha256=src["ref_sha256"], readings={})
                c = counts[src["rate"]]
                c.update(cases=1, opened=opened is not None)
                if opened is not None:
                    record = dict(src, command=row["command"], reference_sha256=src["ref_sha256"])
                    observed, shard = scoring.score_case(record, n, opened, readers, temp, cache)
                    with ZipFile(shard) as z:
                        for name in ("capture", "opus", "sender"):
                            member = src["label"] + "/" + name + ".log"
                            raw = z.read(member)
                            assert observed[name + "_watchdog"] == 0
                            assert early.playback.wakes.digest(raw) == observed[name + "_uart_sha256"]
                            dest.writestr(member, raw)
                            files[member] = observed[name + "_uart_sha256"]
                    row.update({k: observed[k] for k in ("readings", "first", "stop", "fifo_sha256", "samples")})
                    for cond, read in row["readings"].items():
                        c[cond + "_qwen"] += int(read["qwen"]["complete"])
                        c[cond + "_turbo"] += int(read["turbo"]["complete_suffix"])
                rows.append(row)
        assert len(rows) == 15
        P.save(temp/OUT.name, dict(records=rows, counts=counts, files=files, qwen=qwen, turbo=baseline["turbo"],
               decoder=baseline["decoder"], sources={str(p):P.sha(p) for p in (early.REPORT, early.PLAN, previous.OUT, Path(__file__), *images)},
               archive_sha256=P.sha(temp/archive.name), cue_sha256=native["cue_sha256"]))
        (temp/archive.name).replace(archive)
        (temp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
