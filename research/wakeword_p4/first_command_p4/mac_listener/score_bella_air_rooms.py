"""Decode the full first send for each intended address in both Aachen rooms."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import argparse

import bella_air_rooms as rooms
import bella_air_master as master
import score_bella_headroom_commands as previous

P, scoring = previous.P, previous.scoring


def score(experiment, expected_count):
    out = experiment.OUT.with_name(experiment.OUT.stem+"-commands.json")
    archive = experiment.WORK / out.with_suffix(".zip").name
    wakes, parent = P.read(experiment.OUT), P.read(previous.OUT)
    assert wakes["plan_sha256"] == P.sha(experiment.PLAN) and wakes["build"]["qemu_sha256"] == P.sha(P.QEMU)
    images = (scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",
              scoring.sender.BUILD/"board.bin")
    expected = P.read(previous.baseline.earlier.OUT)["sources"]
    assert all(P.sha(p) == expected[str(p)] for p in images)
    proc, model, qwen = scoring.prior.load()
    assert qwen == parent["qwen"] and "4090" in qwen["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == parent["decoder"]
    readers = proc, model, scoring.prior.reader(parent["turbo"]), decoder
    cache = experiment.WORK / ("sends-" + P.sha(experiment.OUT)[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, counts, files = [], defaultdict(Counter), {}
    with TemporaryDirectory(dir=experiment.WORK) as temp:
        work = Path(temp)
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as dest:
            for i, r in enumerate(wakes["records"]):
                if r["role"] == "absent":
                    continue
                engine = "willow" if r["role"] == "willow" else "nyx"
                opened = r[engine+"_open"]
                row = {k: r[k] for k in ("label", "room", "direction", "scene", "role", "mode", "command", "end")}
                row.update(opened=opened, timely=r[engine+"_timely"])
                stats = counts["/".join((r["room"], r["mode"], r["role"]))]
                stats.update(cases=1, opened=int(opened is not None), timely=int(row["timely"]))
                if opened is not None:
                    u, shard = scoring.score_case(dict(r, reference_sha256=r["ref_sha256"]), i, opened, readers, work, cache)
                    assert all(u[k] == 0 for k in ("capture_watchdog", "opus_watchdog", "sender_watchdog")), (r["label"], shard)
                    row["uplink"] = u
                    with ZipFile(shard) as saved:
                        for name in saved.namelist():
                            if name.endswith(".log"):
                                data = saved.read(name)
                                dest.writestr(name, data)
                                files[name] = experiment.study.sha256(data).hexdigest()
                    for timing, v in u["readings"].items():
                        stats[timing+"_qwen"] += int(v["qwen"]["complete"])
                        stats[timing+"_whisper"] += int(v["turbo"]["complete_suffix"])
                rows.append(row)
                print(len(rows), r["label"], opened, flush=True)
        assert len(rows) == len({r["label"] for r in rows}) == expected_count
        assert len(files) == 3*sum(r["opened"] is not None for r in rows)
        deps = (experiment.PLAN, experiment.OUT, experiment.PROTO, previous.OUT,
                Path(__file__).resolve().relative_to(Path.cwd()), *images)
        P.save(work/out.name, dict(records=rows, counts=counts, files=files, qwen=qwen, turbo=parent["turbo"],
                                   decoder=parent["decoder"], sources={str(p): P.sha(p) for p in deps},
                                   archive=archive.name, archive_sha256=P.sha(work/archive.name)))
        (work/archive.name).replace(archive)
        (work/out.name).replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hall-master", action="store_true")
    if parser.parse_args().hall_master:
        score(master, 18)
    else:
        score(rooms, 72)
