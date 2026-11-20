"""Score both readiness schedules at each detector's actual opening in the measured room."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import bella_surrey_orientations as study
import score_bella_early_marker as earlier

P, scoring = earlier.P, earlier.scoring
OUT = study.OUT.with_name("4090-bella-surrey-orientations-commands.json")
ARCHIVE = study.WORK / "4090-bella-surrey-orientations-command-uart.zip"


def main():
    wakes = P.read(study.OUT)
    assert wakes["plan_sha256"] == P.sha(study.PLAN) and P.sha(P.QEMU) == wakes["build"]["qemu_sha256"]
    parent = P.read(earlier.OUT)
    for image in (scoring.clock.BUILD / "nyx_mac_listener.bin", scoring.prior.CODEC / "nyx_first_command.bin",
                  scoring.sender.BUILD / "board.bin"):
        assert P.sha(image) == parent["sources"][str(image)]
    proc, model, qid = scoring.prior.load()
    assert qid == parent["qwen"] and "4090" in qid["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == parent["decoder"]
    readers = proc, model, scoring.prior.reader(parent["turbo"]), decoder
    cache = study.WORK / ("sends-"+P.sha(study.OUT)[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, counts, members = [], defaultdict(Counter), {}
    with TemporaryDirectory(dir=study.WORK) as temp:
        work = Path(temp)
        with ZipFile(work/ARCHIVE.name, "w", ZIP_DEFLATED) as dest:
            for i, r in enumerate(wakes["records"]):
                if r["role"] not in ("early-nyx", "willow"):
                    continue
                engine = "nyx" if r["role"]=="early-nyx" else "willow"
                opened = r[engine+"_open"]
                count = counts[r["mode"]+"/"+engine]
                count.update(cases=1, opened=int(opened is not None), timely=int(r[engine+"_timely"]))
                row = dict(label=r["label"], angle=r["angle"], direction=r["direction"], scene=r["scene"],
                           mode=r["mode"], engine=engine, command=r["command"], opened=opened, command_at=r["end"])
                if opened is not None:
                    item, shard = scoring.score_case(dict(r, reference_sha256=r["ref_sha256"]), i, opened, readers, work, cache)
                    for schedule, pair in item["readings"].items():
                        count[schedule+"_qwen"] += int(pair["qwen"]["complete"])
                        count[schedule+"_turbo"] += int(pair["turbo"]["complete_suffix"])
                    row["uplink"] = item
                    with ZipFile(shard) as logs:
                        for name in logs.namelist():
                            if name.endswith(".log"):
                                raw = logs.read(name)
                                dest.writestr(name, raw)
                                members[name] = study.sha256(raw).hexdigest()
                rows.append(row)
                if len(rows)%8==0:
                    print(len(rows),flush=True)
        assert len(rows)==48 and all(c["cases"]==12 for c in counts.values())
        P.save(work/OUT.name,dict(records=rows,counts=counts,files=members,archive_sha256=P.sha(work/ARCHIVE.name),
               decoder=parent["decoder"],qwen=qid,turbo=parent["turbo"],source_sha256=P.sha(study.OUT),
               protocol_sha256=P.sha(study.PROTOCOL),script_sha256=P.sha(Path(__file__))))
        (work/ARCHIVE.name).replace(ARCHIVE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
