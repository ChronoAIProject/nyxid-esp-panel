"""Identify exact previous P4 sender replays for every open under 150 ms confirmation."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import check_part_model_first_commands as commands
import check_parts_confirmation as study

fit = study.fit
OUT = fit.RESULTS / (study.STEM + "-first-commands.json")


def main(archive):
    original = fit.read(commands.OUT)
    assert fit.sha(archive) == original["archive_sha256"]
    prior = {r["index"]:r for r in original["records"]}
    speech = fit.RESULTS / (study.STEM+"-speech.json")
    rows, counts = [], defaultdict(Counter)
    with ZipFile(archive) as uart:
        for source in fit.read(speech)["records"]:
            if source["group"] != "connected" or source["role"] not in ("Hey Nyx","Hey Willow"):
                continue
            n = int(source["key"])
            old = prior[n]
            assert old["pcm_sha256"] == source["pcm_sha256"] and old["command_at"] == source["command_at"]
            routes = {"willow":source["willow"]} if source["role"] == "Hey Willow" else {
                "c161":source["c161"], "150ms":source["union"]["150ms"]}
            for arm, times in routes.items():
                opened = (times[0]+255)//256*256 if times else None
                timely = opened is not None and source["start"] <= opened < source["command_at"]
                tally = counts[arm]
                tally.update(cases=1, opened=int(opened is not None), timely=int(timely))
                row = dict(index=n, route=arm,voice=source["voice"],command=old["command"],pcm_sha256=old["pcm_sha256"],
                           command_at=old["command_at"], opened=opened, timely=timely)
                if opened is not None:
                    path = old["received"][str(opened)]
                    assert path["opened"] == opened
                    row["uart"] = {}
                    for stage in ("capture","opus","sender"):
                        member = f"{n}/{opened}/{stage}.log"
                        digest = sha256(uart.read(member)).hexdigest()
                        assert digest == path[stage+"_uart_sha256"] and path[stage+"_watchdog"] == 0
                        row["uart"][member] = digest
                    row["readings"] = {}
                    for condition, asr in path["readings"].items():
                        qwen = asr["qwen"]
                        turbo = asr["turbo"]
                        row["readings"][condition] = dict(qwen=qwen["text"], turbo=turbo["text"],
                                  qwen_complete=qwen["complete"], turbo_complete=turbo["complete_suffix"],
                                  pcm_sha256=qwen["pcm_sha256"])
                        tally[condition+"_qwen"] += int(qwen["complete"])
                        tally[condition+"_turbo"] += int(turbo["complete_suffix"])
                rows.append(row)
    for route in ("c161", "willow"):
        assert all(original["counts"][route][key] == value for key,value in counts[route].items())
    assert {k:x["cases"] for k,x in counts.items()} == {"c161":12, "willow":12, "150ms":12}
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        p = Path(tmp)/OUT.name
        fit.save(p, dict(records=rows, counts=counts, sources={p.name:fit.sha(p) for p in (
                 Path(__file__), speech, commands.OUT)}, archive=original["archive"], archive_sha256=fit.sha(archive),
                 scope="Every resulting request uses exactly the previously replayed original microphone waveform "
                 "and rounded open sample, with the same zero reference and sender. Retrieved both ASR transcripts "
                 "from those verified digital replays. No new sender runs or independent trials."))
        p.replace(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
