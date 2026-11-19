"""Aggregate the matched calls, counting equivalent written forms of twenty percent."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from transition_inputs import AUDIO, PLAN, STEM, WORK, SCHEDULES
from replay import read, sha


def command_words(s):
    tokens = re.findall(r"[a-z]+|[0-9]+|%", s.lower())
    return [{"20":"twenty", "%":"percent"}.get(t,t) for t in tokens]


def summarize(source, output):
    plan, report = read(source/PLAN), read(output/(STEM+".json"))
    assert report["plan_sha256"] == sha(source/PLAN)
    assert sha(source/AUDIO) == plan["archive_sha256"]
    archive = output/(STEM+"-uart.zip")
    assert sha(archive) == report["uart_sha256"]
    records = []
    with ZipFile(archive) as uart:
        def check(name, expected):
            assert hashlib.sha256(uart.read(name)).hexdigest() == expected
        for r in report["compatibility"]:
            check(f"mono-{r['index']}.log", r["uart_sha256"])
        for src, r in zip(plan["records"],report["records"],strict=True):
            index = r["index"]
            assert index == src["index"]
            check(f"{index:03}.log", r["uart_sha256"])
            row = dict(index=index, group=src["group"], role=src["role"], timing=src.get("timing"),
                       opened=r["first"]>=0, strict={}, equivalent={})
            target = command_words(src["carrier"])
            for lane, reading in r["uplink"].items():
                pair = 0 if lane in ("raw","selected") else 1
                check(f"{index:03}-opus-{pair}.log",reading["uart_sha256"])
                row["strict"][lane] = reading["exact_suffix"]
                row["equivalent"][lane] = command_words(reading["asr"]["text"])[-len(target):] == target
            records.append(row)
    totals = []
    for timing in (*SCHEDULES, "all"):
        group = [r for r in records if r["group"] == "challenge" and (timing=="all" or r["timing"]==timing)]
        totals.append(dict(timing=timing, inputs=len(group), opens=sum(r["opened"] for r in group),
            **{metric:{lane:sum(r[metric].get(lane, False) for r in group)
                       for lane in ("raw","delayed_raw","processed","selected")} for metric in ("strict","equivalent")}))
    return dict(plan_sha256=sha(source/PLAN), report_sha256=sha(output/(STEM+".json")),
                records=records, totals=totals)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,default=WORK/"inputs")
    p.add_argument("--output",type=Path,default=WORK/"score")
    args=p.parse_args()
    print(json.dumps(summarize(args.inputs,args.output),indent=2))
