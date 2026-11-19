"""Measure frozen reusable prefixes alone and at identical command slots across 18 recorded commands."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import torch

import prepare_reusable_nyx as study
import score_late_nick_gate as veto
from score_paired_piper_willow import capture, uplink

ch, voices = study.ch, study.voices
OUT = ch.RESULTS / (study.STEM + ".json")


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    plan = ch.read(study.PLAN)
    paths = (study.PACK, study.BASE, study.context.PLAN, voices.gate.fit.COEFF,
             veto.fit.FIT, Path(study.__file__))
    assert plan["sources"] == {p.name: ch.sha(p) for p in paths}
    fw = uplink.firmware()
    reader, identity = ch.load_reader()
    dec = uplink.decoder_api()
    net = voices.timing.spoken.natural.Detector()
    spec = ch.read(voices.gate.fit.COEFF)
    fit = ch.read(veto.fit.FIT)
    assert fit["models"] == spec["models"] == net.identity
    costs = ch.read(voices.timing.names.OUT)
    captures, records = [], []
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        uplink.WORK = work
        for i, ((row, pcm), published) in enumerate(zip(study.inputs(), plan["records"], strict=True)):
            assert row == published
            raw = net.score(pcm)
            measured, checks, _ = voices.score(net, pcm, raw["c161"], raw["proposals"], raw["events"], spec, costs)
            stages = dict(raw=dict(c161=raw["c161"]), filtered=dict(c161=[], proposals=raw["proposals"]))
            state = dict(start=row["cue_at"], end=len(pcm), command_at=row["cue_end"], stages=stages)
            events = veto.evaluate(state, checks, spec, {"paired": fit["heads"]["paired"]}, costs)
            assert events["prior"] == measured["linear"]
            events["c161"] = measured["c161"]
            solo = row["pair"] is None
            group = "alone" if solo else "with_command"
            item = dict(index=i, cue=row["cue"], pair=row["pair"], pcm_sha256=row["pcm_sha256"],
                        proposals=raw["proposals"], checks=checks, routes={})
            per_open = {}
            for name, ts in events.items():
                assert all(row["cue_at"] <= t <= len(pcm) for t in ts)
                stats = summary[group][row["role"]+"."+row["style"]][name]
                stats.update(trials=1, hits=bool(ts))
                route = dict(events=ts)
                if ts and not solo:
                    timely = ts[0] < row["cue_end"]
                    route.update(timely=timely, lead_samples=row["cue_end"]-ts[0])
                    stats["timely"] += timely
                    if row["role"] == "Hey Nyx":
                        if ts[0] not in per_open:
                            idx = len(captures)
                            cap = capture(reader, dec, work, logs, idx, "reusable", pcm, ts[0], row["command"])
                            captures.append(dict(id=idx, index=i, opened=ts[0], **cap))
                            per_open[ts[0]] = idx
                        idx = per_open[ts[0]]
                        complete = captures[idx]["lanes"]["selected"]["asr"]["complete_suffix"]
                        stats.update(complete=complete, timely_complete=timely and complete)
                        route.update(capture=idx, complete=complete)
                item["routes"][name] = route
            records.append(item)
            if len(records) % 19 == 0:
                print(len(records), "inputs", len(captures), "captures", flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for p in retries:
            logs.write(p, "retry/"+p.name)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for cap in captures:
            text = logs.read(f'opus/{cap["id"]:03}-reusable.log').decode().replace("\r\n", "\n")
            for lane, name in enumerate(("selected", "raw")):
                packets = uplink.packets(text, cap["id"], cap["emitted"]-cap["first"], lane)
                assert len(packets) == cap["lanes"][name]["packets"]
                assert ch.digest(uplink.decode(dec, packets)) == cap["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=records, captures=captures, summary=summary, asr=identity, models=net.identity, **fw,
                      plan_sha256=ch.sha(study.PLAN), archive_sha256=ch.sha(archive), retries=[p.name for p in retries],
                      sources={p.name: ch.sha(p) for p in (Path(__file__), Path(veto.__file__), Path(voices.__file__),
                               voices.timing.names.OUT, veto.fit.FIT, voices.gate.fit.COEFF)},
                      policy="Every prepublished cue and command including no-command controls. Frozen native C161 "
                      "and two research routes. Component P95 clock, block-aligned delivery, two-second history. "
                      "Match entire ASR command suffix in the selected stream of the P4 emulator's encoded packets. "
                      "Actual physical timing and acoustic replay are pending."))


if __name__ == "__main__":
    main()
