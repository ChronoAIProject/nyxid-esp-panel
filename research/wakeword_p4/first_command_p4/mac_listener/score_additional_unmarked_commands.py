"""Pair second-cohort unmarked proposals with full P4-encoded first commands and ordinary speech controls."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import check_additional_connected_voices as fresh
import check_overlap_first_commands as earlier
import score_overlap_name_history as previous

ch, mixed, tx = fresh.ch, fresh.mixed, fresh.tx
OUT = ch.RESULTS / (fresh.STEM+"-unmarked.json")


def main():
    host = ch.read(fresh.OUT)
    weight = ch.read(previous.fit.FIT)
    old = ch.read(previous.score.OUT)
    calibrated = ch.read(previous.OUT)
    background = ch.read(previous.background.OUT)
    assert weight["overlap_sha256"] == host["models"]["overlap"] == old["models"]["overlap"]
    assert host["five_sha256"] == old["five_sha256"] == weight["five_sha256"] and host["cutoff"] == weight["original_cutoff"]
    assert calibrated["sources"][previous.fit.FIT.name] == ch.sha(previous.fit.FIT)
    assert host["pack_sha256"] == ch.sha(fresh.PACK)
    arm = weight["arms"]["current"]
    reference = ch.read(fresh.timing.GUEST)
    small, identity = ch.load_reader()
    turbo = tx.diag.reader(reference["turbo"])
    proc, model, qid = fresh.prefix.balanced.original.asr.load()
    assert identity == reference["small"] and qid == reference["qwen"]
    uplink = earlier.codec.uplink
    assert uplink.firmware() == reference["codec"]
    decoder = uplink.decoder_api()
    rows, captures, summary = [], [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        uplink.WORK = work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for (s, name, pcm, _), h in zip(fresh.scenes(),host["records"],strict=True):
                if name == "marked":
                    continue
                raw = h["result"]
                assert (s,name,ch.digest(pcm)) == (h["source"],h["track"],raw["pcm_sha256"])
                ts = [c["t"] for c in h["checks"] if previous.fit.dot(c["codes"],arm) >= arm["cutoff"]]
                hits = dict(raw,c161=sorted(set(raw["c161"]+ts)))
                echo = mixed.panel.middle.single.hits(pcm,s["transmit_event"])
                current = mixed.panel.middle.pair.admit(hits,s["transmit_event"],echo,np.zeros_like(pcm))
                opens = dict(baseline=raw["opened"],overlap=h["overlap"]["opened"],
                             confirm_0=h["confirm_0"]["opened"],current=current["opened"])
                cache, results = {}, {}
                for method, events in opens.items():
                    total = summary[name+"."+method]
                    timely = mixed.study.tally(total,s,dict(opened=events))
                    capture_id, complete = None, False
                    if events:
                        at, = events
                        if at not in cache:
                            ci = len(captures)
                            cap = earlier.codec.capture(small,decoder,work,logs,ci,"external",pcm,at,s["command"])
                            uart = logs.read(f"opus/{ci:03}-external.log").decode().replace("\r\n","\n")
                            packets = uplink.packets(uart,ci,cap["emitted"]-cap["first"],0)
                            decoded = uplink.decode(decoder,packets)
                            assert ch.digest(decoded) == cap["lanes"]["selected"]["decoded_sha256"]
                            reading = fresh.prefix.balanced.evaluate(proc,model,decoded,s["command"])
                            cache[at] = ci
                            captures.append(dict(source=s,track=name,opened=at,pcm_sha256=raw["pcm_sha256"],capture=cap,
                                qwen=reading,turbo=uplink.recognition(turbo,decoded,s["command"])))
                        capture_id = cache[at]
                        complete = captures[capture_id]["qwen"]["complete"]
                    total.update(complete=int(complete),timely_complete=int(timely and complete))
                    results[method] = dict(opened=events,timely=timely,capture_id=capture_id,complete=complete)
                rows.append(dict(source=s,track=name,results=results,accepted_current=ts))
            for p in sorted(work.glob("*-attempt*.gz")):
                logs.write(p,"retry/"+p.name)
        assert len(rows)==72
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        controls = {}
        for method in ("baseline","overlap","confirm_0","current"):
            counts = calibrated if method=="current" else old
            groups = {k:counts["summary"][k+"."+method] for k in
                      ("daily.daily","daily.outside","bot.reply","bot.explicit_mention","bot.outside")}
            if method == "baseline":
                groups["background_events"] = background["baseline_events"]
            elif method == "current":
                groups["background_events"] = calibrated["summary"]["background.current"]["events"]
            else:
                groups["background_events"] = background["notifications"][method]
            controls[method] = groups
        ch.save(work/OUT.name,dict(records=rows,captures=captures,summary=summary,existing_controls=controls,
                existing_background_samples=background["total_samples"],archive_sha256=ch.sha(work/archive.name),
                codec=reference["codec"],qwen=qid,turbo=reference["turbo"],model=host["c161"],
                sources={p.name:ch.sha(p) for p in (fresh.PACK,fresh.OUT,previous.score.OUT,previous.fit.FIT,
                        previous.OUT,previous.background.OUT,Path(earlier.codec.__file__),Path(__file__))},
                policy="Unmarked Nyx or removed address, 36 cases each. Reconstruct zero-R FIFO at each distinct "
                       "host INT8 OPEN clock and run P4 Opus in QEMU. Score unprompted complete command suffix. "
                       "False wakes on removed-address controls stay false even when the command is recognized. "
                       "Join unchanged policy results on complete historical daily and Bot recordings, including "
                       "ordinary replies and explicit self-mentions. Reused background exposure."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
