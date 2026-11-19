"""Measure frozen natural and fixed routes on the prospectively recorded continuous command set."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
import connected_nick_willow as study
import replay_aligned_nick as aligned
from score_handoff_willow import willow_events
from score_paired_piper_willow import capture, host, uplink


def main():
    plan = read(study.PLAN)
    assert sha(study.PACK) == plan["audio_sha256"]
    detector = aligned.parent.Detector()
    extra = aligned.Checks()
    reader, identity = host.piper.load_reader()
    decoder, native = uplink.decoder_api(), host.fixed.ch.c161_binding()
    firmware = uplink.firmware()
    rows, totals = [], defaultdict(Counter)
    archive = RESULTS / (study.STEM + "-uart.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        coded = host.fixed.ch.binding(work / "v5.so", coded="v5")
        for src in plan["records"]:
            i, role = src["index"], src["target_id"]
            pcm = host.piper.load_wav(src)
            assert src["sha256"] == sha(Path(src["path"])) and src["pcm_sha256"] == aligned.base.poll.digest(pcm)
            willow = willow_events(i, src, sha(study.PLAN), uart, runs=study.WORK / "runs")
            trace = detector.score(pcm)
            frozen = host.fixed.ch.routes(coded, native, pcm)
            assert trace["c161"] == frozen["c161"]
            kept = [t for t in trace["live_ends"] if trace["checks"][t]["pass"]]
            events, checks = extra.routes(detector.native, pcm, trace["c161"], kept,
                                         {str(t):c for t,c in trace["checks"].items()})
            assert events["parent"] == trace["events"]
            events.update(c161=frozen["c161"], panel=frozen["merged"], willow=willow)
            captures, routes = {}, {}
            for key, ts in events.items():
                result = dict(events=ts, timely=False, complete=False, exact=False)
                if ts:
                    opened = (ts[0]+255)//256*256
                    if opened not in captures:
                        captures[opened] = capture(reader, decoder, work, uart, i, key, pcm, opened, src["carrier"])
                    asr = captures[opened]["lanes"]["selected"]["asr"]
                    result.update(opened=opened, timely=src["start"] <= opened < src["end"],
                                  margin_samples=src["end"]-opened, complete=asr["complete_suffix"], exact=asr["exact"])
                totals[role+"."+key].update(dict(trials=1, hit=bool(ts), timely=result["timely"], complete=result["complete"],
                                                 exact=result["exact"], timely_complete=result["timely"] and result["complete"]))
                routes[key] = result
            board = None
            if role == "fixed-v5" and frozen["merged"]:
                meta = dict(index=i, cue_at=src["start"], command_at=src["end"], carrier=src["carrier"])
                ref = dict(c161=frozen["c161"], fixed=frozen["fixed"], events=dict(nyx=frozen["merged"]))
                board = uplink.p4(meta, pcm, coded, i, decoder, reader, ref, uart, require_timely=False)
                expected = captures[routes["panel"]["opened"]]
                assert board["first"] == expected["first"] and board["delivered"] == routes["panel"]["opened"]
                for lane, data in expected["lanes"].items():
                    assert data["decoded_sha256"] == board["uplink"][lane]["decoded_sha256"]
            rows.append(dict(index=i, role=role, voice=src["voice"], carrier=src["carrier"], end=src["end"],
                             pcm_sha256=src["pcm_sha256"], routes=routes, proposals=trace["proposals"],
                             five_spelling=trace["checks"], extra_checks=checks, captures=captures, board=board))
            if len(rows)%10 == 0:
                print(len(rows), flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for f in failures:
            uart.write(f, "retry/"+f.name)
    archive.with_suffix(".tmp").replace(archive)
    save(RESULTS / (study.STEM+".json"), dict(records=rows, summary=totals, models=detector.identity, asr=identity,
         plan_sha256=sha(study.PLAN), archive_sha256=sha(archive), nick_fit_sha256=sha(aligned.fit.FIT),
         script_sha256=sha(Path(__file__)), retries=[f.name for f in failures], **firmware,
         policy="Actual official DEFAULT Willow returns; frozen C161, v5 and early natural branches in host time. "
                "Round first notification up to the 256-sample listener block. Require it in cue before command "
                "for timely, but transcribe all opened streams. Every distinct opening encoded by P4 QEMU and "
                "decoded on both lanes. Fixed-cue opens additionally pass the complete P4 listener."))


if __name__ == "__main__":
    main()
