"""Score actual P4 Opus packets at the simulated OPEN clocks of unmarked connected wake trials."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import check_balanced_connected_commands as balanced
import score_connected_overlap as score
import score_paired_piper_willow as codec

ch, mixed, tx = balanced.ch, balanced.mixed, balanced.tx
OUT = ch.RESULTS / (score.fit.STEM+"-commands.json")


def main():
    measurements = ch.read(score.OUT)
    prior = ch.read(balanced.OUT)
    cases = {(r["source"]["index"],r["source"]["external_db"]):r for r in measurements["records"]
             if r["lane"]=="both" and r["source"]["target"]=="Hey Nyx"}
    models = ("baseline", "overlap", "confirm_0", "confirm_180", "confirm_360")
    small, identity = ch.load_reader()
    assert identity == prior["small"]
    turbo = tx.diag.reader(prior["turbo"])
    proc, model, qid = balanced.original.asr.load()
    assert qid == prior["qwen"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == prior["codec"]
    decoder = uplink.decoder_api()
    rows, captures, counts = [], [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        uplink.WORK = work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for s,mic in mixed.study.scenes():
                if s["view"]!="both" or s["target"]!="Hey Nyx":
                    continue
                ref = cases.pop((s["index"],s["external_db"]))
                assert ref["pcm_sha256"] == ch.digest(mic)
                cache, routes = {}, {}
                for arm in models:
                    opens = ref["opened"][arm]
                    timely = bool(opens and s["start"] <= opens[0] < s["command_at"])
                    ci, complete, whisper = None, False, False
                    if opens:
                        at = opens[0]
                        if at not in cache:
                            ci = len(captures)
                            cache[at] = ci
                            cap = codec.capture(small,decoder,work,logs,ci,"external",mic,at,s["command"])
                            uart = logs.read(f"opus/{ci:03}-external.log").decode().replace("\r\n","\n")
                            size = cap["emitted"]-cap["first"]
                            packets = uplink.packets(uart,ci,size,0)
                            decoded = uplink.decode(decoder,packets)
                            selected = cap["lanes"]["selected"]
                            assert ch.digest(decoded)==selected["decoded_sha256"] and len(packets)==selected["packets"]
                            captures.append(dict(source=s, opened=at, pcm_sha256=ref["pcm_sha256"], capture=cap,
                                qwen=balanced.evaluate(proc,model,decoded,s["command"]),
                                turbo=uplink.recognition(turbo,decoded,s["command"])))
                        ci = cache[at]
                        complete = captures[ci]["qwen"]["complete"]
                        whisper = captures[ci]["turbo"]["complete_suffix"]
                    routes[arm] = dict(opened=opens,timely=timely,capture_index=ci,qwen=complete,turbo=whisper)
                    counts[arm].update(trials=1,opened=int(bool(opens)),timely=int(timely),qwen=int(complete),
                                       timely_qwen=int(timely and complete),turbo=int(whisper))
                rows.append(dict(source=s, routes=routes))
            for p in sorted(work.glob("*-attempt*.gz")):
                logs.write(p,"retry/"+p.name)
        assert not cases and len(rows)==36
        with ZipFile(work/archive.name) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        ch.save(work/OUT.name, dict(records=rows,captures=captures,summary=counts,source_sha256=ch.sha(score.OUT),
                archive_sha256=ch.sha(work/archive.name),codec=prior["codec"],qwen=qid,turbo=prior["turbo"],
                small=identity,script_sha256=ch.sha(Path(__file__)),capture_script_sha256=ch.sha(Path(codec.__file__)),policy="Host INT8 proposal and veto clocks. "
                "At each first OPEN reproduce the zero-reference FIFO, including the 90 ms delay and two seconds "
                "of lookback. Run the actual P4 Opus codec in QEMU. All 36 trials in ASR denominators."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
