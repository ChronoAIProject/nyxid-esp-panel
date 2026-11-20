"""Compare AEC alone and AGC alone before Opus and on both emitted sender streams."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import capture_bella_weak_dsp as capture

raw, P = capture.raw, capture.P
scored, prior = raw.scored, raw.prior
OUT = scored.OUT.with_name("4090-c161-bella-weak-dsp-sender.json")
TRACE = capture.WORK / (OUT.stem + "-uart.zip")


def main():
    inputs, base, alternate = (P.read(p) for p in (capture.OUT, scored.OUT, raw.OUT))
    assert P.sha(capture.TRACE) == inputs["archive_sha256"]
    assert P.sha(scored.sender.BUILD / "board.bin") == base["sources"]["board.bin"]
    assert P.sha(prior.CODEC / "nyx_first_command.bin") == base["sources"]["nyx_first_command.bin"]
    assert P.sha(P.QEMU) == inputs["qemu_sha256"] == base["p4_qemu_sha256"]
    processor, model, identity = prior.load()
    assert identity == base["qwen"] and identity["device"].endswith("RTX 4090")
    turbo, decoder = prior.reader(base["turbo"]), prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == base["decoder"]
    processed = {r["index"]: r["uplink"] for r in base["records"] if (r["role"], r["mode"]) == ("nyx", "target_x2")}
    bypass = {r["index"]: r for r in alternate["records"]}
    totals, rows = defaultdict(Counter), []

    def read_audio(pcm, command):
        qwen = prior.evaluate(processor, model, pcm, command)
        assert qwen["pcm_sha256"] == P.digest(pcm)
        return dict(qwen=qwen, turbo=prior.recognition(turbo, pcm, command))

    with TemporaryDirectory(dir=capture.WORK) as folder, ZipFile(capture.TRACE) as saved:
        work = Path(folder)
        scored.clock.reference.WORK = work
        with ZipFile(work / TRACE.name, "w", ZIP_DEFLATED) as logs:
            for r in inputs["records"]:
                if not r["label"].startswith("speech/"):
                    continue
                i, number, first, stop = (r[k] for k in ("index", "number", "first", "stop"))
                count = stop - first
                ref = processed[i]
                assert (ref["first"], ref["stop"], ref["opened"]) == (first, stop, r["opened"])
                trace = scored.sender.archived(saved, r["label"] + ".log", r["uart_sha256"])
                stages = {s: prior.samples(trace, s, count) for s in ("AEC", "AGC")}
                for s, pcm in stages.items():
                    assert P.digest(pcm) == r["streams"][s]["pcm_sha256"]
                header = struct.pack("<3I", 0x3552504f, number, count)
                data = header + stages["AEC"].tobytes() + stages["AGC"].tobytes()

                def inspect(uart):
                    for lane in (0, 1):
                        prior.packets(uart, number, count, lane)

                label = f"{i:03}"
                text, opus_sha = scored.clock.reference.verified_boot(work, logs, label + "/opus.log", prior.CODEC,
                                                 "nyx_first_command.bin", b"UPLINK PASS", data, inspect)
                variants = {}
                for lane, (stage, pcm) in enumerate(stages.items()):
                    packets = prior.packets(text, number, count, lane)
                    payload = struct.pack("<4I", number, r["opened"], first, count) + pcm.tobytes()
                    log, ((cold, ready), (warm, waited)) = scored.sender.run_case(work, payload, packets)
                    assert cold == packets and waited == ref["sender"]
                    logs.write(log, label + "/" + stage + "-sender.log")
                    readings = dict(pcm=read_audio(pcm, r["command"]))
                    for condition, sent in (("immediate", cold), ("waited", warm)):
                        readings[condition] = read_audio(prior.decode(decoder, sent), r["command"])
                        for reader, key in (("qwen", "complete"), ("turbo", "complete_suffix")):
                            value = readings[condition][reader][key]
                            original = ref["readings"][condition][reader][key]
                            unprocessed = bypass[i]["readings"][condition][reader][key]
                            totals[".".join((stage, condition, reader))].update(
                                cases=1, success=int(value), cascade=int(original), raw=int(unprocessed),
                                gained_vs_cascade=int(value and not original), lost_vs_cascade=int(original and not value))
                    variants[stage] = dict(readings=readings, fifo_sha256=P.digest(pcm), packets=len(cold),
                                           immediate=ready, waited=waited, sender_uart_sha256=P.sha(log),
                                           sender_watchdog=log.read_text().count("Task watchdog got triggered"))
                rows.append(dict(index=i, number=number, file=r["file"], command=r["command"], first=first, stop=stop,
                                 opened=r["opened"], variants=variants, opus_uart_sha256=opus_sha,
                                 opus_watchdog=text.count("Task watchdog got triggered")))
                if len(rows) % 8 == 0:
                    print(len(rows), "of 32 sender comparisons", flush=True)
        with ZipFile(work / TRACE.name) as logs:
            assert len(rows) == len({r["index"] for r in rows}) == 32 and logs.testzip() is None
            for r in rows:
                label = str(r["index"]).zfill(3)
                scored.sender.archived(logs, label + "/opus.log", r["opus_uart_sha256"])
                for stage, v in r["variants"].items():
                    scored.sender.archived(logs, label + "/" + stage + "-sender.log", v["sender_uart_sha256"])
        P.save(work / OUT.name, dict(records=rows, counts=totals, archive=TRACE.name, archive_sha256=P.sha(work/TRACE.name),
               sources={p.name:P.sha(p) for p in (Path(__file__), capture.TRACE, capture.OUT, scored.OUT, raw.OUT)},
               model=base["c161"], qwen=identity, turbo=base["turbo"], decoder=base["decoder"], qemu_sha256=P.sha(P.QEMU),
               policy="32 consumed weak target-gain cases across eight room files. Two separately initialized "
                      "encoder lanes, then independent sender sessions with exact archived packet checks and "
                      "unchanged readiness clocks. Score both intermediate PCM and the entire decoded send."))
        (work / TRACE.name).replace(TRACE)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
