"""Send the raw mic at the same 32 measured wake clocks through the board Opus sender."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import score_bella_weak_signal_path as study

scored, prior, P = study.scored, study.prior, study.P
OUT = scored.OUT.with_name("4090-c161-bella-weak-raw-mic-sender.json")
TRACE = scored.trial.WORK / (OUT.stem + "-uart.zip")


def main():
    base, stages, plan = (P.read(p) for p in (scored.OUT, study.OUT, scored.trial.PLAN))
    assert stages["sources"][scored.OUT.name] == P.sha(scored.OUT)
    assert P.sha(scored.sender.BUILD / "board.bin") == base["sources"]["board.bin"]
    assert P.sha(prior.CODEC / "nyx_first_command.bin") == base["sources"]["nyx_first_command.bin"]
    assert P.sha(P.QEMU) == base["p4_qemu_sha256"]
    processor, model, identity = prior.load()
    assert identity == base["qwen"] == stages["qwen"] and identity["device"].endswith("RTX 4090")
    turbo, decoder = prior.reader(base["turbo"]), prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == base["decoder"]
    records = {r["index"]: r for r in stages["records"]}
    controls = {r["label"]: r for r in base["records"]}
    rows, totals = [], defaultdict(Counter)
    with TemporaryDirectory(dir=scored.trial.WORK) as tmp:
        work = Path(tmp)
        scored.clock.reference.WORK = work
        with ZipFile(work / TRACE.name, "w", ZIP_DEFLATED) as logs:
            for number, clip in enumerate(plan["records"]):
                if (clip["target_id"], clip["mode"]) != ("nyx", "target_x2"):
                    continue
                r = records[clip["index"]]
                control = controls[clip["label"]]["uplink"]
                mic, rate = sf.read(clip["path"], dtype="int16")
                assert rate == 16000 and P.digest(mic) == r["mic_sha256"]
                audio = study.window(mic, r["first"], r["stop"])
                digest = P.digest(audio)
                assert digest == r["views"]["mic"]["qwen"]["pcm_sha256"]
                opened, first = control["opened"], control["first"]
                assert first == r["first"] and opened == r["opened"] and len(audio) == control["samples"]
                label = str(r["index"]).zfill(3)
                cold, decoded, codec = scored.clock.opus(work, logs, number, label, audio, decoder)
                payload = struct.pack("<4I", number, opened, first, len(audio)) + audio.tobytes()
                log, ((_, ready), (warm, waited)) = scored.sender.run_case(work, payload, cold)
                logs.write(log, label + "/sender.log")
                for key in ("captured", "first_pcm_sample", "first_send_pts", "first_captured_pts",
                            "sent_silence", "unsent", "replayed_samples", "terminal_padding", "peak_fifo_samples"):
                    assert waited[key] == control["sender"][key]
                views = {}
                for condition, pcm in (("immediate", decoded), ("waited", prior.decode(decoder, warm))):
                    measured = dict(qwen=prior.evaluate(processor, model, pcm, r["command"]),
                                    turbo=prior.recognition(turbo, pcm, r["command"]))
                    assert measured["qwen"]["pcm_sha256"] == P.digest(pcm)
                    views[condition] = measured
                    for reader, key in (("qwen", "complete"), ("turbo", "complete_suffix")):
                        before = control["readings"][condition][reader][key]
                        after = measured[reader][key]
                        totals[condition + "_" + reader].update(cases=1, processed=int(before), raw=int(after),
                                                                 gained=int(after and not before), lost=int(before and not after))
                rows.append(dict(index=r["index"], file=r["file"], command=r["command"], command_at=r["command_at"],
                                 opened=opened, first=first, stop=r["stop"], mic_window_sha256=digest,
                                 processed_decoded_sha256=control["decoded_sha256"], codec=codec, immediate=ready,
                                 waited=waited, readings=views, sender_uart_sha256=P.sha(log),
                                 sender_watchdog=log.read_text().count("Task watchdog got triggered")))
                if len(rows) % 8 == 0:
                    print(len(rows), "of 32", flush=True)
        assert len(rows) == len({r["index"] for r in rows}) == 32
        with ZipFile(work / TRACE.name) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
            for r in rows:
                label = str(r["index"]).zfill(3)
                assert study.scored.sender.archived(z, label + "/sender.log", r["sender_uart_sha256"])
                assert study.scored.sender.archived(z, label + "/opus.log", r["codec"]["opus_uart_sha256"])
        P.save(work / OUT.name, dict(records=rows, counts=totals, archive=TRACE.name, archive_sha256=P.sha(work / TRACE.name),
               qwen=identity, turbo=base["turbo"], decoder=base["decoder"], qemu_sha256=base["p4_qemu_sha256"],
               sources={p.name:P.sha(p) for p in (Path(__file__), study.OUT, scored.OUT, scored.trial.PLAN,
                       scored.sender.BUILD / "board.bin", prior.CODEC / "nyx_first_command.bin")},
               policy="Use the raw digital mic window at each original request, FIFO start and stop of the "
                      "32 consumed target-gain cases. Replace only the PCM supplied to the encoder. Run the "
                      "firmware encoder and sender with immediate readiness and 500/1500 ms connection/readiness. "
                      "Verify immediate packets byte-for-byte and delayed sender timing against the baseline. "
                      "This is a simulation diagnostic with acoustic echo retained in the raw mic."))
        (work / TRACE.name).replace(TRACE)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
