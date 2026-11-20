"""Repeat only the encoder from each saved FIFO and compare both packet lanes byte for byte."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import bella_delayed_command as trial

P, clock, scoring = trial.P, trial.scoring.clock, trial.scoring
OUT = trial.OUT.with_name("4090-bella-delayed-command-codec-repeat.json")


def main():
    source = P.read(trial.OUT)
    assert P.sha(trial.TRACE) == source["archive_sha256"]
    assert P.sha(P.QEMU) == source["build"]["qemu_sha256"]
    old = P.read(trial.weak.OUT)
    image = scoring.prior.CODEC / "nyx_first_command.bin"
    assert P.sha(image) == old["sources"][str(image)]
    decoder = scoring.prior.decoder_api()
    rows = []
    archive = trial.WORK / "4090-bella-delayed-command-codec-repeat.zip"
    with TemporaryDirectory(dir=trial.WORK) as folder, ZipFile(trial.TRACE) as baseline:
        work = Path(folder)
        clock.reference.WORK = work
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as new:
            for i, r in enumerate(source["records"]):
                u, label = r["uplink"], r["label"]
                capture = scoring.sender.archived(baseline, label+"/capture.log", u["capture_uart_sha256"])
                pcm = scoring.prior.samples(capture, "PROCESSED", u["samples"])
                assert P.digest(pcm) == u["fifo_sha256"]
                original = scoring.sender.archived(baseline, label+"/opus.log", u["opus_uart_sha256"])
                packets, audio, meta = clock.opus(work, new, i, label, pcm, decoder)
                for lane in (0, 1):
                    assert packets == scoring.prior.packets(original, i, len(pcm), lane)
                assert P.digest(audio) == u["decoded_sha256"]
                rows.append(dict(label=label, original_warnings=u["opus_watchdog"], **meta))
        with ZipFile(work/archive.name) as logs:
            members = {n:trial.receiver.test.fixed.digest(logs.read(n)) for n in logs.namelist()}
        P.save(work/OUT.name, dict(records=rows, files=members, all_packets_equal_in_both_lanes=True,
                                   source_sha256=P.sha(trial.OUT), encoder_sha256=P.sha(image), qemu_sha256=P.sha(P.QEMU),
                                   script_sha256=P.sha(Path(__file__)), archive_sha256=P.sha(work/archive.name)))
        (work/OUT.name).replace(OUT)
        (work/archive.name).replace(archive)


if __name__ == "__main__":
    main()
