"""Compare the six weak Nyx commands at the microphone and pre-Opus FIFO."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import soundfile as sf

import score_fixed_rearm_demand_peer as peer

P, scoring = peer.P, peer.scoring
OUT = peer.OUT.with_stem(peer.OUT.stem + "-stages")


def main():
    report = P.read(peer.OUT)
    assert P.sha(peer.TRACE) == report["archive_sha256"]
    proc, model, qid = scoring.prior.load()
    assert qid == report["qwen"]
    turbo = scoring.prior.reader(report["turbo"])
    clean, command = peer.clean_command()
    rows = []
    with ZipFile(peer.TRACE) as z:
        for r in report["records"]:
            if (r["role"], r["variant"]) != ("nyx", "whole"):
                continue
            u = r["uplink"]
            mic, rate = sf.read(r["path"], dtype="int16")
            assert rate == 16000 and P.digest(mic) == r["pcm_sha256"]
            raw = z.read(r["label"]+"/capture.log")
            assert sha256(raw).hexdigest() == u["capture_uart_sha256"]
            fifo = scoring.prior.samples(raw.decode().replace("\r\n","\n"), "PROCESSED", u["samples"])
            assert P.digest(fifo) == u["fifo_sha256"] and u["first"] == 0
            stages = {}
            for name, wave in (("mic",mic), ("fifo",fifo)):
                stages[name] = dict(pcm_sha256=P.digest(wave), samples=len(wave),
                                    qwen=scoring.prior.evaluate(proc,model,wave,command),
                                    turbo=scoring.prior.recognition(turbo,wave,command),
                                    waveform=peer.matching.locate(clean,wave))
            rows.append(dict(scene=r["scene"], label=r["label"], readings=stages))
    assert len(rows) == 6
    P.save(OUT, dict(records=rows, measured_peer_report_sha256=P.sha(peer.OUT), uart_sha256=P.sha(peer.TRACE),
                     script_sha256=P.sha(Path(__file__)), qwen=qid, turbo=report["turbo"]))


if __name__ == "__main__":
    main()
