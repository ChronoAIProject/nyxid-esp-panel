"""Measure each P4 capture path's delay and replay the delay alone on the raw microphones."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import correlate
import soundfile as sf

import capture_bella_weak_dsp as test

P, raw, scored = test.P, test.raw, test.scored
OUT = test.OUT.with_name("4090-c161-bella-weak-dsp-delay.json")


def main():
    capture, stages, plan = (P.read(p) for p in (test.OUT, raw.study.OUT, scored.trial.PLAN))
    guest = test.WORK / "build"
    assert P.sha(guest / "board.bin") == capture["guest_sha256"] and P.sha(P.QEMU) == capture["qemu_sha256"]
    opened, length = 32768, 112000
    probes = []
    with TemporaryDirectory(dir=test.WORK) as folder:
        work = Path(folder)
        archive = OUT.with_suffix(".zip")
        with ZipFile(work / archive.name, "w", ZIP_DEFLATED) as logs:
            for i, amplitude in enumerate((327, 3277)):
                mic = np.zeros(length, dtype="<i2")
                rng = np.random.Generator(np.random.PCG64(161))
                mic[16000:96000] = rng.integers(-amplitude, amplitude+1, size=80000, dtype=np.int16)
                ref = np.zeros_like(mic)
                payload = struct.pack("<3I", 0x4b4c434e, i, length) + mic.tobytes() + ref.tobytes() + struct.pack("<I", opened)
                log = test.WORK / ("probe-" + P.sha(guest/"board.bin")[:12] + "-" + P.digest(mic)[:12] + ".log")
                if not log.exists() or b"LISTENER PASS" not in log.read_bytes():
                    P.boot(guest, log, work/"flash.bin", b"LISTENER PASS", (2 << 20, payload))
                text = log.read_text()
                stop, first, count, *_ = P.values(text, "FINISH")
                assert P.values(text, "OPEN") == [opened, opened//160*160, first] and stop-first == count
                # Compare a two-second interior segment after warm-up at each possible positive delay to 250 ms.
                sample = 48000
                signal = mic[sample:sample+32000].astype(float)
                peak = {}
                for stage in ("AEC", "AGC", "PROCESSED"):
                    pcm = raw.prior.samples(text, stage, count)
                    search = pcm[sample-first:sample-first+len(signal)+4000].astype(float)
                    corr = correlate(search, signal, mode="valid", method="fft")
                    lag = int(np.argmax(corr))
                    rho = float(corr[lag] / (np.linalg.norm(signal) * np.linalg.norm(search[lag:lag+len(signal)])))
                    print(amplitude, stage, lag, rho, flush=True)
                    assert np.isfinite(rho) and 0 <= lag < 4000
                    peak[stage] = dict(lag_samples=lag, correlation=rho)
                logs.write(log, f"{i}.log")
                probes.append(dict(amplitude=amplitude, input_sha256=P.digest(mic), uart_sha256=P.sha(log), stages=peak,
                                   watchdog=text.count("Task watchdog got triggered")))
        delays = {k:v["lag_samples"] for k,v in probes[0]["stages"].items()}
        assert delays == {k:v["lag_samples"] for k,v in probes[1]["stages"].items()}
        processor, model, qid = raw.prior.load()
        assert qid == stages["qwen"] and qid["device"].endswith("RTX 4090")
        turbo = raw.prior.reader(stages["turbo"])
        known = {r["index"]:r for r in stages["records"]}
        totals, rows = defaultdict(Counter), []
        for clip in plan["records"]:
            if (clip["mode"],clip["target_id"]) != ("target_x2","nyx"):
                continue
            case = known[clip["index"]]
            mic, rate = sf.read(clip["path"],dtype="int16")
            assert rate == 16000 and P.digest(mic) == case["mic_sha256"]
            values = {}
            for lag in sorted({0, *delays.values()}):
                pcm = raw.study.window(np.pad(mic, (lag, 0)), case["first"], case["stop"])
                if lag == 0:
                    reading = case["views"]["mic"]
                else:
                    reading = dict(qwen=raw.prior.evaluate(processor, model, pcm, case["command"]),
                                   turbo=raw.prior.recognition(turbo, pcm, case["command"]))
                assert reading["qwen"]["pcm_sha256"] == P.digest(pcm)
                totals[lag].update(cases=1, qwen=int(reading["qwen"]["complete"]), turbo=int(reading["turbo"]["complete_suffix"]))
                values[lag] = reading
            rows.append(dict(index=case["index"], file=case["file"], first=case["first"], stop=case["stop"], readings=values))
        assert len(rows) == 32
        P.save(work / OUT.name,dict(probes=probes, delays=delays, counts=totals, records=rows, qwen=qid, turbo=stages["turbo"],
               archive_sha256=P.sha(work/archive.name), capture_guest_sha256=capture["guest_sha256"], qemu_sha256=P.sha(P.QEMU),
               sources={p.name:P.sha(p) for p in (Path(__file__),test.OUT,raw.study.OUT,scored.trial.PLAN)},
               policy="Two deterministic broadband digital probes on a zero electrical playback reference, fixed "
                      "clock and two peak amplitudes. Measure correlation delay. Shift raw M by each measured "
                      "lag and extract the original 32 FIFO windows. Only those raw controls use a time shift."))
        (work / OUT.name).replace(OUT)
        (work / archive.name).replace(archive)


if __name__ == "__main__":
    main()
