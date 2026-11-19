"""Cold-boot the frozen P4 listener on each standalone waveform with a silent playback reference."""
import hashlib
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import prepare_reusable_nyx as study
from score_paired_piper_willow import uplink, host

ch = study.ch
OUT = ch.RESULTS / (study.STEM + "-standalone-p4.json")


def main():
    plan = ch.read(study.PLAN)
    fw = uplink.firmware()
    archive = OUT.with_suffix(".zip")
    rows = []
    native = host.fixed.ch.c161_binding()
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        uplink.WORK = work
        detector = host.fixed.ch.binding(work/"v5.so", coded="v5")
        for (row, mic), published in zip(study.inputs(), plan["records"], strict=True):
            assert row == published
            if row["pair"] is not None:
                continue
            i = row["cue"]
            assert i == len(rows)
            pair = np.column_stack((mic, np.zeros_like(mic)))
            payload = struct.pack("<3I", 0x4452594e, i, len(mic)) + pair.T.copy().tobytes()
            path = work/f"cue-{i:02}.log"
            uplink.boot(uplink.BUILD, path, b"LISTENER PASS", payload, work)
            detail, _ = uplink.analyze(path.read_text(), dict(index=i, group="external", command_at=row["cue_end"]),
                                       pair, detector)
            expected = host.fixed.ch.routes(detector, native, mic)
            assert detail["c161"] == expected["c161"]
            assert detail["allowed_v5"] == [e["t"] for e in expected["fixed"]]
            delivered = [study.voices.timing.block(t) for t in expected["merged"]]
            assert detail.get("delivered") == next(iter(delivered), None)
            rows.append(dict(index=i, pcm_sha256=row["pcm_sha256"], uart_sha256=ch.sha(path), result=detail))
            logs.write(path, path.name)
            path.unlink()
            print(i, detail["c161"], detail["v5"], flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for p in retries:
            logs.write(p, "retry/"+p.name)
    assert len(rows) == 16
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for r in rows:
            assert hashlib.sha256(logs.read(f'cue-{r["index"]:02}.log')).hexdigest() == r["uart_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, plan_sha256=ch.sha(study.PLAN), archive_sha256=ch.sha(archive),
                      firmware=fw, script_sha256=ch.sha(Path(__file__)), retries=[p.name for p in retries],
                      policy="Frozen C161 and v5 joint P4 listener in QEMU, 81600 zeros of priming, silent "
                      "electrical channel. Compare streamed C161 and v5 timings and first OPEN with the native "
                      "detectors on each of the 16 standalone clips. Each clip ends at sample 48000 with 2 s of "
                      "following silence. Full emulator UART archived; physical acoustic path unmeasured."))


if __name__ == "__main__":
    main()
