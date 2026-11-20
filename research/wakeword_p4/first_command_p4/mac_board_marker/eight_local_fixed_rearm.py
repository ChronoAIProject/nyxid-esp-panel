"""Run the standalone receiver with the shipped playback recovery logic on the same 100 inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import eight_local_fixed_only as prior
from build_rev1_dev import REFERENCE_BARGE, check_sources
from build_fixed_sound_rev1 import FIXED

P, fixed = prior.P, prior.fixed
WORK = prior.WORK / "rearm"
OUT = prior.OUT.with_stem("4090-eight-local-fixed-rearm")
PLAN = prior.PLAN.with_name("eight_local_fixed_rearm_protocol.md")


def run():
    work = WORK.resolve()
    source = work / "src"
    check_sources(source, extra_patches=(FIXED, *REFERENCE_BARGE))
    flash = fixed.build(source, work / "guest", True, prior.dual.test.IDF)
    history = P.read(prior.OUT)
    assert P.sha(fixed.QEMU) == history["qemu_sha256"]
    guest_sha = P.sha(work / "guest/board.bin")
    code = fixed.digest(flash)
    cache = work / history["qemu_sha256"] / code
    cache.mkdir(parents=True, exist_ok=True)
    previous = prior.indexed(prior.OUT)
    rows, files = [], {}
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=work) as tmp:
        tmp = Path(tmp)
        with ZipFile(tmp / archive.name, "w", ZIP_DEFLATED) as logs:
            for label, mic, ref, _, _ in prior.cases():
                old = previous.pop(label)
                block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                assert fixed.digest(block) == old["input_sha256"]
                epochs = len(old["fixed"])
                for attempt in range(2):
                    try:
                        observed, log, faults = fixed.boot(cache, flash, cache, block, epochs)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                event = observed[0]["events"]
                assert all(r["events"] == event for r in observed) and not event["M"]
                for path in (log, *faults):
                    suffix = "wake.log" if path == log else path.name
                    name = f"{label}/{suffix}"
                    logs.write(path, name)
                    files[name] = P.sha(path)
                rows.append(dict(label=label, input_sha256=old["input_sha256"], epochs=observed,
                             marker_matches_c161=event["C"] == old["c161"][0]["events"]["C"],
                             released_fixed_events=old["fixed"][0]["events"]))
                print(len(rows), label, event, flush=True)
        assert len(rows) == 100 and not previous
        with ZipFile(tmp / archive.name) as logs:
            assert not logs.testzip() and len(logs.namelist()) == len(files)
            assert all(fixed.digest(logs.read(name)) == sha for name, sha in files.items())
        deps = (prior.OUT, PLAN, Path(__file__), Path(prior.__file__), FIXED, *REFERENCE_BARGE)
        P.save(tmp / OUT.name, dict(records=rows, files=files, archive_sha256=P.sha(tmp / archive.name),
               guest_sha256=guest_sha, flash_sha256=code, qemu_sha256=history["qemu_sha256"],
               sources={p.name: P.sha(p) for p in deps}, independent_background_hours=0))
        (tmp / archive.name).replace(archive)
        (tmp / OUT.name).replace(OUT)


if __name__ == "__main__":
    run()
