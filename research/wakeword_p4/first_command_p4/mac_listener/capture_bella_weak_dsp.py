"""Export AEC alone, AGC alone and the original cascade on the weak-room capture clock."""
import argparse
import json
from pathlib import Path
from shutil import copyfile
import struct
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_daily_overlap as daily
import send_bella_weak_raw_mic as raw
from build_guest import link_guest

P, scored = raw.P, raw.scored
WORK = Path(".local/wakeword/c161-bella-weak-dsp")
OUT = scored.OUT.with_name("4090-c161-bella-weak-dsp-capture.json")
TRACE = WORK / (OUT.stem + "-uart.zip")
PATCH = Path(__file__).with_name("weak-capture-stages.patch").resolve()


def build(idf_python):
    build_dir = scored.clock.BUILD.resolve()
    commands = P.read(build_dir / "compile_commands.json")
    src, = (Path(c["file"]) for c in commands if c["file"].endswith("/listener.cpp"))
    assert P.sha(src) == "db67642a398dd8f1b22a7360c17150cecb0ab2938c6c6086a80d4b46de752c12"
    out = (WORK / "build").resolve()
    out.mkdir(exist_ok=True, parents=True)
    source = out / "listener.cpp"
    copyfile(src, source)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH)], cwd=out, check=True)
    link_guest(build_dir, out, "nyx_mac_listener.elf", [("listener.cpp", source)], [src.parent], idf_python.absolute())
    config = P.read(build_dir / "flasher_args.json")
    files = {k: "board.bin" if v == config["app"]["file"] else str(build_dir / v)
             for k, v in config["flash_files"].items()}
    P.save(out / "flasher_args.json", dict(flash_files=files))
    return out, src


def cases(plan, measured):
    seen = set()
    rooms, _ = daily.rooms()
    weak = {r[0]["file"]: r for r in rooms[-6]}
    for number, clip in enumerate(plan["records"]):
        if (clip["target_id"], clip["mode"]) != ("nyx", "target_x2"):
            continue
        r = measured[clip["label"]]
        mic, rate = sf.read(clip["path"], dtype="int16")
        ref, ref_rate = sf.read(clip["reference_path"], dtype="int16")
        assert ref_rate == rate == 16000 and P.digest(mic) == r["mic_sha256"] and P.digest(ref) == r["ref_sha256"]
        yield number, r, "speech", mic, ref
        if r["file"] not in seen:
            seen.add(r["file"])
            _, _, render, original_ref, _, tracks = weak[r["file"]]
            assert np.array_equal(ref, original_ref)
            panel, clips = render(np.zeros(len(tracks["original"])), include_music=False)
            assert not clips and np.any(panel)
            yield number, r, "panel", panel, ref


def rms(pcm):
    mean_square = np.mean(pcm.astype(np.float64) ** 2)
    return float(10 * np.log10(mean_square / 32768**2)) if mean_square else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python", type=Path, required=True)
    args = parser.parse_args()
    base, plan = P.read(scored.OUT), P.read(scored.trial.PLAN)
    assert P.sha(scored.clock.BUILD / "nyx_mac_listener.bin") == base["sources"]["nyx_mac_listener.bin"]
    assert P.sha(P.QEMU) == base["p4_qemu_sha256"] and P.sha(scored.trial.PLAN) == base["sources"][scored.trial.PLAN.name]
    guest, source = build(args.idf_python)
    prior = {r["label"]: r for r in base["records"]}
    rows = []
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work / TRACE.name, "w", ZIP_DEFLATED) as logs:
            for number, r, group, mic, ref in cases(plan, prior):
                opened = r["opened"]
                first, stop, count = (r["uplink"][key] for key in ("first", "stop", "samples"))
                label = f'{group}/{r["index"]:03}'
                payload = struct.pack("<3I", 0x4b4c434e, number, len(mic)) + mic.tobytes() + ref.tobytes() + struct.pack("<I", opened)
                log = work / "capture.log"
                P.boot(guest, log, work / "flash.bin", b"LISTENER PASS", (2 << 20, payload))
                text = log.read_text()
                assert P.values(text, "READY") == [number, len(mic), 256, 81600]
                assert P.values(text, "OPEN") == [opened, opened//160*160, first]
                assert P.values(text, "FINISH")[:3] == [stop, first, count]
                streams = {name: raw.prior.samples(text, name, count) for name in ("AEC", "AGC", "PROCESSED")}
                streams["raw"] = raw.study.window(mic, first, stop)
                if group == "speech":
                    assert P.digest(streams["PROCESSED"]) == r["uplink"]["fifo_sha256"]
                stats = {name: dict(pcm_sha256=P.digest(v), rms_dbfs=rms(v), post_open_rms_dbfs=rms(v[opened-first:]),
                                    peak=int(np.max(np.abs(v.astype(np.int32))))) for name, v in streams.items()}
                rows.append(dict(label=label, index=r["index"], number=number, file=r["file"], command=r["command"],
                                 opened=opened, first=first, stop=stop, command_at=r["command_at"], streams=stats,
                                 last_nonzero_reference=int(np.flatnonzero(ref)[-1]), mic_sha256=P.digest(mic),
                                 ref_sha256=P.digest(ref), uart_sha256=P.sha(log),
                                 watchdog=text.count("Task watchdog got triggered")))
                logs.write(log, label + ".log")
                if len(rows) % 8 == 0:
                    print("Captured", len(rows), "of 40", flush=True)
        assert len(rows) == 40 and sum(r["label"].startswith("panel/") for r in rows) == 8
        with ZipFile(work / TRACE.name) as z:
            assert z.testzip() is None and len(z.namelist()) == 40
        sources = (Path(__file__), PATCH, source, scored.OUT, scored.trial.PLAN, daily.OUT)
        P.save(work / OUT.name, dict(records=rows, archive=TRACE.name, archive_sha256=P.sha(work / TRACE.name),
               guest_sha256=P.sha(guest/"board.bin"), qemu_sha256=P.sha(P.QEMU), sources={p.name:P.sha(p) for p in sources},
               policy="Original wake clocks in all 32 target-gain cases. Run the frozen capture with a tapped "
                      "AEC output and a second identically configured AGC fed directly from M. Check each full "
                      "cascade against the archived processed FIFO. The first case for each of eight room files "
                      "also supplies a forced capture clock for a panel-only echo control with music and both "
                      "external voices muted. Electrical R is active starting with the block after the open."))
        (work / TRACE.name).replace(TRACE)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
