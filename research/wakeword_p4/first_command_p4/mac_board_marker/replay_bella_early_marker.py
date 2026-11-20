"""Confirm the two-marker playback on historical external and self-play microphones."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf
import bella_early_marker as early

playback = early.playback
P, fixed, wakes = playback.P, playback.fixed, playback.wakes
OUT = early.BASE.with_name(early.BASE.name + "-playback.json")
TRACE = OUT.with_suffix(".zip")


def main():
    selected = P.read(early.REPORT)
    assert all(selected["counts"][k][1]==3 for k in ("nominal", "slow100ppm", "fast100ppm"))
    assert sum(selected["counts"][k][1] for k in ("slow1pct", "fast1pct")) > 0
    original, _, _ = playback.candidate()
    cue, sr = sf.read(early.WAV, dtype="int16")
    assert sr == 16000 and P.sha(early.WAV) == selected["cue_sha256"]
    baseline = P.read(playback.OUT)
    assert P.sha(playback.OUT.with_suffix(".zip")) == baseline["archive_sha256"]
    old = {r["label"]: r for r in baseline["records"]}
    meta = P.read(fixed.REPORT)
    assert meta["qemu_sha256"] == baseline["qemu_sha256"] and meta["guest_sha256"] == baseline["guest_sha256"]
    image = wakes.fixed_image(meta, "dual")
    cache = early.WORK / "historical" / wakes.digest(bytes.fromhex(meta["qemu_sha256"]) + image[:2<<20])
    cache.mkdir(parents=True, exist_ok=True)
    rows, counts, files = [], defaultdict(Counter), {}
    with TemporaryDirectory(dir=early.WORK) as folder, ZipFile(playback.OUT.with_suffix(".zip")) as archive:
        temp = Path(folder)
        with ZipFile(temp / TRACE.name, "w", ZIP_DEFLATED) as logs:
            for generator in (playback.playback.external, playback.playback.panel):
                for key, r, mic, ref in generator(original, cue):
                    anchor = old[key.replace("/plus2", "/plus5")]
                    label = key.replace("/plus2", "/early")
                    if r["role"] == "plus2":
                        r["role"] = "early"
                    payload = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                    checksum = wakes.digest(payload)
                    failed = []
                    if r["role"] == "absent":
                        assert checksum == anchor["input_sha256"]
                        raw = archive.read("dual/" + key + ".log")
                        assert wakes.digest(raw) == anchor["dual"]["uart_sha256"]
                        events = anchor["dual"]["events"]
                    else:
                        for attempt in range(2):
                            try:
                                epochs, log, failed = fixed.boot(cache, image, cache, payload, 1)
                                break
                            except AssertionError:
                                if attempt:
                                    raise
                        events, raw = epochs[0]["events"], log.read_bytes()
                    for fault in failed:
                        member = "failed/" + label + "/" + fault.name
                        logs.write(fault, member)
                        files[member] = P.sha(fault)
                    assert b"BOARD PASS\r\n" in raw
                    member = label + ".log"
                    logs.writestr(member, raw)
                    files[member] = wakes.digest(raw)
                    c = counts[r["group"] + "/" + r["role"]]
                    c.update(cases=1, opened=bool(events["O"]), marker=bool(events["C"]), neural=bool(events["M"]))
                    if r["group"] == "external":
                        c["timely"] += bool(events["O"] and r["start"] <= events["O"][0] < r["command_at"])
                    rows.append(dict(**r, label=label, events=events, original_plus5=anchor["dual"]["events"],
                                     input_sha256=checksum, mic_sha256=P.digest(mic), ref_sha256=P.digest(ref), uart_sha256=files[member]))
                    if len(rows) % 40 == 0:
                        print(len(rows), flush=True)
        assert len(rows)==len({r["label"] for r in rows})==200
        sources = (Path(__file__), early.REPORT, early.PROTOCOL, early.WAV, playback.OUT, Path(playback.playback.__file__))
        P.save(temp/OUT.name, dict(records=rows, counts=counts, files=files, archive_sha256=P.sha(temp/TRACE.name),
               cue_sha256=selected["cue_sha256"], guest_sha256=meta["guest_sha256"]["dual"],
               qemu_sha256=meta["qemu_sha256"], model_sha256=meta["model_sha256"], sources={str(p): P.sha(p) for p in sources}))
        (temp/TRACE.name).replace(TRACE)
        (temp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
