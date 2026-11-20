"""Replay the gain selected by the weak-room speed test with both Mac receiver images."""
import argparse
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_player_gain_margin as trial
import check_bella_plus2_playback as playback
import report_bella_player_gain_margin as selection

P, wakes, fixed = playback.P, playback.wakes, playback.fixed
OUT = trial.PLAN.with_name("4090-bella-gain-candidate-playback.json")


def candidate():
    db = P.read(selection.OUT)["selected_gain_db"]
    assert db in (3, 4, 5)
    original, rate = sf.read(playback.daily.board.bella.CUE, dtype="int16")
    assert rate == 16000 and P.sha(playback.daily.board.bella.CUE) == playback.daily.board.bella.CUE_SHA
    boosted = np.rint(original.astype(float) * 10 ** (db / 20))
    assert np.all((-32768 <= boosted) & (boosted <= 32767))
    cue = boosted.astype("<i2")
    plan = P.read(playback.gain.PLAN)
    assert P.sha(playback.gain.PACK) == plan["archive_sha256"]
    with ZipFile(playback.gain.PACK) as pack:
        for r in plan["records"]:
            if r["role"] != "nyx" or r["level"] != f"plus{db}dB":
                continue
            erased = f"{r['scene']:03}/absent/erased.wav"
            full, absent = (sf.read(BytesIO(pack.read(path)), dtype="int32")[0] // 16384 for path in (r["member"], erased))
            diff = full - absent
            isolated = np.zeros_like(diff)
            isolated[r["end"]-len(cue):r["end"]] = cue
            np.testing.assert_array_equal(diff, isolated)
    return original, cue, db


def replay(kind):
    original, cue, db = candidate()
    meta = P.read(fixed.REPORT)
    image = wakes.fixed_image(meta, kind)
    cache = trial.WORK / ("playback-" + kind) / wakes.digest(bytes.fromhex(meta["qemu_sha256"]) + image[:2<<20])
    cache.mkdir(parents=True, exist_ok=True)
    prior = P.read(playback.OUT)
    baseline = {r["label"]: r for r in prior["records"]}
    assert P.sha(playback.TRACE) == prior["archive_sha256"] and meta["guest_sha256"]["fixed"] == prior["guest_sha256"]
    assert meta["qemu_sha256"] == prior["qemu_sha256"]
    rows = []
    dest = trial.WORK / ("replay-" + kind)
    with TemporaryDirectory(dir=trial.WORK) as folder, ZipFile(playback.TRACE) as old:
        temp = Path(folder)
        with ZipFile(temp / "uart.zip", "w", ZIP_DEFLATED) as logs:
            for group in (playback.external, playback.panel):
                for key, row, mic, ref in group(original, cue):
                    base = baseline[key]
                    block = fixed.run_guest.payload(mic.tobytes(), ref.tobytes())
                    checksum = wakes.digest(block)
                    if row["group"] == "external" and row["role"] == "absent":
                        assert checksum == base["input_sha256"]
                    if row["group"] == "external" and row["role"] == "plus2":
                        row["role"] = f"plus{db}"
                    label = key.replace("/plus2", f"/plus{db}")
                    reused = kind == "fixed" and checksum == base["input_sha256"]
                    faults = {}
                    if reused:
                        raw = old.read(key + ".log")
                        assert wakes.digest(raw) == base["uart_sha256"]
                        events = base["events"]
                    else:
                        for attempt in range(2):
                            try:
                                epochs, log, failed = fixed.boot(cache, image, cache, block, 1)
                                break
                            except AssertionError:
                                if attempt:
                                    raise
                        raw = log.read_bytes()
                        events = epochs[0]["events"]
                        for path in failed:
                            member = f"failed/{label}/{path.name}"
                            logs.write(path, member)
                            faults[member] = P.sha(path)
                    assert b"BOARD PASS\r\n" in raw
                    logs.writestr(label + ".log", raw)
                    row.update(label=label, input_sha256=checksum, mic_sha256=P.digest(mic),
                               reference_sha256=P.digest(ref), events=events, uart_sha256=wakes.digest(raw),
                               reused=reused, failures=faults, plus2_fixed=base["events"])
                    rows.append(row)
                    if len(rows) % 40 == 0:
                        print(kind, len(rows), flush=True)
        assert len(rows) == len({r["label"] for r in rows}) == 200
        P.save(temp / "report.json", dict(records=rows, guest_sha256=meta["guest_sha256"][kind],
               qemu_sha256=meta["qemu_sha256"], archive_sha256=P.sha(temp / "uart.zip")))
        (temp / "uart.zip").replace(dest.with_suffix(".zip"))
        (temp / "report.json").replace(dest.with_suffix(".json"))


def report():
    _, cue, db = candidate()
    wav = OUT.with_name(f"hye-nyx-af-bella-plus{db}db.wav")
    archive = OUT.with_suffix(".zip")
    files, rows = {}, {}
    summary = defaultdict(Counter)
    meta = P.read(fixed.REPORT)
    with TemporaryDirectory(dir=trial.WORK) as folder:
        temp = Path(folder)
        (temp / wav.name).write_bytes(playback.sources.wav(cue, "PCM_16"))
        with ZipFile(temp / archive.name, "w", ZIP_DEFLATED) as out:
            for kind in ("fixed", "dual"):
                base = trial.WORK / ("replay-" + kind)
                doc = P.read(base.with_suffix(".json"))
                assert doc["guest_sha256"] == meta["guest_sha256"][kind] and doc["qemu_sha256"] == meta["qemu_sha256"]
                assert P.sha(base.with_suffix(".zip")) == doc["archive_sha256"]
                with ZipFile(base.with_suffix(".zip")) as logs:
                    for r in doc["records"]:
                        label = r["label"]
                        row = {k: v for k, v in r.items() if k not in ("events", "failures", "reused", "uart_sha256")}
                        if label in rows:
                            assert {k: rows[label][k] for k in row} == row
                        else:
                            rows[label] = row.copy()
                        rows[label][kind] = {k: r[k] for k in ("events", "failures", "reused", "uart_sha256")}
                        members = {label + ".log": r["uart_sha256"], **r["failures"]}
                        for member, checksum in members.items():
                            raw = logs.read(member)
                            assert wakes.digest(raw) == checksum
                            full = kind + "/" + member
                            out.writestr(full, raw)
                            files[full] = checksum
                        events = r["events"]
                        c = summary["/".join((kind, r["group"], r["role"]))]
                        c.update(cases=1, opened=bool(events["O"]), neural=bool(events["M"]), marker=bool(events["C"]))
                        if r["group"] == "external":
                            c["timely"] += bool(events["O"] and r["start"] <= events["O"][0] < r["command_at"])
        sources = (Path(__file__), selection.OUT, playback.gain.PLAN, playback.OUT, playback.TRACE,
                   Path(playback.__file__), trial.PROTOCOL, fixed.REPORT, playback.daily.OUT)
        P.save(temp / OUT.name, dict(records=list(rows.values()), summary=summary, selected_gain_db=db,
               cue=wav.name, cue_sha256=P.sha(temp / wav.name), rate=16000, samples=len(cue),
               sample_peak=int(np.abs(cue).max()), bytes=(temp / wav.name).stat().st_size, pcm_sha256=P.digest(cue),
               files=files, archive_sha256=P.sha(temp / archive.name), guest_sha256=meta["guest_sha256"],
               qemu_sha256=meta["qemu_sha256"], model_sha256=meta["model_sha256"], sources={str(p): P.sha(p) for p in sources}))
        for path in (archive, wav, OUT):
            (temp / path.name).replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fixed", "dual", "report"))
    stage = ap.parse_args().stage
    if stage == "report":
        report()
    else:
        replay(stage)
