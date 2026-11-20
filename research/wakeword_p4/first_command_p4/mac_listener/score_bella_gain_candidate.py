"""Read complete commands from the selected cue's simulated P4 uplinks on the RTX 4090."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import replay_bella_gain_candidate as playback
import score_bella_plus2_commands as baseline
import report_bella_player_gain_margin as margin

P, scoring, old_gain = baseline.P, baseline.scoring, baseline.sender
trial = playback.trial
OUT = playback.OUT.with_name("4090-bella-gain-candidate-commands.json")
TRACE = trial.WORK / (OUT.stem + "-uart.zip")


def cases(temp, db, original, cue, reception):
    native = {r["label"]: r for r in reception["records"]}
    for key, r, mic, ref in playback.playback.external(original, cue):
        if r["role"] != "plus2":
            continue
        label = key.replace("/plus2", f"/plus{db}")
        n = native[label]
        assert P.digest(mic) == n["mic_sha256"] and P.digest(ref) == n["reference_sha256"]
        mic_path, ref_path = temp / "mic.wav", temp / "ref.wav"
        sf.write(mic_path, mic, 16000, subtype="PCM_16")
        sf.write(ref_path, ref, 16000, subtype="PCM_16")
        yield dict(label=label, scene=r["scene"], gain_db=db, rate="nominal", command=r["command"],
                   path=mic_path, reference_path=ref_path, pcm_sha256=n["mic_sha256"], reference_sha256=n["reference_sha256"],
                   start=r["start"], end=r["command_at"], opened=next(iter(n["dual"]["events"]["O"]), None))
    for source_db, plan, report in ((2, trial.rates.PLAN, trial.rates.OUT), (db, trial.PLAN, margin.OUT)):
        rows = {r["label"]: r for r in P.read(report)["records"]}
        for r in P.read(plan)["records"]:
            if r["role"] != "nyx" or r["scene"] == "quiet" or r["rate"] not in ("slow100ppm", "fast100ppm"):
                continue
            if r.get("gain_db", 2) != source_db:
                continue
            events = rows[r["label"]]["dual" if source_db == db else "nyx"]["events"]
            yield dict(label=f"rate/{source_db}/{r['label']}", scene=int(r["scene"]), gain_db=source_db, rate=r["rate"],
                       command=r["command"], path=r["path"], reference_path=r["reference_path"], pcm_sha256=r["pcm_sha256"],
                       reference_sha256=r["ref_sha256"], start=r["start"], end=r["end"], opened=next(iter(events["O"]), None))


def main():
    original, cue, db = playback.candidate()
    wake, old, two = (P.read(p) for p in (playback.OUT, old_gain.OUT, baseline.OUT))
    assert P.sha(old_gain.TRACE) == old["archive_sha256"]
    checks = (Path(__file__), Path(scoring.__file__), playback.OUT, margin.OUT, trial.PLAN, trial.rates.OUT, trial.rates.PLAN,
              baseline.OUT, old_gain.OUT, trial.ROOT / "experiments/bella_gain_candidate_commands_protocol.md",
              scoring.clock.BUILD / "nyx_mac_listener.bin", scoring.prior.CODEC / "nyx_first_command.bin", scoring.sender.BUILD / "board.bin")
    sources = {p.name: P.sha(p) for p in checks}
    for name in ("board.bin", "nyx_mac_listener.bin", "nyx_first_command.bin"):
        assert sources[name] == two["sources"][name] == old["sources"][name]
    assert P.sha(P.QEMU) == wake["qemu_sha256"]
    proc, model, qwen = scoring.prior.load()
    decoder = scoring.prior.decoder_api()
    assert qwen == two["qwen"] == old["qwen"] and "4090" in qwen["device"]
    assert old["turbo"] == two["turbo"] and decoder.opus_get_version_string().decode() == old["decoder"] == two["decoder"]
    readers = proc, model, scoring.prior.reader(old["turbo"]), decoder
    anchors = {r["scene"]: r for r in old["records"] if r["role"] == "nyx" and r["gain_db"] == db}
    assert len(anchors) == 3
    cache = trial.WORK / ("sender-" + sources[playback.OUT.name][:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, files = [], {}
    counts = defaultdict(Counter)
    two_nominal = {r["index"]: r for r in two["records"]}
    with TemporaryDirectory(dir=trial.WORK) as folder, ZipFile(old_gain.TRACE) as historical:
        temp = Path(folder)
        with ZipFile(temp / TRACE.name, "w", ZIP_DEFLATED) as logs:
            for number, r in enumerate(cases(temp, db, original, cue, wake)):
                row = {k: v for k, v in r.items() if k not in ("path", "reference_path")}
                row.update(readings={}, uart={}, reused=False)
                scene, opened, label = r["scene"], r["opened"], r["label"]
                if opened is not None:
                    if scene in anchors and r["rate"] == "nominal":
                        read = anchors[scene]
                        assert (r["pcm_sha256"], r["reference_sha256"], opened) == (
                            read["pcm_sha256"], read["reference_sha256"], read["opened"])
                        evidence = {Path(k).name: historical.read(k) for k in read["uart"]}
                        for k, expected in read["uart"].items():
                            assert playback.wakes.digest(evidence[Path(k).name]) == expected
                        row["reused"] = True
                    else:
                        read, path = scoring.score_case(r, number, opened, readers, temp, cache)
                        with ZipFile(path) as shard:
                            evidence = {}
                            for stage in ("capture", "opus", "sender"):
                                raw = shard.read(f"{label}/{stage}.log")
                                assert playback.wakes.digest(raw) == read[stage + "_uart_sha256"] and read[stage + "_watchdog"] == 0
                                evidence[stage + ".log"] = raw
                    for name, raw in evidence.items():
                        member = label + "/" + name
                        logs.writestr(member, raw)
                        files[member] = row["uart"][member] = playback.wakes.digest(raw)
                    row.update({k: read[k] for k in ("first", "stop", "samples", "fifo_sha256", "readings")})
                c = counts[f"{r['gain_db']}/{r['rate']}"]
                c.update(cases=1, opened=opened is not None)
                for policy in ("immediate", "waited"):
                    for reader, field in (("qwen", "complete"), ("turbo", "complete_suffix")):
                        score = policy + "_" + reader
                        passed = bool(row["readings"] and row["readings"][policy][reader][field])
                        c[score] += int(passed)
                        if r["rate"] == "nominal":
                            ref = two_nominal[scene]
                            assert ref["command"] == r["command"] and ref["command_at"] == r["end"]
                            before = bool(ref["readings"] and ref["readings"][policy][reader][field])
                            c[score + "_plus2"] += int(before)
                            c[score + "_gained"] += int(passed and not before)
                            c[score + "_lost"] += int(before and not passed)
                rows.append(row)
                if len(rows) % 8 == 0:
                    print(len(rows), flush=True)
        assert len(rows) == len({r["label"] for r in rows}) == 108
        assert len(files) == 3 * sum(bool(r["uart"]) for r in rows)
        P.save(temp / OUT.name, dict(records=rows, summary=counts, sources=sources, files=files, qwen=qwen, turbo=old["turbo"],
               decoder=old["decoder"], archive=TRACE.name, archive_sha256=P.sha(temp / TRACE.name),
               qemu_sha256=wake["qemu_sha256"], cue_sha256=wake["cue_sha256"],
               scope="96 nominal interruptions at the selected level and 12 small-offset playbacks across both levels in three selection rooms."))
        (temp / TRACE.name).replace(TRACE)
        (temp / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
